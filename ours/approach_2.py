"""
Memory Bank Anomaly Detection with Lighting Augmentation
=========================================================
PatchCore-style memory bank built from normal training images, augmented
with physics-based lighting perturbations to capture lighting invariance
without contrastive training.

Pipeline:
  Offline (per class):
    for each I in train/good:
        bank.add( DINOv2(I) )
        for k in range(K):
            bank.add( DINOv2(physics_augment(I)) )
    bank = greedy_coreset(bank, bank_size)

  Online (per test image I):
    f         = DINOv2(I)                    # [N_patches, C]
    dist      = nearest_neighbor(f, bank)    # [N_patches]
    amap      = upsample + gaussian_smooth(dist.reshape(H_p, W_p))
    img_score = top_k_mean(amap, ratio=image_score_topk)

Metric accumulation follows MVTec convention (approach_1_v2.py): collect
raw maps across all test images, compute once at end.

Usage (one class):
  python approach_2.py \
    --train_dir  C:/anomaly_detection/data/mvtec_ad_2/can/train/good \
    --image_dir  C:/anomaly_detection/data/mvtec_ad_2/can/test_public \
    --mask_dir   C:/anomaly_detection/data/mvtec_ad_2/can/test_public/ground_truth/bad \
    --output_dir ./results/can_v2 \
    --K 8 --bank_size 5000
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
DINO_STD = torch.tensor([0.229, 0.224, 0.225])
DINOV2_PATCH = 14


# ─────────────────────────────────────────────────────────────────────────────
# Physics-based lighting augmentation
# ─────────────────────────────────────────────────────────────────────────────

def aug_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    return np.clip(img ** gamma, 0.0, 1.0).astype(np.float32)


def aug_color_temp(img: np.ndarray, t: float) -> np.ndarray:
    """t in [-1, 1]: negative = cool (blue), positive = warm (red)."""
    mult = np.array([1.0 + 0.35 * t, 1.0, 1.0 - 0.35 * t], dtype=np.float32)
    return np.clip(img * mult, 0.0, 1.0).astype(np.float32)


def aug_vignette(img: np.ndarray, strength: float,
                 rng: np.random.Generator) -> np.ndarray:
    H, W = img.shape[:2]
    cy = H / 2.0 + rng.uniform(-H * 0.1, H * 0.1)
    cx = W / 2.0 + rng.uniform(-W * 0.1, W * 0.1)
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / (np.sqrt(H * H + W * W) / 2.0)
    falloff = np.clip(1.0 - strength * dist ** 2, 0.0, 1.0).astype(np.float32)
    return np.clip(img * falloff[..., None], 0.0, 1.0).astype(np.float32)


def aug_directional_gradient(img: np.ndarray, strength: float,
                              rng: np.random.Generator) -> np.ndarray:
    H, W = img.shape[:2]
    angle = rng.uniform(0.0, 2.0 * np.pi)
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W),
                         indexing="ij")
    g = np.cos(angle) * xx + np.sin(angle) * yy
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    mult = (1.0 - strength) + strength * g
    return np.clip(img * mult[..., None], 0.0, 1.0).astype(np.float32)


def aug_exposure(img: np.ndarray, mult: float) -> np.ndarray:
    return np.clip(img * mult, 0.0, 1.0).astype(np.float32)


def physics_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random composition of physics-based lighting perturbations.

    All transforms are multiplicative / global so the underlying texture
    (and thus any anomaly signal) is preserved. Augmentations that could
    fake a defect (additive noise, local patches, blur) are deliberately
    excluded.
    """
    out = img
    out = aug_gamma(out, float(rng.uniform(0.4, 2.2)))
    out = aug_color_temp(out, float(rng.uniform(-0.6, 0.6)))
    if rng.random() < 0.5:
        out = aug_vignette(out, float(rng.uniform(0.2, 0.6)), rng)
    if rng.random() < 0.5:
        out = aug_directional_gradient(out, float(rng.uniform(0.2, 0.5)), rng)
    out = aug_exposure(out, float(rng.uniform(0.55, 1.6)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Letterbox (reused from approach_1_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

PadInfo = Tuple[int, int, int, int]   # (pad_top, pad_left, new_h, new_w)


def letterbox_resize(img: Image.Image, target: int) -> Tuple[np.ndarray, PadInfo]:
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(w * scale), int(h * scale)
    arr = np.array(img.resize((nw, nh), Image.BILINEAR)).astype(np.float32) / 255.0
    pt = (target - nh) // 2
    pl = (target - nw) // 2
    canvas = np.full((target, target, 3), float(arr.mean()), dtype=np.float32)
    canvas[pt:pt + nh, pl:pl + nw] = arr
    return canvas, (pt, pl, nh, nw)


def letterbox_mask(mask: Image.Image, target: int, pad: PadInfo) -> np.ndarray:
    pt, pl, nh, nw = pad
    arr = (np.array(mask.resize((nw, nh), Image.NEAREST)) > 127).astype(np.uint8)
    canvas = np.zeros((target, target), dtype=np.uint8)
    canvas[pt:pt + nh, pl:pl + nw] = arr
    return canvas


def unpad(amap: np.ndarray, pad: PadInfo, orig_w: int, orig_h: int) -> np.ndarray:
    pt, pl, nh, nw = pad
    return cv2.resize(amap[pt:pt + nh, pl:pl + nw],
                      (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_dinov2(name: str, device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", name)
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_patch_features(model, img_np: np.ndarray, device: torch.device,
                            layer_indices: Optional[List[int]] = None) -> torch.Tensor:
    """img_np: [H, W, 3] float32 in [0, 1]. Returns [N_patches, C] on CPU."""
    x = torch.from_numpy(img_np).permute(2, 0, 1).float()
    x = ((x - DINO_MEAN.view(3, 1, 1)) / DINO_STD.view(3, 1, 1)).unsqueeze(0).to(device)

    if layer_indices is not None:
        feats: List[torch.Tensor] = []

        def hook(_module, _inputs, output):
            feats.append(output[:, 1:, :].squeeze(0).cpu())

        handles = [model.blocks[idx].register_forward_hook(hook)
                   for idx in layer_indices]
        _ = model(x)
        for h in handles:
            h.remove()
        return torch.cat(feats, dim=-1).float()

    out = model.forward_features(x)
    return out["x_norm_patchtokens"].squeeze(0).cpu().float()


# ─────────────────────────────────────────────────────────────────────────────
# Greedy farthest-point coreset subsampling
# ─────────────────────────────────────────────────────────────────────────────

def greedy_coreset(features: torch.Tensor, n_select: int,
                    device: torch.device, seed: int = 0) -> torch.Tensor:
    N = features.shape[0]
    n_select = min(n_select, N)
    if n_select == N:
        return features

    feats_gpu = features.to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx0 = int(torch.randint(0, N, (1,), generator=g).item())

    selected: List[int] = [idx0]
    min_dist = torch.norm(feats_gpu - feats_gpu[idx0:idx0 + 1], dim=1)

    for _ in tqdm(range(n_select - 1), desc="  coreset", leave=False):
        idx = int(min_dist.argmax().item())
        selected.append(idx)
        d = torch.norm(feats_gpu - feats_gpu[idx:idx + 1], dim=1)
        min_dist = torch.minimum(min_dist, d)

    return features[selected]


# ─────────────────────────────────────────────────────────────────────────────
# Memory bank
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBank:
    def __init__(self,
                 model,
                 device: torch.device,
                 K: int,
                 bank_size: int,
                 resolution: int,
                 layer_indices: Optional[List[int]] = None,
                 max_pre_coreset: int = 200_000,
                 seed: int = 42):
        self.model = model
        self.device = device
        self.K = K
        self.bank_size = bank_size
        self.resolution = resolution
        self.layer_indices = layer_indices
        self.max_pre_coreset = max_pre_coreset
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.bank: Optional[torch.Tensor] = None

    def _feat(self, img_np: np.ndarray) -> torch.Tensor:
        return extract_patch_features(self.model, img_np, self.device,
                                      self.layer_indices)

    def fit(self, normal_image_paths: List[Path]) -> None:
        running: List[torch.Tensor] = []
        bank: Optional[torch.Tensor] = None
        target_interim = max(self.bank_size * 10, self.bank_size)

        def flush(bank, running):
            if not running:
                return bank
            new = torch.cat(running, dim=0)
            return new if bank is None else torch.cat([bank, new], dim=0)

        total_raw = 0
        for p in tqdm(normal_image_paths, desc="Bank build"):
            img_pil = Image.open(p).convert("RGB")
            img_lb, _ = letterbox_resize(img_pil, self.resolution)
            running.append(self._feat(img_lb))
            for _ in range(self.K):
                aug = physics_augment(img_lb, self.rng)
                running.append(self._feat(aug))

            cur = sum(f.shape[0] for f in running) + (bank.shape[0] if bank is not None else 0)
            if cur >= self.max_pre_coreset:
                bank = flush(bank, running)
                running = []
                total_raw += bank.shape[0] if bank is not None else 0
                if bank.shape[0] > target_interim:
                    bank = greedy_coreset(bank, target_interim, self.device,
                                          seed=self.seed)

        bank = flush(bank, running)
        running = []
        assert bank is not None, "no features extracted"

        n_keep = min(self.bank_size, bank.shape[0])
        print(f"  Pre-coreset patches: {bank.shape[0]:,}  →  keeping {n_keep:,}")
        self.bank = greedy_coreset(bank, n_keep, self.device, seed=self.seed).cpu()
        print(f"  Bank ready: shape={tuple(self.bank.shape)}")

    @torch.no_grad()
    def score_patches(self, img_np: np.ndarray) -> np.ndarray:
        """Returns [N_patches] NN distance per patch."""
        assert self.bank is not None, "bank not built"
        f = self._feat(img_np).to(self.device)
        bank_gpu = self.bank.to(self.device)

        chunk = 4096
        out: List[torch.Tensor] = []
        for i in range(0, f.shape[0], chunk):
            d = torch.cdist(f[i:i + chunk], bank_gpu)
            out.append(d.min(dim=1).values.cpu())
        return torch.cat(out, dim=0).numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "bank": self.bank,
            "K": self.K,
            "bank_size": self.bank_size,
            "resolution": self.resolution,
            "layer_indices": self.layer_indices,
        }, path)

    def load(self, path: Path) -> None:
        d = torch.load(path, map_location="cpu")
        self.bank = d["bank"]
        self.K = d["K"]
        self.bank_size = d["bank_size"]
        self.resolution = d["resolution"]
        self.layer_indices = d["layer_indices"]


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def gather_images(d: Path) -> List[Path]:
    return sorted([p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def find_mask(stem: str, mask_dir: Optional[Path]) -> Optional[Path]:
    if not mask_dir:
        return None
    for c in (mask_dir / f"{stem}_mask.png",
              mask_dir / f"{stem}.png",
              mask_dir / f"{stem}_mask.jpg"):
        if c.exists():
            return c
    return None


def infer_one(image_path: Path,
              mask_path: Optional[Path],
              bank: MemoryBank,
              args) -> Tuple[np.ndarray, np.ndarray, int, float, np.ndarray, Tuple[int, int]]:
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    img_lb, pad = letterbox_resize(img_pil, args.resolution)

    gt_label = 0 if "good" in [p.lower() for p in image_path.parts] else 1
    if mask_path and mask_path.exists():
        gt_mask = letterbox_mask(Image.open(mask_path).convert("L"),
                                 args.resolution, pad)
    else:
        gt_mask = np.zeros((args.resolution, args.resolution), dtype=np.uint8)

    dist = bank.score_patches(img_lb)               # [N_patches]
    side = int(np.sqrt(dist.size))
    assert side * side == dist.size, f"non-square patch grid: N={dist.size}"
    score_map = dist.reshape(side, side)

    amap = cv2.resize(score_map, (args.resolution, args.resolution),
                      interpolation=cv2.INTER_LINEAR)
    amap = gaussian_filter(amap, sigma=args.sigma).astype(np.float32)

    # image score on RAW amap (NN distances comparable across images)
    flat = amap.flatten()
    k = max(1, int(flat.size * args.image_score_topk))
    image_score = float(np.partition(flat, -k)[-k:].mean())

    # heatmap visual gets per-image normalization
    lo, hi = amap.min(), amap.max()
    amap_vis = (amap - lo) / (hi - lo + 1e-8)
    amap_orig = unpad(amap_vis, pad, orig_w, orig_h)

    return amap, gt_mask, gt_label, image_score, amap_orig, (orig_w, orig_h)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (same convention as approach_1_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_metrics(all_amaps, all_masks, all_image_scores, all_gt_labels):
    from torchmetrics.classification import BinaryAUROC
    from anomalib.metrics.aupro import _AUPRO

    results = {}
    all_pred = np.concatenate([a.flatten() for a in all_amaps])
    all_mask = np.concatenate([m.flatten() for m in all_masks])
    pred_t = torch.from_numpy(all_pred).float()
    mask_t = torch.from_numpy(all_mask).long()

    if len(np.unique(all_mask)) >= 2:
        results["pixel_auroc"] = BinaryAUROC()(pred_t, mask_t).item()
    else:
        results["pixel_auroc"] = float("nan")
        print("  [warn] no positive pixels → pixel_auroc skipped")

    for fpr_limit, key in [(0.05, "aupro_05"), (0.30, "aupro_30")]:
        try:
            au = _AUPRO(fpr_limit=fpr_limit)
            for amap, m in zip(all_amaps, all_masks):
                au.update(torch.from_numpy(amap).float().unsqueeze(0),
                          torch.from_numpy(m).long().unsqueeze(0))
            results[key] = au.compute().item()
        except Exception as e:
            results[key] = float("nan")
            print(f"  [warn] {key}: {e}")

    if len(set(all_gt_labels)) >= 2:
        results["image_auroc"] = BinaryAUROC()(
            torch.tensor(all_image_scores).float(),
            torch.tensor(all_gt_labels).long(),
        ).item()
    else:
        results["image_auroc"] = float("nan")
        print("  [warn] only one class in labels → image_auroc skipped")
    return results


def save_heatmap(amap: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = float(amap.min()), float(amap.max())
    amap = (amap - lo) / (hi - lo + 1e-8)
    cv2.imwrite(str(path), cv2.applyColorMap(
        (amap * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    if args.resolution % DINOV2_PATCH != 0:
        raise SystemExit(f"--resolution ({args.resolution}) must be a multiple "
                         f"of DINOv2 patch size ({DINOV2_PATCH})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = Path(args.train_dir)
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = gather_images(train_dir)
    test_paths = gather_images(image_dir)

    print(f"Device       : {device}")
    print(f"Train dir    : {train_dir}  ({len(train_paths)} images)")
    print(f"Image dir    : {image_dir}  ({len(test_paths)} images)")
    print(f"Mask dir     : {mask_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Backbone     : {args.backbone}")
    print(f"Resolution   : {args.resolution}")
    print(f"K            : {args.K}  (augmentations per training image)")
    print(f"Bank size    : {args.bank_size}")
    print(f"Layer indices: {args.layer_indices}")
    print(f"Score topk   : {args.image_score_topk}")
    print(f"Sigma        : {args.sigma}")

    if not train_paths:
        raise SystemExit(f"[ERROR] no training images in {train_dir}")
    if not test_paths:
        raise SystemExit(f"[ERROR] no test images in {image_dir}")

    print(f"\nLoading DINOv2 ({args.backbone}) ...")
    model = load_dinov2(args.backbone, device)

    bank = MemoryBank(model, device,
                      K=args.K,
                      bank_size=args.bank_size,
                      resolution=args.resolution,
                      layer_indices=args.layer_indices if args.layer_indices else None,
                      seed=args.seed)

    bank_path = output_dir / "bank.pth"
    if bank_path.exists() and not args.rebuild_bank:
        print(f"\nLoading cached bank: {bank_path}")
        bank.load(bank_path)
        print(f"  shape={tuple(bank.bank.shape)}")
    else:
        print(f"\nBuilding memory bank ...")
        bank.fit(train_paths)
        bank.save(bank_path)
        print(f"  saved → {bank_path}")

    all_amaps: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_scores: List[float] = []
    all_labels: List[int] = []
    per_image_rows: List[dict] = []

    print(f"\nScoring test images ...")
    for p in tqdm(test_paths):
        mask = find_mask(p.stem, mask_dir)
        try:
            amap, gt_mask, gt_label, image_score, amap_orig, (ow, oh) = \
                infer_one(p, mask, bank, args)
            hp = output_dir / f"{p.stem}_heatmap.png"
            save_heatmap(amap_orig, hp)
            all_amaps.append(amap)
            all_masks.append(gt_mask)
            all_scores.append(image_score)
            all_labels.append(gt_label)
            per_image_rows.append({
                "image": str(p),
                "orig_size": f"{ow}x{oh}",
                "gt_label": gt_label,
                "image_score": image_score,
                "heatmap": str(hp),
            })
        except Exception as e:
            print(f"  [ERROR] {p.name}: {e}")

    if not all_amaps:
        raise SystemExit("No results.")

    print(f"\nComputing global metrics ...")
    metrics = compute_global_metrics(all_amaps, all_masks, all_scores, all_labels)

    print("\n" + "=" * 60)
    print("RESULTS — Memory Bank + Lighting Augmentation")
    print("=" * 60)
    print(f"  Images processed : {len(all_amaps)}")
    print(f"  Image AUROC      : {metrics.get('image_auroc', float('nan')) * 100:.2f}%")
    print(f"  Pixel AUROC      : {metrics.get('pixel_auroc', float('nan')) * 100:.2f}%")
    print(f"  AUPRO @ 0.05     : {metrics.get('aupro_05',   float('nan')) * 100:.2f}%")
    print(f"  AUPRO @ 0.30     : {metrics.get('aupro_30',   float('nan')) * 100:.2f}%")
    print("=" * 60)

    fieldnames = ["image", "orig_size", "gt_label", "image_score", "heatmap"]
    with open(output_dir / "results_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in per_image_rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fieldnames if k in r})
    with open(output_dir / "results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_images", len(all_amaps)])
        w.writerow(["image_auroc", f"{metrics.get('image_auroc', float('nan')):.6f}"])
        w.writerow(["pixel_auroc", f"{metrics.get('pixel_auroc', float('nan')):.6f}"])
        w.writerow(["aupro_05", f"{metrics.get('aupro_05', float('nan')):.6f}"])
        w.writerow(["aupro_30", f"{metrics.get('aupro_30', float('nan')):.6f}"])
        w.writerow(["backbone", args.backbone])
        w.writerow(["resolution", args.resolution])
        w.writerow(["K", args.K])
        w.writerow(["bank_size", args.bank_size])
        w.writerow(["layer_indices", str(args.layer_indices)])
        w.writerow(["sigma", args.sigma])
        w.writerow(["image_score_topk", args.image_score_topk])
    print(f"\nPer-image CSV : {output_dir / 'results_per_image.csv'}")
    print(f"Summary CSV   : {output_dir / 'results_summary.csv'}")
    print(f"Bank          : {bank_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Memory Bank + Lighting Augmentation Anomaly Detection (DINOv2)")
    p.add_argument("--train_dir", required=True,
                   help="Directory of normal training images (recursive)")
    p.add_argument("--image_dir", required=True,
                   help="Test image directory (recursive)")
    p.add_argument("--mask_dir", default=None,
                   help="Directory containing GT masks named <stem>_mask.png")
    p.add_argument("--output_dir", default="./results_approach_2")

    p.add_argument("--resolution", type=int, default=518,
                   help="Letterbox target resolution (must be multiple of 14)")
    p.add_argument("--backbone", default="dinov2_vitl14",
                   choices=["dinov2_vits14", "dinov2_vitb14",
                            "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--layer_indices", type=int, nargs="+", default=None,
                   help="DINOv2 block indices to concat features from "
                        "(default: last-layer x_norm_patchtokens). "
                        "Concat layers multiplies feature dim → memory cost.")

    p.add_argument("--K", type=int, default=8,
                   help="Number of physics-based augmentations per training image. "
                        "Set 0 to disable augmentation (ablation).")
    p.add_argument("--bank_size", type=int, default=5000,
                   help="Final coreset bank size")
    p.add_argument("--rebuild_bank", action="store_true",
                   help="Force rebuild bank even if cached on disk")

    p.add_argument("--sigma", type=float, default=4.0,
                   help="Gaussian smoothing sigma applied to anomaly map")
    p.add_argument("--image_score_topk", type=float, default=0.001,
                   help="Fraction of top pixels averaged for image-level score "
                        "(more stable than p95)")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
