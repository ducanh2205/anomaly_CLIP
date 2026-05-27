"""
SVD-based Zero-Shot Anomaly Detection — Batch Folder Evaluation
================================================================
Metrics tính theo chuẩn MVTec:
  - Pixel AUROC  : accumulate toàn bộ pixel predictions → tính 1 lần cuối
  - AUPRO@0.05/0.30 : accumulate toàn bộ ảnh → compute 1 lần cuối
  - Image AUROC  : accumulate image-level scores → tính 1 lần cuối

Naming convention:
  test image  : <stem>.png
  mask file   : <stem>_mask.png   (trong mask_dir)
  heatmap out : <stem>_heatmap.png (trong output_dir)

Usage:
  python svd_zsad_batch.py \
    --image_dir  path/to/test_images \
    --mask_dir   path/to/ground_truth \
    --output_dir ./results
"""

import argparse
import csv
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Retinex
# ─────────────────────────────────────────────────────────────────────────────

def multi_scale_retinex(img_np: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    img_log = np.log1p(img_np * 255.0)
    msr = np.zeros_like(img_log)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img_log, (0, 0), sigma)
        msr += img_log - blur
    msr /= len(sigmas)
    for c in range(3):
        ch = msr[..., c]
        msr[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return msr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Letterbox
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_resize(img: Image.Image, target_size: int):
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    arr = np.array(img.resize((new_w, new_h), Image.BILINEAR)).astype(np.float32) / 255.0
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas = np.full((target_size, target_size, 3), arr.mean(), dtype=np.float32)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = arr
    return canvas, (pad_top, pad_left, new_h, new_w)


def letterbox_mask(mask: Image.Image, target_size: int, pad_info: tuple) -> np.ndarray:
    pad_top, pad_left, new_h, new_w = pad_info
    arr = (np.array(mask.resize((new_w, new_h), Image.NEAREST)) > 127).astype(np.uint8)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = arr
    return canvas


def unpad_map(amap: np.ndarray, pad_info: tuple, orig_w: int, orig_h: int) -> np.ndarray:
    pad_top, pad_left, new_h, new_w = pad_info
    return cv2.resize(amap[pad_top:pad_top+new_h, pad_left:pad_left+new_w],
                      (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2
# ─────────────────────────────────────────────────────────────────────────────

def load_dinov2(model_name: str = "dinov2_vitl14", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model, device


DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
DINO_STD  = torch.tensor([0.229, 0.224, 0.225])


@torch.no_grad()
def extract_patch_features(model, img_np: np.ndarray,
                            device: torch.device,
                            layer_indices: list = None) -> torch.Tensor:
    x = torch.from_numpy(img_np).permute(2, 0, 1).float()
    x = ((x - DINO_MEAN.view(3,1,1)) / DINO_STD.view(3,1,1)).unsqueeze(0).to(device)

    if layer_indices is not None:
        features_list = []
        def hook_fn(module, input, output):
            features_list.append(output[:, 1:, :].squeeze(0).cpu())
        handles = [model.blocks[idx].register_forward_hook(hook_fn)
                   for idx in layer_indices]
        _ = model(x)
        for h in handles:
            h.remove()
        return torch.cat(features_list, dim=-1).float()
    else:
        out = model.forward_features(x)
        return out["x_norm_patchtokens"].squeeze(0).cpu().float()


# ─────────────────────────────────────────────────────────────────────────────
# SVD anomaly scoring
# ─────────────────────────────────────────────────────────────────────────────

def svd_anomaly_score(features: torch.Tensor, k: int) -> torch.Tensor:
    F_c = features - features.mean(dim=0, keepdim=True)
    try:
        _, _, Vt = torch.linalg.svd(F_c, full_matrices=False)
    except Exception:
        _, _, Vt = torch.svd(F_c); Vt = Vt.T
    k = min(k, Vt.shape[0])
    Vt_k = Vt[:k]
    F_proj = F_c @ Vt_k.T @ Vt_k
    return ((F_c - F_proj) ** 2).sum(dim=-1)


def _gaussian_weight(size: int) -> torch.Tensor:
    if size == 1:
        return torch.ones(1, 1)
    sigma = size / 4.0
    coords = torch.arange(size).float() - (size - 1) / 2.0
    g1d = torch.exp(-coords**2 / (2 * sigma**2))
    g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
    return g2d / g2d.max()


def multi_scale_svd_score(model, img_np: np.ndarray, device: torch.device,
                           scales: list, k_ratio: float = 0.1,
                           layer_indices: list = None) -> np.ndarray:
    H, W, _ = img_np.shape
    patch_size = 14
    H_p, W_p = H // patch_size, W // patch_size

    features    = extract_patch_features(model, img_np, device, layer_indices)
    features_2d = features.view(H_p, W_p, -1)
    score_map   = torch.zeros(H_p, W_p)
    weight_map  = torch.zeros(H_p, W_p)

    for scale in scales:
        stride = max(1, scale // 2)
        for i in range(0, H_p - scale + 1, stride):
            for j in range(0, W_p - scale + 1, stride):
                f = features_2d[i:i+scale, j:j+scale].reshape(scale*scale, -1)
                k = max(1, int(scale*scale * k_ratio))
                scores_2d = svd_anomaly_score(f, k).view(scale, scale)
                gauss = _gaussian_weight(scale)
                score_map[i:i+scale, j:j+scale]  += scores_2d * gauss
                weight_map[i:i+scale, j:j+scale] += gauss

    return (score_map / (weight_map + 1e-8)).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def save_heatmap(amap: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = amap.min(), amap.max()
    if hi > lo:
        amap = (amap - lo) / (hi - lo)
    cv2.imwrite(str(path), cv2.applyColorMap(
        (amap * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET))


# ─────────────────────────────────────────────────────────────────────────────
# Per-image inference  (returns raw amap + metadata, NO metric computation)
# ─────────────────────────────────────────────────────────────────────────────

def infer_one(image_path: Path, mask_path, model, device, args):
    """
    Returns:
        amap_lb  : anomaly map in letterbox space [resolution x resolution]
        gt_mask  : binary mask in letterbox space [resolution x resolution]
        gt_label : 0 or 1
        image_score : float (p95 of amap_lb)
        amap_orig   : anomaly map at original resolution (for heatmap)
        orig_size   : (w, h)
    """
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    img_lb, pad_info = letterbox_resize(img_pil, args.resolution)
    img_input = multi_scale_retinex(img_lb) if args.retinex else img_lb

    # gt_label solely from folder name
    gt_label = 0 if "good" in image_path.parts else 1

    # gt_mask
    if mask_path and mask_path.exists():
        mask_pil = Image.open(mask_path).convert("L")
        gt_mask  = letterbox_mask(mask_pil, args.resolution, pad_info)
    else:
        gt_mask = np.zeros((args.resolution, args.resolution), dtype=np.uint8)

    # SVD score
    layer_indices = args.layer_indices if args.layer_indices else None
    score_map = multi_scale_svd_score(
        model, img_input, device,
        scales=args.scales, k_ratio=args.k_ratio,
        layer_indices=layer_indices,
    )

    amap = cv2.resize(score_map, (args.resolution, args.resolution),
                      interpolation=cv2.INTER_LINEAR)
    amap = gaussian_filter(amap, sigma=args.sigma)

    # Normalize per-image (needed for heatmap only; global metrics use raw)
    lo, hi = amap.min(), amap.max()
    if hi > lo:
        amap = (amap - lo) / (hi - lo)

    amap_orig   = unpad_map(amap, pad_info, orig_w, orig_h)
    image_score = float(np.percentile(amap, 95))

    return amap, gt_mask, gt_label, image_score, amap_orig, (orig_w, orig_h)


# ─────────────────────────────────────────────────────────────────────────────
# Global metric computation  (MVTec-correct: accumulate then compute once)
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_metrics(all_amaps, all_masks, all_image_scores, all_gt_labels):
    """
    all_amaps        : list of np.ndarray [H, W]  — per-image anomaly maps (letterbox)
    all_masks        : list of np.ndarray [H, W]  — per-image binary masks (letterbox)
    all_image_scores : list of float
    all_gt_labels    : list of int (0/1)
    """
    from torchmetrics.classification import BinaryAUROC
    from anomalib.metrics.aupro import _AUPRO

    results = {}

    # ── Pixel AUROC — flatten all pixels from all images ──────────────────
    all_pred_flat = np.concatenate([a.flatten() for a in all_amaps])
    all_mask_flat = np.concatenate([m.flatten() for m in all_masks])

    pred_t = torch.from_numpy(all_pred_flat).float()
    mask_t = torch.from_numpy(all_mask_flat).long()

    if len(np.unique(all_mask_flat)) >= 2:
        pauroc = BinaryAUROC()
        results["pixel_auroc"] = pauroc(pred_t, mask_t).item()
    else:
        results["pixel_auroc"] = float("nan")
        print("  [warn] no positive pixels across all masks → pixel_auroc skipped")

    # ── AUPRO — accumulate per image then compute once ────────────────────
    for fpr_limit, key in [(0.05, "aupro_05"), (0.30, "aupro_30")]:
        try:
            aupro = _AUPRO(fpr_limit=fpr_limit)
            for amap, mask in zip(all_amaps, all_masks):
                aupro.update(
                    torch.from_numpy(amap).float().unsqueeze(0),
                    torch.from_numpy(mask).long().unsqueeze(0),
                )
            results[key] = aupro.compute().item()
        except Exception as e:
            results[key] = float("nan")
            print(f"  [warn] {key}: {e}")

    # ── Image AUROC — one score per image ────────────────────────────────
    if len(set(all_gt_labels)) >= 2:
        img_pred = torch.tensor(all_image_scores).float()
        img_lbl  = torch.tensor(all_gt_labels).long()
        img_auroc = BinaryAUROC()
        results["image_auroc"] = img_auroc(img_pred, img_lbl).item()
    else:
        results["image_auroc"] = float("nan")
        print("  [warn] only one class in gt_labels → image_auroc skipped")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir  = Path(args.image_dir)
    mask_dir   = Path(args.mask_dir) if args.mask_dir else None

    print(f"Device      : {device}")
    print(f"Image dir   : {image_dir}")
    print(f"Mask dir    : {mask_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Resolution  : {args.resolution}")

    image_paths = sorted([p for p in image_dir.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        print(f"[ERROR] No images found in {image_dir}")
        return
    print(f"\nFound {len(image_paths)} images.\n")

    print(f"Loading DINOv2 ({args.backbone}) ...")
    model, device = load_dinov2(args.backbone, device)
    print()

    # Accumulators for global metrics
    all_amaps        = []
    all_masks        = []
    all_image_scores = []
    all_gt_labels    = []

    # Per-image rows for CSV
    per_image_rows = []

    for idx, image_path in enumerate(image_paths, 1):
        # Match mask
        mask_path = None
        if mask_dir:
            for candidate in [
                mask_dir / f"{image_path.stem}_mask{image_path.suffix}",
                mask_dir / f"{image_path.stem}_mask.png",
            ]:
                if candidate.exists():
                    mask_path = candidate
                    break

        mask_status = mask_path.name if mask_path else "no mask"
        print(f"[{idx:3d}/{len(image_paths)}] {image_path.name}  |  mask: {mask_status}")

        try:
            amap, gt_mask, gt_label, image_score, amap_orig, (orig_w, orig_h) = \
                infer_one(image_path, mask_path, model, device, args)

            # Save heatmap
            heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
            save_heatmap(amap_orig, heatmap_path)

            # Accumulate
            all_amaps.append(amap)
            all_masks.append(gt_mask)
            all_image_scores.append(image_score)
            all_gt_labels.append(gt_label)

            per_image_rows.append({
                "image":       str(image_path),
                "orig_size":   f"{orig_w}x{orig_h}",
                "gt_label":    gt_label,
                "image_score": image_score,
                "heatmap":     str(heatmap_path),
            })
            print(f"           image_score={image_score:.4f}  gt={gt_label}  "
                  f"heatmap saved → {heatmap_path.name}")

        except Exception as e:
            print(f"    [ERROR] {e}")

    if not all_amaps:
        print("No results to save.")
        return

    # ── Global metrics (computed once over all images) ────────────────────
    print("\nComputing global metrics ...")
    try:
        global_metrics = compute_global_metrics(
            all_amaps, all_masks, all_image_scores, all_gt_labels)
    except Exception as e:
        print(f"  [ERROR] global metrics failed: {e}")
        global_metrics = {}

    print("\n" + "=" * 60)
    print("GLOBAL RESULTS  (MVTec-correct accumulation)")
    print("=" * 60)
    print(f"  Images processed   : {len(all_amaps)}")
    print(f"  Image AUROC        : {global_metrics.get('image_auroc', float('nan'))*100:.2f}%")
    print(f"  Pixel AUROC        : {global_metrics.get('pixel_auroc', float('nan'))*100:.2f}%")
    print(f"  AUPRO @ FPR 0.05   : {global_metrics.get('aupro_05',   float('nan'))*100:.2f}%")
    print(f"  AUPRO @ FPR 0.30   : {global_metrics.get('aupro_30',   float('nan'))*100:.2f}%")
    print("=" * 60)

    # ── Per-image CSV ─────────────────────────────────────────────────────
    per_image_csv = output_dir / "results_per_image.csv"
    fieldnames = ["image", "orig_size", "gt_label", "image_score", "heatmap"]
    with open(per_image_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in per_image_rows:
            w.writerow({k: (f"{row[k]:.6f}" if isinstance(row[k], float) else row[k])
                        for k in fieldnames if k in row})
    print(f"\nPer-image CSV  : {per_image_csv}")

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_csv = output_dir / "results_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_images",       len(all_amaps)])
        w.writerow(["image_auroc",    f"{global_metrics.get('image_auroc', float('nan')):.6f}"])
        w.writerow(["pixel_auroc",    f"{global_metrics.get('pixel_auroc', float('nan')):.6f}"])
        w.writerow(["aupro_05",       f"{global_metrics.get('aupro_05',   float('nan')):.6f}"])
        w.writerow(["aupro_30",       f"{global_metrics.get('aupro_30',   float('nan')):.6f}"])
        w.writerow(["backbone",       args.backbone])
        w.writerow(["resolution",     args.resolution])
        w.writerow(["scales",         str(args.scales)])
        w.writerow(["k_ratio",        args.k_ratio])
        w.writerow(["sigma",          args.sigma])
        w.writerow(["retinex",        args.retinex])
    print(f"Summary CSV    : {summary_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVD Zero-Shot Anomaly Detection — Batch Folder Evaluation")

    parser.add_argument("--image_dir",     required=True)
    parser.add_argument("--mask_dir",      default=None)
    parser.add_argument("--output_dir",    default="./svd_results")
    parser.add_argument("--resolution",    type=int,   default=518)
    parser.add_argument("--backbone",      default="dinov2_vitl14",
                        choices=["dinov2_vits14","dinov2_vitb14",
                                 "dinov2_vitl14","dinov2_vitg14"])
    parser.add_argument("--scales",        type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--k_ratio",       type=float, default=0.1)
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[8, 16, 23])
    parser.add_argument("--sigma",         type=float, default=4.0)
    parser.add_argument("--retinex",       action="store_true")

    main(parser.parse_args())