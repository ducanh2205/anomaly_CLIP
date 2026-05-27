"""
SVD-based Zero-Shot Anomaly Detection
======================================
Core idea:
  Normal industrial surface patches lie on a low-dimensional manifold.
  We estimate this manifold via SVD of DINOv2 patch features extracted
  from the test image itself — no reference images, no text prompts,
  truly zero-shot.

  Anomaly score per patch = reconstruction error after projecting onto
  top-k singular vectors of the patch feature matrix.

Additional components:
  - Retinex normalization  → lighting robustness
  - Letterbox resize       → aspect ratio preservation
  - Multi-scale SVD        → tiny defect sensitivity

Install:
  pip install torch torchvision timm opencv-python-headless Pillow numpy scipy

Usage:
  python svd_zsad.py --image path/to/test.png --mask  path/to/mask.png --resolution 518 --output_dir ./results
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# Retinex illumination normalization
# ─────────────────────────────────────────────────────────────────────────────

def multi_scale_retinex(img_np: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    """
    Multi-Scale Retinex on a float32 RGB image in [0,1].
    Returns reflectance component (lighting removed) in [0,1].
    """
    img_log = np.log1p(img_np * 255.0)
    msr = np.zeros_like(img_log)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img_log, (0, 0), sigma)
        msr += img_log - blur
    msr /= len(sigmas)
    # Normalize per-channel to [0,1]
    for c in range(3):
        ch = msr[..., c]
        msr[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return msr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Letterbox resize (preserve aspect ratio, pad with mean)
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_resize(img: Image.Image, target_size: int) -> tuple[np.ndarray, tuple]:
    """
    Resize image to target_size x target_size preserving aspect ratio.
    Padding is filled with the mean pixel value.
    Returns (padded_np_float32 [H,W,3] in [0,1], (pad_top, pad_left, new_h, new_w))
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32) / 255.0

    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2

    mean_val = arr.mean()
    canvas = np.full((target_size, target_size, 3), mean_val, dtype=np.float32)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = arr

    return canvas, (pad_top, pad_left, new_h, new_w)


def letterbox_mask(mask: Image.Image, target_size: int,
                   pad_info: tuple) -> np.ndarray:
    """Resize and pad binary mask to match letterbox image."""
    pad_top, pad_left, new_h, new_w = pad_info
    mask_resized = mask.resize((new_w, new_h), Image.NEAREST)
    arr = (np.array(mask_resized) > 127).astype(np.uint8)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = arr
    return canvas


def unpad_map(amap: np.ndarray, pad_info: tuple,
              orig_w: int, orig_h: int) -> np.ndarray:
    """Crop padding from anomaly map and resize back to original resolution."""
    pad_top, pad_left, new_h, new_w = pad_info
    cropped = amap[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def load_dinov2(model_name: str = "dinov2_vitl14", device: torch.device = None):
    """Load DINOv2 from torch hub."""
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
                            layer_indices: list[int] = None
                            ) -> torch.Tensor:
    """
    Extract patch-level features from DINOv2.
    img_np: float32 [H, W, 3] in [0,1]
    Returns: [N_patches, C] float32 on CPU
    """
    x = torch.from_numpy(img_np).permute(2, 0, 1).float()  # [3,H,W]
    mean = DINO_MEAN.view(3, 1, 1)
    std  = DINO_STD.view(3, 1, 1)
    x = (x - mean) / std
    x = x.unsqueeze(0).to(device)  # [1,3,H,W]

    # Extract intermediate features if layer_indices specified
    if layer_indices is not None:
        features_list = []
        def hook_fn(module, input, output):
            # output: [1, N+1, C] — remove CLS token
            features_list.append(output[:, 1:, :].squeeze(0).cpu())

        handles = []
        for idx in layer_indices:
            h = model.blocks[idx].register_forward_hook(hook_fn)
            handles.append(h)

        _ = model(x)
        for h in handles:
            h.remove()

        # Concatenate features from all layers: [N_patches, C*len(layers)]
        features = torch.cat(features_list, dim=-1)
    else:
        # Use final layer features
        out = model.forward_features(x)
        features = out["x_norm_patchtokens"].squeeze(0).cpu()  # [N_patches, C]

    return features.float()


# ─────────────────────────────────────────────────────────────────────────────
# SVD-based anomaly scoring — core contribution
# ─────────────────────────────────────────────────────────────────────────────

def svd_anomaly_score(features: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute per-patch anomaly score via SVD reconstruction error.

    Normal patches lie on a low-dimensional manifold.
    We estimate this manifold with the top-k singular vectors.
    Anomaly score = ||f - f_reconstructed||^2

    features: [N, C]
    k: number of singular vectors to keep (manifold rank)
    Returns: [N] float32 anomaly scores
    """
    # Center features
    mean = features.mean(dim=0, keepdim=True)      # [1, C]
    F_centered = features - mean                    # [N, C]

    # SVD: F = U S Vt
    # U: [N, N], S: [min(N,C)], Vt: [min(N,C), C]
    try:
        U, S, Vt = torch.linalg.svd(F_centered, full_matrices=False)
    except Exception:
        # Fallback for older torch versions
        U, S, Vt = torch.svd(F_centered)
        Vt = Vt.T

    # Keep top-k components
    k = min(k, S.shape[0])
    Vt_k = Vt[:k, :]                               # [k, C]

    # Reconstruct: project onto k-dim subspace then back
    F_proj = F_centered @ Vt_k.T @ Vt_k           # [N, C]

    # Reconstruction error per patch
    error = ((F_centered - F_proj) ** 2).sum(dim=-1)  # [N]

    return error


def multi_scale_svd_score(model, img_np: np.ndarray,
                          device: torch.device,
                          scales: list[int],
                          k_ratio: float = 0.1,
                          layer_indices: list[int] = None) -> np.ndarray:
    """
    Multi-scale SVD anomaly scoring.
    For each scale, crop tiles and compute SVD score, then merge.

    scales: list of tile sizes (in patches). E.g. [1, 2, 4] means
            compute SVD over 1x1, 2x2, 4x4 local windows.
    k_ratio: fraction of patches to keep as manifold rank.
    Returns: anomaly map [H_patches, W_patches] float32
    """
    H, W, _ = img_np.shape
    patch_size = 14  # DINOv2 patch size
    H_p = H // patch_size
    W_p = W // patch_size

    # Extract features at multiple DINOv2 layers for richer representation
    features = extract_patch_features(model, img_np, device, layer_indices)
    # features: [H_p * W_p, C]
    features_2d = features.view(H_p, W_p, -1)  # [H_p, W_p, C]

    score_map = torch.zeros(H_p, W_p)
    weight_map = torch.zeros(H_p, W_p)

    for scale in scales:
        # Slide window of size scale x scale over patch grid
        stride = max(1, scale // 2)  # 50% overlap
        for i in range(0, H_p - scale + 1, stride):
            for j in range(0, W_p - scale + 1, stride):
                window = features_2d[i:i+scale, j:j+scale, :]  # [s,s,C]
                N = scale * scale
                f = window.reshape(N, -1)                       # [N, C]

                k = max(1, int(N * k_ratio))
                scores = svd_anomaly_score(f, k)               # [N]
                scores_2d = scores.view(scale, scale)

                # Gaussian weight — center patches more reliable
                gauss = _gaussian_weight(scale)
                score_map[i:i+scale, j:j+scale]  += scores_2d * gauss
                weight_map[i:i+scale, j:j+scale] += gauss

    # Normalize by weight
    score_map = score_map / (weight_map + 1e-8)
    return score_map.numpy()


def _gaussian_weight(size: int) -> torch.Tensor:
    """Gaussian weight map for a size x size window (center = 1, edge → 0)."""
    if size == 1:
        return torch.ones(1, 1)
    sigma = size / 4.0
    coords = torch.arange(size).float() - (size - 1) / 2.0
    g1d = torch.exp(-coords**2 / (2 * sigma**2))
    g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
    return g2d / g2d.max()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred_map: np.ndarray, gt_mask: np.ndarray,
                    image_score: float, gt_label: int) -> dict:
    from torchmetrics.classification import BinaryAUROC, BinaryF1Score
    from anomalib.metrics.aupro import _AUPRO

    pred_t = torch.from_numpy(pred_map).float()
    mask_t = torch.from_numpy(gt_mask).long()

    results = {}

    # Pixel AUROC
    if len(np.unique(gt_mask)) >= 2:
        pauroc = BinaryAUROC()
        results["pixel_auroc"] = pauroc(pred_t.flatten(),
                                        mask_t.flatten()).item()
    else:
        results["pixel_auroc"] = float("nan")
        print("  [warn] only one class in mask, pixel_auroc skipped")

    # AUPRO @ 0.05
    aupro05 = _AUPRO(fpr_limit=0.05)
    aupro05.update(pred_t.unsqueeze(0), mask_t.unsqueeze(0))
    try:
        results["aupro_05"] = aupro05.compute().item()
    except Exception as e:
        results["aupro_05"] = float("nan")
        print(f"  [warn] AUPRO@0.05: {e}")

    # AUPRO @ 0.30
    aupro03 = _AUPRO(fpr_limit=0.30)
    aupro03.update(pred_t.unsqueeze(0), mask_t.unsqueeze(0))
    try:
        results["aupro_30"] = aupro03.compute().item()
    except Exception as e:
        results["aupro_30"] = float("nan")
        print(f"  [warn] AUPRO@0.30: {e}")

    # Image score
    results["image_score"] = image_score
    results["gt_label"]    = gt_label

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Save heatmap
# ─────────────────────────────────────────────────────────────────────────────

def save_heatmap(amap: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = amap.min(), amap.max()
    if hi > lo:
        amap = (amap - lo) / (hi - lo)
    uint8 = (amap * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(uint8, cv2.COLORMAP_JET))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Image       : {args.image}")
    print(f"Resolution  : {args.resolution}")

    image_path = Path(args.image)
    mask_path  = Path(args.mask) if args.mask else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GT label
    if args.label is not None:
        gt_label = int(args.label)
    else:
        gt_label = 0 if "good" in image_path.parts else 1
    print(f"GT label    : {gt_label}")

    # ── Load image ────────────────────────────────────────────────────────
    img_pil  = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    print(f"Original    : {orig_w}x{orig_h}")

    # Letterbox resize
    img_lb, pad_info = letterbox_resize(img_pil, args.resolution)
    print(f"Letterbox   : {args.resolution}x{args.resolution}  "
          f"pad=({pad_info[0]},{pad_info[1]}) content=({pad_info[3]}x{pad_info[2]})")

    # Retinex normalization
    if args.retinex:
        img_input = multi_scale_retinex(img_lb)
        print("Retinex     : enabled")
    else:
        img_input = img_lb

    # ── Load mask ─────────────────────────────────────────────────────────
    if mask_path and mask_path.exists():
        mask_pil = Image.open(mask_path).convert("L")
        gt_mask  = letterbox_mask(mask_pil, args.resolution, pad_info)
    else:
        gt_mask = np.zeros((args.resolution, args.resolution), dtype=np.uint8)

    # ── Load DINOv2 ───────────────────────────────────────────────────────
    print(f"Loading DINOv2 ({args.backbone}) ...")
    model, device = load_dinov2(args.backbone, device)

    # ── Extract features & SVD scoring ───────────────────────────────────
    print("Computing SVD anomaly scores ...")
    layer_indices = args.layer_indices if args.layer_indices else None
    scales = args.scales

    score_map = multi_scale_svd_score(
        model, img_input, device,
        scales=scales,
        k_ratio=args.k_ratio,
        layer_indices=layer_indices,
    )
    # score_map: [H_patches, W_patches]

    # Upsample to letterbox resolution
    H_p, W_p = score_map.shape
    amap = cv2.resize(score_map, (args.resolution, args.resolution),
                      interpolation=cv2.INTER_LINEAR)

    # Gaussian smoothing
    amap = gaussian_filter(amap, sigma=args.sigma)

    # Normalize
    lo, hi = amap.min(), amap.max()
    if hi > lo:
        amap = (amap - lo) / (hi - lo)

    # Remove padding — get anomaly map at original resolution
    amap_orig = unpad_map(amap, pad_info, orig_w, orig_h)

    # Resize gt_mask back to original for metric computation on padded version
    # (keep metrics in letterbox space for fair comparison)
    image_score = float(amap.max())

    # ── Save heatmap ──────────────────────────────────────────────────────
    heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
    save_heatmap(amap_orig, heatmap_path)
    print(f"Heatmap saved: {heatmap_path}  ({orig_w}x{orig_h})")

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = {}
    print("Computing metrics ...")
    try:
        metrics = compute_metrics(amap, gt_mask.astype(np.int64),
                                  image_score, gt_label)
        print("\n" + "=" * 50)
        print(f"Results — {image_path.stem}")
        print("=" * 50)
        print(f"  Image anomaly score : {image_score:.4f}  (gt={gt_label})")
        print(f"  Pixel AUROC         : {metrics.get('pixel_auroc', float('nan'))*100:.2f}%")
        print(f"  AUPRO @ FPR 0.05    : {metrics.get('aupro_05',   float('nan'))*100:.2f}%")
        print(f"  AUPRO @ FPR 0.30    : {metrics.get('aupro_30',   float('nan'))*100:.2f}%")
        print("=" * 50)
        print("NOTE: metrics meaningful only when aggregated over full test set.")
    except Exception as e:
        print(f"  [warn] metrics failed: {e}")
        print(f"  Image anomaly score: {image_score:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    import csv
    csv_path = output_dir / f"{image_path.stem}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["image", str(image_path)])
        w.writerow(["resolution", args.resolution])
        w.writerow(["gt_label", gt_label])
        w.writerow(["image_score", f"{image_score:.6f}"])
        for k, v in metrics.items():
            if k not in ("image_score", "gt_label"):
                w.writerow([k, f"{v:.6f}"])
    print(f"CSV saved   : {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SVD-based Zero-Shot Anomaly Detection with DINOv2")

    parser.add_argument("--image",      required=True,
                        help="Path to test image")
    parser.add_argument("--mask",       default=None,
                        help="Path to ground-truth mask (optional)")
    parser.add_argument("--label",      default=None,
                        help="GT label: 0=good, 1=anomaly (auto-inferred if omitted)")
    parser.add_argument("--resolution", type=int, default=518,
                        help="Letterbox target resolution (default: 518)")
    parser.add_argument("--backbone",   default="dinov2_vitl14",
                        choices=["dinov2_vits14", "dinov2_vitb14",
                                 "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 backbone (default: vitl14)")
    parser.add_argument("--scales",     type=int, nargs="+", default=[4, 8, 16],
                        help="Multi-scale window sizes in patches (default: 4 8 16)")
    parser.add_argument("--k_ratio",    type=float, default=0.1,
                        help="Fraction of singular vectors to keep (default: 0.1)")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[8, 16, 23],
                        help="DINOv2 layer indices to extract features from")
    parser.add_argument("--sigma",      type=float, default=4.0,
                        help="Gaussian smoothing sigma (default: 4.0)")
    parser.add_argument("--retinex",    action="store_true",
                        help="Enable Retinex illumination normalization")
    parser.add_argument("--output_dir", default="./svd_results",
                        help="Output directory")

    main(parser.parse_args())