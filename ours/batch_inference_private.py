"""
Batch Inference: MVTec AD 2 - test_private & test_private_mixed Splits
========================================================================
Inference on splits WITHOUT public ground truth.

Usage:
  python batch_inference_private.py \
    --data_root C:\\anomaly_detection\\data\\mvtec_ad_2 \
    --output_dir ./results_private \
    --resolution 518
"""

import argparse
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
import csv

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent / "MVTecAD2_public_code_utils"))
from mvtec_ad_2_public_offline import MVTecAD2

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Retinex
# ─────────────────────────────────────────────────────────────────────────────

def multi_scale_retinex(img_np: np.ndarray, sigmas=(15, 80, 250)) -> np.ndarray:
    """Multi-Scale Retinex normalization."""
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
# Letterbox resize
# ─────────────────────────────────────────────────────────────────────────────

def letterbox_resize(img: Image.Image, target_size: int):
    """Resize preserving aspect ratio, pad with mean."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    arr = np.array(img.resize((new_w, new_h), Image.BILINEAR)).astype(np.float32) / 255.0
    pad_top  = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas = np.full((target_size, target_size, 3), arr.mean(), dtype=np.float32)
    canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = arr
    return canvas, (pad_top, pad_left, new_h, new_w)


def unpad_map(amap: np.ndarray, pad_info: tuple, orig_w: int, orig_h: int) -> np.ndarray:
    """Crop padding and resize back to original resolution."""
    pad_top, pad_left, new_h, new_w = pad_info
    return cv2.resize(amap[pad_top:pad_top+new_h, pad_left:pad_left+new_w],
                      (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2
# ─────────────────────────────────────────────────────────────────────────────

def load_dinov2(model_name: str = "dinov2_vitl14", device: torch.device = None):
    """Load DINOv2 model."""
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
    """Extract DINOv2 patch features."""
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
    """Compute SVD-based anomaly score."""
    F_c = features - features.mean(dim=0, keepdim=True)
    try:
        _, _, Vt = torch.linalg.svd(F_c, full_matrices=False)
    except Exception:
        _, _, Vt = torch.svd(F_c)
        Vt = Vt.T
    k = min(k, Vt.shape[0])
    Vt_k = Vt[:k]
    F_proj = F_c @ Vt_k.T @ Vt_k
    return ((F_c - F_proj) ** 2).sum(dim=-1)


def _gaussian_weight(size: int) -> torch.Tensor:
    """Generate Gaussian weight map."""
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
    """Multi-scale SVD anomaly score."""
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
    """Save anomaly map as jet colormap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = amap.min(), amap.max()
    if hi > lo:
        amap = (amap - lo) / (hi - lo)
    cv2.imwrite(str(path), cv2.applyColorMap(
        (amap * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET))


# ─────────────────────────────────────────────────────────────────────────────
# Per-image inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_one(img_path: Path, model, device, args):
    """
    Inference on single image.

    Returns:
        amap_orig : anomaly map at original resolution
        image_score : p95 percentile anomaly value
        img_size : (width, height)
    """
    img_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    img_lb, pad_info = letterbox_resize(img_pil, args.resolution)
    img_input = multi_scale_retinex(img_lb) if args.retinex else img_lb

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

    # Normalize for heatmap
    lo, hi = amap.min(), amap.max()
    if hi > lo:
        amap = (amap - lo) / (hi - lo)

    amap_orig   = unpad_map(amap, pad_info, orig_w, orig_h)
    image_score = float(np.percentile(amap, 95))

    return amap_orig, image_score, (orig_w, orig_h)


# ─────────────────────────────────────────────────────────────────────────────
# Main batch inference
# ─────────────────────────────────────────────────────────────────────────────

MVTEC_AD2_OBJECTS = [
    'can', 'fabric', 'fruit_jelly', 'rice',
    'sheet_metal', 'vial', 'wallplugs', 'walnuts',
]

SPLITS = ['test_private', 'test_private_mixed']


def main(args):
    # Load model
    print(f"Loading DINOv2 (device: {args.device})...")
    model, device = load_dinov2("dinov2_vitl14", torch.device(args.device))

    output_root = Path(args.output_dir)

    for obj_name in MVTEC_AD2_OBJECTS:
        print(f"\n{'='*70}")
        print(f"Object: {obj_name}")
        print(f"{'='*70}")

        for split in SPLITS:
            print(f"\n  Split: {split}")

            try:
                # Load dataset
                dataset = MVTecAD2(
                    mad2_object=obj_name,
                    split=split,
                    transform=None,
                )
            except Exception as e:
                print(f"    [ERROR] Failed to load {obj_name}/{split}: {e}")
                continue

            if len(dataset) == 0:
                print(f"    [SKIP] No images in {split}")
                continue

            # Output dirs
            obj_output_dir = output_root / obj_name
            heatmap_dir = obj_output_dir / f"{split}_heatmaps"
            csv_path = obj_output_dir / f"{split}_results.csv"

            heatmap_dir.mkdir(parents=True, exist_ok=True)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Inference
            results = []
            for idx in tqdm(range(len(dataset)), desc=f"  {obj_name}/{split}", ncols=80):
                item = dataset[idx]
                img_pil = Image.open(item['image_path']).convert("RGB")

                # Infer
                amap_orig, img_score, img_size = infer_one(
                    Path(item['image_path']), model, device, args
                )

                # Save heatmap
                stem = Path(item['image_path']).stem
                heatmap_path = heatmap_dir / f"{stem}_heatmap.png"
                save_heatmap(amap_orig, heatmap_path)
                # Dump raw float score for lossless submission conversion
                np.save(heatmap_dir / f"{stem}_score.npy",
                        amap_orig.astype(np.float32))

                # Record result
                results.append({
                    'image_path': item['image_path'],
                    'image_size': f"{img_size[0]}x{img_size[1]}",
                    'anomaly_score': f"{img_score:.6f}",
                    'heatmap': str(heatmap_path),
                })

            # Save CSV
            if results:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['image_path', 'image_size', 'anomaly_score', 'heatmap'])
                    writer.writeheader()
                    writer.writerows(results)
                print(f"    ✓ Saved {len(results)} results to {csv_path}")
            else:
                print(f"    [WARN] No results for {obj_name}/{split}")

    print(f"\n{'='*70}")
    print("Done! All results saved.")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch inference on MVTec AD 2 private splits"
    )
    parser.add_argument("--data_root", type=str,
                        default=r"C:\anomaly_detection\data\mvtec_ad_2",
                        help="Path to MVTec AD 2 dataset root")
    parser.add_argument("--output_dir", type=str, default="./results_private",
                        help="Output directory for results")
    parser.add_argument("--resolution", type=int, default=518,
                        help="Input resolution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    parser.add_argument("--retinex", action="store_true", default=True,
                        help="Apply Retinex normalization")
    parser.add_argument("--scales", type=list, default=[1, 2, 4, 8, 16],
                        help="Multi-scale sizes")
    parser.add_argument("--k_ratio", type=float, default=0.1,
                        help="SVD k ratio")
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Gaussian filter sigma")
    parser.add_argument("--layer_indices", type=list, default=None,
                        help="DINOv2 layer indices to extract")

    args = parser.parse_args()
    main(args)
