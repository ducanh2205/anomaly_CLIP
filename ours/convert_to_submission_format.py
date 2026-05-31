"""Convert per-image anomaly scores to MVTec AD 2 submission format.

Input layout (per object x split):
    {results_dir}/{obj}/{split}_heatmaps/{idx:03d}_{regular|mixed}_score.npy   (preferred)
    {results_dir}/{obj}/{split}_heatmaps/{idx:03d}_{regular|mixed}_heatmap.png (fallback)

PREFERRED FLOW (lossless): patch the inference script to dump raw float scores
alongside the heatmap PNG:

    # inside infer_one / batch loop, right after `amap_orig` is computed:
    np.save(heatmap_path.with_name(f"{stem}_score.npy"),
            amap_orig.astype(np.float32))

The .npy is expected to be a 2D float array already normalized to [0, 1]
at the original image resolution (same as `unpad_map` produces). The
converter casts directly to float16 with NO further normalization, so
cross-image score scale is preserved for AUROC/AUPRO.

FALLBACK (lossy): if .npy is absent, the converter inverts the JET PNG
via nearest-neighbour lookup against the 256-entry LUT. This only
recovers 256 score levels and ruins cross-image calibration -- use only
when the float data is unrecoverable.

Output:
    {output_dir}/anomaly_images/{obj}/{split}/{idx:03d}_{regular|mixed}.tiff
        (float16, single-channel)
    {output_dir}/anomaly_images_thresholded/{obj}/{split}/{idx:03d}_{regular|mixed}.png
        (uint8 in {0, 255})

Usage:
    python convert_to_submission_format.py \\
        --results_dir ./results_private \\
        --output_dir ./submission_private \\
        --threshold 0.5
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy.spatial import cKDTree
from tqdm import tqdm


MVTEC_AD2_OBJECTS: tuple[str, ...] = (
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
)

OBJECT_FILE_COUNTER: dict[str, int] = {
    "can": 321, "fabric": 314, "fruit_jelly": 255, "rice": 277,
    "sheet_metal": 142, "vial": 276, "wallplugs": 232, "walnuts": 228,
}

SPLIT_SUFFIX: dict[str, str] = {
    "test_private": "regular",
    "test_private_mixed": "mixed",
}


def build_jet_lut_tree() -> cKDTree:
    lut_bgr = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET,
    ).reshape(-1, 3).astype(np.int32)
    return cKDTree(lut_bgr)


def invert_jet(img_bgr: np.ndarray, tree: cKDTree) -> np.ndarray:
    h, w, _ = img_bgr.shape
    pixels = img_bgr.reshape(-1, 3).astype(np.int32)
    # Heatmap has <=256 unique colours; dedupe to minimise tree queries.
    unique_colors, inverse = np.unique(pixels, axis=0, return_inverse=True)
    _, lut_idx = tree.query(unique_colors, k=1)
    scores_flat = lut_idx[inverse].astype(np.float32) / 255.0
    return scores_flat.reshape(h, w)


def load_score(
    src_dir: Path,
    stem: str,
    tree: cKDTree,
) -> tuple[np.ndarray, str]:
    """Return (score_map_float32, source_kind) where source_kind in {npy, png}."""
    npy_path = src_dir / f"{stem}_score.npy"
    if npy_path.exists():
        arr = np.load(npy_path)
        if arr.ndim != 2:
            raise RuntimeError(f"{npy_path} expected 2-D, got shape {arr.shape}")
        return arr.astype(np.float32), "npy"

    png_path = src_dir / f"{stem}_heatmap.png"
    if not png_path.exists():
        raise RuntimeError(
            f"No score file for {stem} in {src_dir} "
            f"(looked for {npy_path.name} and {png_path.name})"
        )
    img_bgr = cv2.imread(str(png_path))
    if img_bgr is None:
        raise RuntimeError(f"Cannot read {png_path}")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise RuntimeError(f"Unexpected shape {img_bgr.shape} for {png_path}")
    return invert_jet(img_bgr, tree), "png"


def write_outputs(
    score: np.ndarray,
    tiff_path: Path,
    png_path: Path,
    threshold: float,
) -> None:
    tiff_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(tiff_path), score.astype(np.float16))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    binary = ((score > threshold).astype(np.uint8) * 255)
    Image.fromarray(binary, mode="L").save(str(png_path))


def convert_one(
    src_dir: Path,
    stem: str,
    tiff_path: Path,
    png_path: Path,
    tree: cKDTree,
    threshold: float,
) -> str:
    score, kind = load_score(src_dir, stem, tree)
    write_outputs(score, tiff_path, png_path, threshold)
    return kind


def collect_stems(src_dir: Path, suffix: str) -> list[str]:
    """Stems in src_dir matching '*_{suffix}', dedup across .npy/.png."""
    stems: set[str] = set()
    for p in src_dir.glob(f"*_{suffix}_score.npy"):
        stems.add(p.stem.removesuffix("_score"))
    for p in src_dir.glob(f"*_{suffix}_heatmap.png"):
        stems.add(p.stem.removesuffix("_heatmap"))
    return sorted(stems)


def convert(
    results_dir: Path,
    output_dir: Path,
    threshold: float,
    workers: int,
) -> tuple[int, dict[str, int], list[str]]:
    tree = build_jet_lut_tree()
    total = 0
    kinds: dict[str, int] = {"npy": 0, "png": 0}
    issues: list[str] = []

    for obj in MVTEC_AD2_OBJECTS:
        expected = OBJECT_FILE_COUNTER[obj]
        for split, suffix in SPLIT_SUFFIX.items():
            src_dir = results_dir / obj / f"{split}_heatmaps"
            if not src_dir.is_dir():
                issues.append(f"[MISSING DIR] {src_dir}")
                continue

            stems = collect_stems(src_dir, suffix)
            if len(stems) != expected:
                issues.append(
                    f"[COUNT] {obj}/{split}: got {len(stems)}, "
                    f"expected {expected}"
                )

            tiff_dir = output_dir / "anomaly_images" / obj / split
            png_dir = output_dir / "anomaly_images_thresholded" / obj / split

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {}
                for stem in stems:
                    expected_stem = f"{stem.split('_', 1)[0]}_{suffix}"
                    if stem != expected_stem:
                        issues.append(
                            f"[NAME] {src_dir}/{stem}: does not match "
                            f"{{idx:03d}}_{suffix}"
                        )
                        continue
                    fut = ex.submit(
                        convert_one,
                        src_dir, stem,
                        tiff_dir / f"{stem}.tiff",
                        png_dir / f"{stem}.png",
                        tree, threshold,
                    )
                    futures[fut] = stem

                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"{obj}/{split:<20}",
                    ncols=80,
                ):
                    try:
                        kind = fut.result()
                        kinds[kind] += 1
                        total += 1
                    except Exception as e:
                        issues.append(f"[ERROR] {futures[fut]}: {e}")

    return total, kinds, issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", default="./results_private",
                        help="Directory with {obj}/{split}_heatmaps/*.{npy,png}")
    parser.add_argument("--output_dir", default="./submission_private",
                        help="Submission directory to write")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold in score scale (default: 0.5)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel worker threads (default: 8)")
    args = parser.parse_args()

    total, kinds, issues = convert(
        Path(args.results_dir),
        Path(args.output_dir),
        args.threshold,
        args.workers,
    )

    print(f"\nConverted {total} files (npy={kinds['npy']}, png={kinds['png']})")
    if kinds["png"] > 0 and kinds["npy"] == 0:
        print("WARNING: all inputs came from JET PNGs (lossy fallback). "
              "Patch inference to dump *_score.npy for full precision.")
    if issues:
        print(f"\n{len(issues)} issue(s):")
        for msg in issues:
            print(f"  {msg}")
    else:
        print("No structural issues detected.")


if __name__ == "__main__":
    main()
