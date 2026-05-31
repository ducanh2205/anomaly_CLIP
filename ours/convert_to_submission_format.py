import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image

MVTEC_AD2_OBJECTS = ['can','fabric','fruit_jelly','rice',
                     'sheet_metal','vial','wallplugs','walnuts']

def convert_fast(results_dir, output_dir, threshold=0.5):
    """
    Convert heatmap PNG files (jet colormap) to TIFF (continuous) + PNG (binary)
    Actual structure: results/obj/XXX_YYY_heatmap.png (no split subdirs)
    """
    results_dir = Path(results_dir)
    output_dir  = Path(output_dir)

    total = 0
    for obj in MVTEC_AD2_OBJECTS:
        src_obj_dir = results_dir / obj
        if not src_obj_dir.exists():
            continue

        # tạo thư mục đích
        tiff_dir = output_dir / "anomaly_images" / obj
        png_dir  = output_dir / "anomaly_images_thresholded" / obj
        tiff_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        # lấy tất cả heatmap PNG từ folder obj
        heatmap_files = list(src_obj_dir.glob("*_heatmap.png"))
        
        for hm_path in tqdm(sorted(heatmap_files), desc=obj, ncols=80):
            stem = hm_path.stem.replace("_heatmap", "")  # vd '000_regular'
            
            # đọc heatmap PNG (BGR format từ OpenCV)
            img_bgr = cv2.imread(str(hm_path))
            if img_bgr is None:
                print(f"[ERROR] Cannot read {hm_path}")
                continue
            
            # chuyển BGR → grayscale để lấy độ sáng (anomaly score)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # TIFF continuous (16-bit uint: multiply by 65535)
            tiff_path = tiff_dir / f"{stem}.tiff"
            img_tiff = (img_gray * 65535).astype(np.uint16)
            Image.fromarray(img_tiff, mode='I;16').save(str(tiff_path))
            
            # PNG binary thresholded (8-bit uint: 0 or 255)
            png_path = png_dir / f"{stem}.png"
            img_binary = (img_gray > threshold).astype(np.uint8) * 255
            cv2.imwrite(str(png_path), img_binary)
            
            total += 1
            
            # cleanup
            del img_bgr, img_gray, img_tiff, img_binary
    
    print(f"[Done] {total} files converted")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--output_dir",  default="./submission_private")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    convert_fast(a.results_dir, a.output_dir, a.threshold)