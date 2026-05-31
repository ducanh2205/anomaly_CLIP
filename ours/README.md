# MVTec AD 2 — Submission Pipeline

Pipeline chạy inference trên `test_private` / `test_private_mixed` của MVTec AD 2
rồi đóng gói submission đúng format chấm điểm tại
https://benchmark.mvtec.com.

## Yêu cầu

```bash
pip install torch torchvision opencv-python-headless Pillow numpy scipy tqdm tifffile
```

**MVTec utils** đã được kèm sẵn tại `<repo_root>/MVTecAD2_public_code_utils/` (CC-BY-NC-SA 4.0, xem `license.txt` trong thư mục đó). Chỉ cần:

- Sửa `PATH_TO_MVTEC_AD_2_FOLDER` trong [MVTecAD2_public_code_utils/mvtec_ad_2_public_offline.py](../MVTecAD2_public_code_utils/mvtec_ad_2_public_offline.py) trỏ về thư mục dataset thật.

> Nếu muốn lấy bản gốc / mới hơn: download tại https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2 (nút *Download code utils*). Zip bung ra cấu trúc lồng — phải flatten lên 1 cấp vì `batch_inference_private.py` import từ đường dẫn 1 cấp.

## 3 bước

### 1. Inference → dump heatmap + raw score

```bash
python batch_inference_private.py \
  --data_root C:\path\to\mvtec_ad_2 \
  --output_dir ./results_private \
  --resolution 518
```

Mỗi ảnh tạo 2 file trong `results_private/{obj}/{split}_heatmaps/`:

| File | Nội dung | Vai trò |
|---|---|---|
| `{stem}_heatmap.png` | JET colormap, 8-bit BGR | Preview, fallback |
| `{stem}_score.npy` | float32, shape = ảnh gốc | Submission (lossless) |

`.npy` là **bắt buộc** cho điểm tối đa — converter ưu tiên đọc file này, chỉ dùng PNG khi `.npy` không có (mất precision).

> Disk usage: `.npy` float32 ~9 MB/ảnh × 4090 ảnh ≈ 37 GB. Nếu thiếu chỗ, đổi
> `.astype(np.float32)` → `.astype(np.float16)` trong `batch_inference_private.py`
> (giảm xuống ~18 GB, đủ độ chính xác cho benchmark).

### 2. Convert → MVTec submission layout

```bash
python convert_to_submission_format.py \
  --results_dir ./results_private \
  --output_dir ./submission_private \
  --threshold 0.5 \
  --workers 8
```

Output:
```
submission_private/
├── anomaly_images/{obj}/{split}/{idx:03d}_{regular|mixed}.tiff   # float16, 2D
└── anomaly_images_thresholded/{obj}/{split}/{idx:03d}_{regular|mixed}.png  # uint8, {0,255}
```

Console sẽ in `npy=X, png=Y`:
- `npy=4090, png=0` → lossless ✓
- `npy=0,    png=4090` → lossy fallback (cảnh báo)

### 3. Validate + nén bằng checker chính thức

```bash
python ../MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py \
  ./submission_private
```

Nếu pass, checker tự tạo `./submission_private.tar.gz` (gzip mất ~25 phút cho
4090 file, ~3 GB output).

→ Upload file `.tar.gz` này tại https://benchmark.mvtec.com.

## Format yêu cầu (tham khảo)

Từ [utils.py](../MVTecAD2_public_code_utils/utils.py) trong code utils:

| Item | Spec |
|---|---|
| Cấu trúc | `anomaly_images/{obj}/{split}/{idx:03d}_{regular\|mixed}.{tiff\|png}` |
| Số file/split/object | `OBJECT_FILE_COUNTER` cố định (vd `can` = 321) |
| Suffix theo split | `test_private` → `_regular`, `test_private_mixed` → `_mixed` |
| TIFF | single-channel, `dtype=float16` |
| PNG threshold | single-channel, giá trị ∈ {0, 255} |
| Optional | `anomaly_images_thresholded/` không bắt buộc; bỏ qua sẽ không có điểm SegF1/ClassF1 |

## Threshold

Mặc định `--threshold 0.5` (phù hợp khi score đã normalize về [0,1]).

Threshold tối ưu (gợi ý của MVTec):
```
threshold = mean(val_anomaly_scores) + 3 * std(val_anomaly_scores)
```
Tính trên validation split với chính score float32 đã dump. Yêu cầu inference
cũng chạy trên `validation` rồi gom .npy thành 1 mảng.

## Giới hạn upload

**2 submissions / 7 ngày**. Luôn chạy bước 3 (checker chính thức) trước khi upload —
nếu validate fail, checker sẽ raise `SubmissionException` mô tả cụ thể chỗ sai và
**không** tạo `.tar.gz`. Lúc đó bạn vẫn chưa tốn slot upload.

## Troubleshooting

| Triệu chứng | Khả năng cao |
|---|---|
| `SubmissionException: Expected N files, found M` | Inference thiếu/dư ảnh — kiểm tra log inference, có thể có ảnh bị skip do exception |
| `Anomaly image ... is not of type float16` | Converter chưa chạy/chạy phiên bản cũ |
| `Values of thresholded image ... not in {0, 255}` | PNG bị save dạng colormap; convert lại với `Image.fromarray(..., mode='L')` |
| `npy=0, png=4090` trong log convert | Inference chưa dump `.npy` — patch lại `batch_inference_private.py` theo block "Save heatmap" |
| `Compressed file ended before end-of-stream marker` | Process checker chưa chạy xong, file `.tar.gz` đang ghi dở — đợi |
