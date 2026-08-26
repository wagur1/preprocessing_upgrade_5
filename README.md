# VCM Preprocessing Upgrade 5

Bo cong cu preprocessing va toi uu hoa cho **Video Coding for Machines (VCM)**. Muc tieu khong phai la toi da PSNR mot cach mu quang, ma la giu chat luong tac vu AI sau khi encode/decode trong khi giam bitrate.

## Metric nao can toi uu?

Chon metric theo tac vu downstream va bao cao cung voi bitrate (kbps), do tre va chi phi preprocessing:

| Nhom | Metric khuyen nghi | Y nghia |
|---|---|---|
| Detection | mAP@0.5, mAP@[.5:.95], Recall@N | Do dung vi tri va ty le phat hien vat the |
| Segmentation | mIoU, Dice/F1, boundary F-score | Do trung mask va bien doi tuong |
| Tracking | HOTA, IDF1, ID switches | Can bang localization va duy tri identity |
| Re-identification | mAP, Rank-1/5 | Kha nang truy hoi dung doi tuong |
| Pose/keypoint | OKS-mAP, PCK | Do chinh xac diem khop |
| Perception tong quat | CLIP/Image embedding cosine | Bao ton ngu nghia/feature |
| Tin hieu anh bo tro | PSNR-Y, SSIM, MS-SSIM, VMAF | Chi nen dung lam guardrail, khong phai muc tieu duy nhat |

Metric chinh nen la metric AI (vi du mAP hoac mIoU). PSNR/SSIM co the tang nhung mAP giam khi cac canh/texture quan trong bi lam mo.

## BD-rate va ham muc tieu

Voi moi cau hinh preprocessing, can tao RD points theo cac QP hoac bitrate giong nhau:

```text
(bitrate_kbps, task_quality)
```

`bd_rate(reference, test)` fit da thuc bac 2 tren `ln(rate)` theo quality va tich phan tren khoang giao nhau. Ket qua am nghia la test tiet kiem bitrate (vi du `-18.4%`). `bd_quality` do muc tang quality tai cung bitrate. Day la cach bao cao nen dung trong paper/benchmark, khong chi so sanh mot QP.

Optimizer dung objective:

```text
score = w_quality * task_quality - w_rate * log(1 + bitrate_kbps)
```

Trong benchmark day du, hay toi uu Pareto giua `BD-rate`, metric AI, latency (ms/frame), memory va energy. Khong gop cac metric co don vi khac nhau neu chua normalize theo baseline.

## Cai dat va chay

```bash
pip install -e ".[image]"
vcm-preprocess frames/processed frames/out --config config.json
```

Config mau:

```json
{"denoise": 0.15, "sharpen": 0.20, "contrast": 1.05,
 "saturation": 1.0, "luma_gain": 1.0, "chroma_subsample": false}
```

API Python:

```python
from vcm_preprocess import PreprocessConfig, preprocess_sequence, RandomSearch, BDRateSearch
preprocess_sequence("frames", "frames_pp", PreprocessConfig())

def evaluator(cfg):
    # 1) preprocess_sequence; 2) encode/decode bang VTM/x265/AV1;
    # 3) chay model AI; 4) tra task_quality va bitrate_kbps.
    return {"task_quality": 0.72, "bitrate_kbps": 500}

result = RandomSearch(evaluator, iterations=100, seed=7).run()
print(result.config, result.metrics)
```

Neu muon toi uu truc tiep BD-rate tren nhieu diem QP:

```python
reference = [(120, .55), (220, .63), (410, .70), (820, .75)]
def rd_evaluator(cfg):
    # Tra ve [(bitrate_kbps, mAP), ...] sau khi encode o cac QP co dinh.
    return [(110, .55), (205, .63), (390, .70), (790, .75)]
result = BDRateSearch(reference, rd_evaluator, iterations=100).run()
print(result.metrics["bd_rate_percent"])  # am = tiet kiem bitrate
```

`evaluator` la hop dong duy nhat voi codec/model; co the them `latency_ms`, `vmaf`, `psnr_y` de log va phan tich. Xem `examples/toy_optimize.py` cho evaluator gia lap va `examples/rd_points.json` cho dinh dang RD.

## Quy trinh benchmark de xuat

1. Chia train/validation/test theo video, khong tron frame giua cac tap.
2. Chay cung codec, preset, GOP, resolution va cac QP (thuong 22/27/32/37).
3. Do bitrate trung binh, task metric tren tat ca frame, latency preprocessing + decode + inference.
4. Tinh BD-rate tren quality AI; bao cao them BD-rate theo VMAF/PSNR de phat hien suy giam thi giac.
5. Chon diem Pareto va xac nhan tren test set mot lan duy nhat.

## Ghi chu ky thuat

- Pillow la dependency tuy chon; metric va optimizer chay duoc trong Python stdlib. Yeu cau Python 3.10+.
- `denoise`, `sharpen` nam trong [0,1]; contrast [0.8,1.3], saturation [0.7,1.3], luma_gain [0.85,1.15].
- Pipeline hien tai la baseline CPU, khong thay the learned preprocessor. De dat ket qua tot nhat, co the thay `preprocess_image` bang ONNX/TensorRT model va giu nguyen evaluator/optimizer.
- BD-rate khong co y nghia neu hai duong RD khong co khoang quality giao nhau; khi do ham se bao loi de tranh ket qua gia.

## Kiem thu

```bash
python -m pytest -q
```
