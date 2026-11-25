# SE363: Phát hiện vượt đèn đỏ (YOLO + tracking)

Phần YOLO và tracking của đồ án Traffic Violation Understanding. Từ video camera giao thông, hệ thống phát hiện xe, đọc pha đèn, theo vết từng xe và báo xe vượt vạch khi đèn đỏ.

Repo chỉ chứa mã, cấu hình và báo cáo. Dữ liệu và trọng số không có ở đây (xem mục *Dữ liệu và trọng số*).

## Luồng xử lý

`scripts/run_red_light_pipeline.py --scene-config configs/scene_configs/<camera>.json`

1. **Xe:** YOLOv8s ở 960 px (`weights/kaggle_small_vehicle_20261019/best.pt`) + ByteTrack đã chỉnh (`configs/tracking/bytetrack_v8s_imgsz960_tuned_20261114.yaml`). Có 4 lớp: car, motorcycle, bus, truck.
2. **Đèn** (`--light-reader k1bpos`, mặc định), xét trong ROI cột đèn khai báo ở scene config:
   - Đèn 3 bóng có khai báo `lamp_layout` (`vertical3`/`horizontal3`): đọc theo **vị trí bóng sáng** (`src/signal/lamp_position.py`). Vàng luôn ở giữa.
   - Còn lại: box của mô hình đèn A (YOLO26s, `weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt`) được đọc lại màu bằng bộ phân loại `weights/light_colour_cls_20261117/best.pt`. Nếu không có box thì phân loại cả ROI.
   - Bỏ phiếu pha đèn trong 2 giây (`src/signal/phase_voting.py`).
3. **Luật:** nối vết bị đứt (`src/rules/track_stitch.py`), sau đó áp luật vượt đèn đỏ (`src/rules/red_light.py`).

Scene config (`configs/scene_configs/*.json`) khai báo ảnh tham chiếu, ROI từng cột đèn, vạch dừng và làn. Cột đèn có bóng mũi tên hoặc đèn đếm giây **không** được khai báo `lamp_layout`.

## Kết quả hiện tại

| Hạng mục | Số đo | Ghi chú |
|---|---|---|
| Xe, camera VN chưa train (73 ảnh) | recall 92,1%, precision ~97% | conf 0,25, IoU 0,5 |
| Tracking vntraffic 201–300 | HOTA 0,634, IDF1 0,793 | nhãn đã QA (tạm) |
| Pha đèn iHanoi (188 s) | 95,2% đúng sau bỏ phiếu | chỉ có 3 s pha vàng |
| Bản tin VN | bắt 5/8 vụ, 0 cảnh báo nhầm trên iHanoi | `runs/project363/red_light_pilot/PILOT_REPORT.md` |

Mục tiêu đồ án là **95% recall theo từng lớp**, tính cả vật thể bị sót (không phải mAP). Lớp đèn `off` không tính vào mục tiêu.

## Các thử nghiệm không dùng (đọc trước khi train lại)

Mỗi thử nghiệm có `STATUS.md` riêng trong `weights/<tên>/` hoặc `runs/project363/<tên>/`.

- **Đèn:**
  - Fine-tune thêm DualCam: kém hơn A (90,4%).
  - Train từ đầu trên A + VN + DualCam: kém hơn A (69%).
  - Thêm 182 ảnh vàng sạch: mất đèn LED lớn của C069.
  - Nghi một phần do dao động giữa các lần train. Chưa chạy đối chứng công thức A trên chính dữ liệu A.
- **Xe:** fine-tune thêm ảnh VN cho kết quả kém hơn. Xe bị sót chủ yếu do bị cắt ở mép ảnh, không phải do ngưỡng.
- **Nhãn vàng BDD100K:** chỉ khoảng 10% đúng, đã bỏ.

## Việc đang làm dở: mô hình xe 6 lớp

Lý do: model đang gọi minibus VN là bus và gọi pickup/van là truck.

Lớp mới: 0 car, 1 motorcycle, 2 bus, 3 truck, 4 pickup, 5 minibus. Quy ước:
- minibus = xe ≤16 chỗ và mọi loại van, kể cả van chở hàng;
- MPV/minivan 7 chỗ = car;
- container = truck.

Chi tiết ở `data/vn_youtube_train_pool_20261116/LABELING_RULES.md`.

1. **Xong:** map lại nhãn truck của MIO-TCD theo lớp gốc (`scripts/remap_mio_6class_20261124.py`):
   - pickup_truck → pickup (25.237 khung train), work_van → minibus (5.252);
   - chỉ 3 khung không ghép được.
2. **Xong:** thêm 355 ảnh minibus/van đã QA từ UA-DETRAC và HeteroTraffic (`scripts/build_vehicle_minibus_add_20261122.py`).
3. **Đang làm:** QA 300 khung truck của bộ HCMC (`scripts/hcmc_truck_survey_20261125.py`). Bộ này vẫn lẫn pickup và minibus.
   - Gợi ý của CLIP chỉ đúng khoảng 73% (pickup 58%), không dùng để gán nhãn tự động.
4. **Chưa làm:**
   - Gán lại khoảng 12.000 khung truck của HCMC.
   - Đổi 49 khung xe 16 chỗ VN từ truck sang minibus.
   - Sửa mọi chỗ đang giả định 4 lớp: `NAMES`, nhóm lớp khi nối vết, luật, cấu hình tracking.
   - Tìm tập test để map lại.
   - Train trên Kaggle với công thức của v8s.
   - Đánh giá recall theo lớp.

## Dữ liệu và trọng số

Không đưa lên GitHub vì nặng (model xe 111 MB, vượt giới hạn 100 MB/file của GitHub) và nhiều bộ dữ liệu không cho phát hành lại. Hãy hỏi người phụ trách để lấy:

| Cần cho | Đường dẫn trong repo |
|---|---|
| Pipeline | `weights/kaggle_small_vehicle_20261019/best.pt`, `weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt`, `weights/light_colour_cls_20261117/best.pt` |
| Train xe | Vehicle363 (MIO-TCD + HCMC), bản lưu trên Kaggle `yolo363-vehicle-archive-20261027`; nhãn 6 lớp sinh bằng script ở trên |
| Video thử | thư mục `data/` theo đường dẫn ghi trong từng scene config |

Pipeline kiểm tra SHA-256 của model xe (`expected_sha256` trong file cấu hình tracking), nên phải dùng đúng file.

**Không dùng** tập Roboflow VN và tập iHanoi để train. Kênh ĐTV, các nguồn test và livestream FM90 cũng không đưa vào train, vì sẽ rò rỉ sang tập đánh giá.

## Cài đặt và kiểm thử

```bash
pip install -r requirements.txt     # cài torch theo CUDA của máy trước
python -m pytest tests -q
```

Trên máy có dữ liệu: 352 test đạt, 1 test lỗi (`tests/tracking/test_frame_mapping.py`) vì thiếu video VNTraffic gốc. Test này cần dữ liệu trong `data/`. Trên Windows, nếu pytest báo `PermissionError` ở thư mục Temp, hãy thêm `--basetemp <thư mục ghi được>`.

## Thư mục

- `src/`: lõi (signal, rules, scene, tracking).
- `scripts/`: dựng dataset, QA, đánh giá, đóng gói notebook Kaggle, pipeline.
- `config/`, `configs/`: cấu hình train Kaggle, scene config, tracking.
- `tests/`: kiểm thử.
- Các file `.md` ở gốc: đặc tả hệ thống và dataset, hướng dẫn gán nhãn. `CONTEXT_DU_AN_20261030.md` là ảnh chụp trạng thái ngày 30/10, đã cũ hơn README này.
