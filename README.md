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

Trạng thái ngày 26/11/2026:

1. **Xong: nhãn MIO-TCD.** Map lại truck theo lớp gốc trên bản nguyên khung Vehicle363 (train/val/test), bằng `scripts/remap_mio_6class_20261124.py`.
   - pickup_truck → pickup, work_van → minibus.
   - Không còn khung nào không ghép được.
2. **Xong: 355 ảnh minibus/van bổ sung** từ UA-DETRAC và HeteroTraffic, đã QA (`scripts/build_vehicle_minibus_add_20261122.py`).
3. **Xong: QA 300 khung truck HCMC.** 97% khung nhìn rõ là truck thật, nên **giữ nguyên nhãn HCMC**.
   - CLIP chỉ đúng 72%, không dùng để gán nhãn.
4. **Xong: bộ VN 100 ảnh.** QA 87 khung truck: 39 minibus, 3 pickup, 31 truck, 14 làm mờ (`scripts/build_vehicle_vn_pool_6class_20261125.py`).
5. **Xong: code chuyển sang 6 lớp.**
   - Pipeline nhận bảng lớp của checkpoint, chỉ chấp nhận bảng 4 lớp cũ hoặc 6 lớp.
   - Đã cập nhật luật, tracking, nối track và JSON schema.
   - Có test đi kèm.
6. **Đang chạy: train trên Kaggle.** Notebook `config/kaggle/yolov8s_vehicle_6class_20261126/train.py`, khởi tạo từ v8s, theo công thức gốc, train 10 h.
   - Dev = val gốc. Tập test gốc **không** tải lên, chỉ mở một lần sau khi chốt model.
7. **Đánh giá:** dùng `scripts/eval_vehicle_6class_20261126.py`, đo recall theo §19.2 trên tập test gốc và tập camera VN (`scripts/build_vn_camera_test_6class_20261126.py`, 73 ảnh). Mốc của v8s 4 lớp ở ảnh 960:

   | Lớp | Test gốc | Camera VN |
   |---|---|---|
   | car | 91,8% | 94,1% |
   | motorcycle | 88,0% | 87,4% |
   | bus | 82,9% | 100% (8 khung) |
   | truck | 90,8% | 83,3% |
   | pickup | 0% | 0% |
   | minibus | 0% | — |

   - Precision trên tập test bị thấp hơn thực tế vì nhãn HCMC sót nhiều xe (đã xem mẫu ngày 26/11).
8. **Chưa làm:**
   - Chọn model trên dev, rồi chạy test một lần.
   - Tạo cấu hình tracking mới (SHA-256 của model mới) và chạy lại pipeline.
   - Lọc khung truck HCMC mà model mới đoán là pickup/minibus để QA.

**Thời gian nguồn (§9.1):** từ 26/11, pipeline lấy thời gian khung từ PTS của decoder, không dùng FPS danh nghĩa. Video iHanoi quay màn hình có khung lệch tới ~0,2 s so với FPS danh nghĩa. Manifest ghi `timebase`. Kết quả pilot (5/8 vụ, 0 báo nhầm) đo bằng cách cũ, cần chạy lại để xác nhận.

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
