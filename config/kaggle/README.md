# Kaggle — YOLOv8 nano/small, xe/đèn

## Hai notebook, bốn GPU và backup Drive

Import `notebook_nano.ipynb` vào notebook thứ nhất và `notebook_small.ipynb` vào notebook thứ hai. Mỗi notebook cần Internet, hai GPU visible và Secret `RCLONE_CONFIG_TEXT` chứa cấu hình `[gdrive2]` dạng Base64. Bật quyền Secret riêng trên cả hai notebook; không chia sẻ notebook có credentials với người không tin cậy.

Upload thêm `train_drive.py` vào code dataset cùng `prepare_configs.py`. Chỉ upload các file code/YAML/README; tuyệt đối không upload `client_secret*.json`, `rclone.conf` hoặc token. `.gitignore` không bảo vệ khỏi upload thủ công lên Kaggle. Thay bộ secret/token đã bị lộ trước khi sử dụng.

Notebook mặc định dùng mount `/kaggle/input/datasets/v5nhdf/yolov8/kaggle`; chỉnh cell `CODE_ROOT`, `VEHICLE_ROOT`, `LIGHT_ROOT` nếu mount khác. Cell cuối chạy launcher ở foreground để chờ hai worker background: Vehicle GPU 0, Traffic Light GPU 1. Không chạy thêm các lệnh train thủ công bên dưới cùng lúc. Không chạy lại cell launcher khi worker cũ vẫn còn chạy.

Launcher kiểm tra đủ hai GPU và upload probe trước khi train. Mỗi worker có cấu hình rclone riêng dưới `/tmp`, không đưa secret vào outputs. Sau mỗi epoch, callback upload đồng bộ `last.pt`, `best.pt`, rồi `UPLOAD_COMPLETE.json` có SHA256. Thư mục Drive: `gdrive2:Project363/yolov8/<nano|small>/<run_id>/<vehicle|traffic_light>/epoch_XXXX/`. Chỉ thư mục có receipt mới là backup được xác nhận; hash trong receipt là hash file local, không phải kiểm tra download remote. Upload thất bại làm dừng worker tương ứng, worker còn lại tiếp tục; cell cuối báo lỗi sau khi cả hai kết thúc.

Giữ lịch sử mọi epoch trên Drive, không tự xóa bản cũ: cần theo dõi dung lượng. Local chỉ giữ last/best (không tạo snapshot mỗi 5 epoch). `best.pt` là best-so-far, không nhất thiết của epoch đang upload. Sau train có backup `final/`; checkpoint final có thể bị strip optimizer, ưu tiên `epoch_XXXX/last.pt` khi cần resume. Launcher này chỉ khởi tạo fresh run, chưa hỗ trợ resume.

Cell cuối in trạng thái worker/epoch backup mỗi 30 giây. Chi tiết ở `/kaggle/working/yolo363_training/<run_id>/<task>/train.log`, `training_status.json`, `backup_status.json`. Giữ cell chờ trong Save & Run All; worker không tồn tại ngoài thời hạn phiên Kaggle. Ngắt cell có thể dừng hai worker, không đảm bảo epoch đang chạy được lưu. 100 epoch vẫn là tối đa; patience có thể dừng sớm.

Sáu YAML trong thư mục này là mẫu. Dataset path dùng slug minh họa `vehicle363-v1` và `trafficlight363-v1`; tên mount thật có thể khác. Không dùng đường dẫn Windows hoặc config Slurm trên Kaggle.

## Chuẩn bị

Upload riêng Vehicle363-v1.zip, TrafficLight363-v1.zip trong `exports/yolo363_server_20261013/` thành Kaggle Dataset rồi attach notebook. Attach thêm thư mục `config/kaggle/` làm code dataset. Cần mạng để cài Ultralytics/tải pretrained weights, hoặc attach weights có sẵn. Không chạy Slurm trên Kaggle.

Xem cây `/kaggle/input` để tìm thư mục thật chứa `images` và `labels`. ZIP đã đóng gói có prefix `data/Vehicle363-v1` hoặc `data/TrafficLight363-v1`; layout mount phụ thuộc cách upload/giải nén. Nếu chỉ thấy file ZIP, giải nén vào thư mục riêng dưới `/kaggle/working`, kiểm tra dung lượng trước và trỏ root vào bản đã giải nén.

```python
!find /kaggle/input -maxdepth 5 -type d -name images
```

Tạo config runtime bằng helper (thay cả ba đường dẫn dưới đây bằng mount thật). Helper dùng namespace class của hai working dataset đã đóng gói, không cần source data.yaml có đường dẫn Windows và không chỉnh ảnh/nhãn.

```python
!python /kaggle/input/<code-dataset>/config/kaggle/prepare_configs.py --vehicle-root /kaggle/input/<vehicle-dataset>/data/Vehicle363-v1 --traffic-light-root /kaggle/input/<light-dataset>/data/TrafficLight363-v1 --check-pairs
```

Kết quả: hai YAML dataset và bốn YAML train ở `/kaggle/working/configs`. `--check-pairs` kiểm tra pairing và số ảnh theo release, không thay thế visual QA hoặc kiểm tra CRC. Helper từ chối ghi đè config khác nội dung và không chạy train. Các mẫu YAML bên cạnh để đọc/điều chỉnh; ưu tiên dùng các file helper sinh từ root thật.

## Train — chọn từng lệnh

Sau khi cài Ultralytics vào notebook và bật GPU, chuyển cwd sang thư mục writable. Model basename được tải tại working; không tải vào input readonly.

```python
%cd /kaggle/working
!yolo detect train cfg=/kaggle/working/configs/vehicle_nano_train.yaml
!yolo detect train cfg=/kaggle/working/configs/vehicle_small_train.yaml
!yolo detect train cfg=/kaggle/working/configs/traffic_light_nano_train.yaml
!yolo detect train cfg=/kaggle/working/configs/traffic_light_small_train.yaml
```

Mặc định cả bốn dùng GPU 0, chạy lần lượt/chọn một lệnh, không khởi chạy đồng thời trên một GPU. Nếu có hai GPU visible và muốn chọn GPU thứ hai, thêm `device=1` vào lệnh phù hợp; không mặc định notebook nào cũng có hai GPU.

100 epoch là ngân sách tối đa; patience 20 có thể dừng sớm, không bảo đảm hoàn tất trong một phiên Kaggle. Nano batch 8, small batch 4, workers 2, imgsz 640, AdamW lr0=0.001. Hết VRAM thì giảm batch. Hue/saturation của dataset đèn tắt để bảo toàn trạng thái màu; class off giữ đúng chuỗi `'off'`.

Outputs/checkpoints ở `/kaggle/working/runs/<task>_<variant>_100/`; lưu/tải backup trước khi phiên kết thúc. Config là train mới, không resume. `cache: false` chỉ tắt cache ảnh; loader vẫn có thể thử tạo cache label và bỏ qua nếu input readonly. Dataset nguồn không sửa.
