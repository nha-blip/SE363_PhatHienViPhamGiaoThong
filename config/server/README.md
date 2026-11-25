# YOLO363 server upload

Hai dataset trong gói là bản working local đã hiệu chỉnh, không phải các nguồn raw đã xóa. Giữ nguyên train/val/test và namespace class. Không bao gồm tracking, VLM hoặc checkpoint đã train.

## Upload và giải nén

Upload cả ba ZIP và `SHA256SUMS.txt` từ `exports/yolo363_server_20261013/` vào cùng thư mục dự án trên server (ví dụ thư mục `MFVLR/yolo` bạn vừa chuyển). Gói không tự SSH/upload.

```bash
cd /duong/dan/MFVLR/yolo
sha256sum -c SHA256SUMS.txt
unzip -n Vehicle363-v1.zip
unzip -n TrafficLight363-v1.zip
unzip -n server_config.zip
conda activate yolo363
python config/server/configure_server.py
```

Chỉ giải nén vào project mới hoặc kiểm tra trước rằng không có dataset cũ trùng đường dẫn. `unzip -n` không ghi đè, nhưng không hợp nhất an toàn hai phiên bản dataset. Nếu project được di chuyển, chạy lại `configure_server.py` để cập nhật đường dẫn tuyệt đối.

Nếu server không có `unzip`, kích hoạt Conda rồi dùng `python -m zipfile -e Vehicle363-v1.zip .` (tương tự cho hai ZIP còn lại), chỉ trong project mới vì cách này có thể ghi đè file trùng tên.

Kết quả: `data/Vehicle363-v1/`, `data/TrafficLight363-v1/`, hai YAML dataset và bốn YAML train trong `config/server/`. Config chỉ được sinh sau khi đủ các thư mục split. Script chỉ viết config trong project này, không chạy train.

## Trọng số pretrained và GPU

```bash
cd weights
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"
cd ..
```

Trên HPC, xin GPU compute node theo scheduler của cụm trước khi chạy training; không chạy trên login01. Kiểm tra `nvidia-smi` và CUDA trên compute node đã cấp, không lấy CUDA=False trên login node làm kết luận cấu hình GPU lỗi. Script Slurm bên dưới tự kiểm tra GPU sau khi được cấp tài nguyên. Trọng số không nằm trong gói; server cần mạng để tải hoặc bạn upload hai file PT vào `weights/`.

## Train mới — chọn từng lệnh, không chạy đồng thời trên GPU 0

```bash
yolo detect train cfg=config/server/vehicle_nano_train.yaml
yolo detect train cfg=config/server/vehicle_small_train.yaml
yolo detect train cfg=config/server/traffic_light_nano_train.yaml
yolo detect train cfg=config/server/traffic_light_small_train.yaml
```

Mặc định 100 epoch, imgsz 640, AdamW, patience 20; nano batch 8, small batch 4, workers 2. Có thể dừng sớm. Hết VRAM thì thêm `batch=2`. Kết quả ở `runs/server_training/`. Không resume checkpoint epoch 30 bằng các config này.

## Slurm theo mẫu cụm UIT bạn cung cấp

`train.slurm` giữ account `uit`, partition `small`, 4 CPU, RAM 16G và GPU MIG `1g.18gb:1`. Giữ các header bạn đã chỉnh: job `mfvlr`, thời gian 36 giờ. Script đơn giản chạy lần lượt xe nano, xe small, đèn nano, đèn small; không chạy song song. 36 giờ là giới hạn chung cho cả bốn lần train, không bảo đảm hoàn thành cả bốn. Hết walltime thì cần resume lần train bị ngắt từ `last.pt`; các lần phía sau có thể chưa chạy.

Sau giải nén, configure và chuẩn bị weights, đứng tại ROOT project trên login node, tạo logs trước khi submit. Slurm mở log trước khi chạy nội dung script; `mkdir` bên trong script sẽ quá muộn.

```bash
mkdir -p logs
sbatch config/server/train.slurm
```

Không truyền task/variant vào script nữa. Dùng `/datastore/$USER/miniconda3` và Conda `yolo363`; sửa hai dòng này nếu môi trường cài nơi khác. `source conda.sh` cần cho batch shell chưa khởi tạo Conda. Giữ `set -e` để lỗi lần train trước thì không chạy tiếp, và một dòng giới hạn thread CPU. Đường dẫn project là nơi bạn gọi `sbatch`; script giữ nguyên GPU được Slurm cấp. Không dùng `bash train.slurm` trên login node. Các config phải được tạo trước bằng `configure_server.py`, và hai trọng số phải có trong `weights/`.

Theo dõi: `squeue -u "$USER"`, `tail -f logs/mfvlr_JOBID.out`. Kiểm tra trước submit: `bash -n config/server/train.slurm`. Nếu upload làm file thành CRLF, dùng `sed -i 's/\r$//' config/server/train.slurm` rồi `bash -n` lại. Header `#SBATCH` không mở rộng biến shell; đường dẫn log giữ relative để dùng ROOT project khi submit.

Namespace xe: 0 car, 1 motorcycle, 2 bus, 3 truck. Namespace đèn: 0 red, 1 yellow, 2 green, 3 off. Nhãn rỗng background được giữ; bỏ các file cache loader và không thay đổi annotation.
