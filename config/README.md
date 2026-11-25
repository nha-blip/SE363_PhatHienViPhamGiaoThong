# Config YOLOv8 local — Windows

## Kaggle nano/small

Config riêng xe/đèn nano/small và helper dùng root mount thật: `config/kaggle/README.md`. Có sáu YAML mẫu; helper sinh sáu YAML runtime tại `/kaggle/working/configs`, mặc định GPU 0 cho chạy lần lượt. Không dùng các đường dẫn Windows trong file local trên Kaggle.

## Gói upload server

Gói xe/đèn từ working dataset hiện có: `exports/yolo363_server_20261013/`. Xem `config/server/README.md` để giải nén, kích hoạt Conda và sinh 6 YAML theo đường dẫn server bằng `python config/server/configure_server.py`. Không dùng các YAML local chứa đường dẫn `D:/` trên Linux. Config server có hai biến thể nano/small cho mỗi task, train mới chứ không resume.

Sau cleanup 2026-10-13, nguồn raw detection đã được xóa local theo phê duyệt. Working dataset vẫn giữ đầy đủ; không chạy lại bước copy từ nguồn bên dưới. Nếu cần rebuild từ nguồn, khôi phục từ Drive trước.

## Gói upload server

Gói xe/đèn từ working dataset hiện có: `exports/yolo363_server_20261013/`. Xem `config/server/README.md` để giải nén, kích hoạt Conda và sinh 6 YAML theo đường dẫn server bằng `python config/server/configure_server.py`. Không dùng các YAML local chứa đường dẫn `D:/` trên Linux. Config server có hai biến thể nano/small cho mỗi task, train mới chứ không resume.

Sau cleanup 2026-10-13, nguồn raw detection đã được xóa local theo phê duyệt. Working dataset vẫn giữ đầy đủ; không chạy lại bước copy từ nguồn bên dưới. Nếu cần rebuild từ nguồn, khôi phục từ Drive trước.

Các YAML dùng workspace `D:/se363-bigdata`, không dùng đường dẫn Kaggle. Nếu chuyển dự án, cập nhật `path`, `data` và `project` trong các YAML.

## Bảo toàn dataset frozen

Dataset YAML trỏ vào bản working riêng:

- `data/YOLO363_training_working/Vehicle363-v1`
- `data/YOLO363_training_working/TrafficLight363-v1`

Các thư mục này CHƯA được tạo bởi việc tạo config. Không chạy training trước khi chuẩn bị dữ liệu. `cache: false` tắt cache ảnh, không bảo đảm loader không tạo cache nhãn. Vì vậy không đổi `path` về dataset frozen. Không dùng junction/symlink trỏ nguyên labels về nguồn.

Nếu đủ dung lượng cho bản sao ảnh/nhãn, chạy PowerShell dưới đây. Chỉ copy; không `/MIR`, không xóa, move hoặc sửa nguồn. Không chạy lại bước copy trên working dataset đã được hiệu chỉnh riêng.

```powershell
cd D:\se363-bigdata
foreach ($dataset in @('Vehicle363-v1', 'TrafficLight363-v1')) {
    $destination = "data\YOLO363_training_working\$dataset"
    if (Test-Path -LiteralPath $destination) {
        throw "Working directory already exists: $destination. Inspect it before copying."
    }
    foreach ($kind in @('images', 'labels')) {
        robocopy "data\$dataset\$kind" "$destination\$kind" /E /COPY:DAT /R:1 /W:1 /XF *.cache
        if ($LASTEXITCODE -ge 8) {
            throw "Copy failed: $dataset/$kind. Do not train with an incomplete copy."
        }
    }
}
```

`robocopy` exit codes 0..7 không phải lỗi copy thất bại. Nếu copy bị ngắt, kiểm tra và hoàn thiện working area trước khi train; không bỏ qua việc kiểm tra số ảnh/nhãn. Expected images: Vehicle train/val/test = 110478/21460/21236; Traffic Light = 5541/933/890. Traffic Light có nhãn rỗng được duyệt làm background; không tự xóa chúng.

## Cài gói và kiểm tra GPU

```powershell
conda activate my_env
python -m pip install ultralytics PyYAML
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

Chỉ dùng `device=0` khi PyTorch nhận CUDA. Nếu chưa nhận GPU, kiểm tra driver và bản PyTorch CUDA phù hợp; không tự nâng Torch/CUDA tùy ý. Ghi lại phiên bản Ultralytics/PyTorch trước run dài.

## Lệnh train

```powershell
yolo detect train cfg="D:/se363-bigdata/config/vehicle_train.yaml"
yolo detect train cfg="D:/se363-bigdata/config/traffic_light_train.yaml"
```

Cả hai mặc định GPU 0: chạy lần lượt trên máy một GPU. Nếu có hai GPU và đủ RAM/CPU, mở hai terminal và override lệnh Traffic Light bằng `device=1`. Không khởi chạy hai lần cùng một run. Nếu thiếu VRAM, override `batch=4`; nếu lỗi multiprocessing Windows, thử `workers=0`.

Weights nằm dưới `runs/local_training/<run-name>/weights/`. Run có thể được Ultralytics thêm hậu tố nếu tên đã tồn tại; xem đường dẫn thực trong log. Hai config khởi tạo run mới, không resume và không tự upload Drive. Resume checkpoint `last.pt` là thao tác riêng; không chạy lại config này để tiếp tục epoch.

Việc tạo các config không chạy YOLO, không copy dataset và không thay đổi nguồn/frozen/tracking GT.

## Resume checkpoint Vehicle epoch 5

`vehicle_resume.yaml` dùng `weights/checkpoint_epoch_0005.pt`, được kiểm tra là Vehicle 4-class, epoch lưu = 4 (0-based, đã hoàn tất 5 epoch), tổng kế hoạch 100 và còn optimizer/scaler. Resume thật bắt đầu epoch 6 và còn 95 epoch; KHÔNG đặt `epochs=95`. Checkpoint gốc giữ nguyên.

Đường dẫn Kaggle trong checkpoint được thay bằng dataset YAML local và `save_dir` local qua các override được Ultralytics 8.4.173 hỗ trợ. Không reset optimizer hoặc learning-rate schedule. Batch giảm 8 -> 4 cho GPU RTX 4050 6 GB; workers=0. patience=0 tắt early stopping để hướng tới đủ 100 epoch; vẫn có thể bị ngắt do lỗi hoặc người dùng dừng.

```powershell
python scripts/resume_vehicle_local.py --check-only
python -u scripts/resume_vehicle_local.py
```

Script kiểm tra đủ ảnh/nhãn, class namespace, phiên bản, CUDA và optimizer trước khi chạy; từ chối ghi đè run đã tồn tại. Không chạy hai lần đồng thời. Kết quả: `runs/local_training/vehicle_resume_epoch5_100_adamw_compatible/weights/last.pt` và `best.pt`. Nếu đã train tiếp đến epoch cao hơn, dùng checkpoint last.pt mới, không chạy lại từ checkpoint epoch 5. Không chạy Traffic Light đồng thời trên GPU 0.

Checkpoint Kaggle dùng state AdamW có moment FP16/FP32 khác nhau; local PyTorch 2.7 fused AdamW yêu cầu cùng dtype. Script giữ model parameters FP32 trước khi restore, rồi đưa moment buffers về đúng dtype/device, không xóa hoặc reset state. Scaler, EMA, epoch và moment/step counters vẫn được restore. Sau restore, patience=0 được áp lại để checkpoint stopper cũ không bật lại early stopping. Dùng script này thay vì lệnh CLI trực tiếp cho checkpoint chuyển từ Kaggle. Run thử đầu `vehicle_resume_epoch5_100` dừng ở optimizer step đầu tiên; giữ log chẩn đoán, không dùng nó như run đã hoàn tất epoch 6.

Do kernel fused local vẫn lỗi khi step, script dùng AdamW non-fused/foreach=false. Đây là thay kernel thực thi, không đổi loại optimizer hoặc reset moments/counters; có thể chậm hơn fused. Run `vehicle_resume_epoch5_100_compatible` cũng là run thử thất bại, không có epoch 6 hoàn tất. Không nâng Torch hoặc ghi đè checkpoint nguồn để chữa lỗi này.

Đã xác định thêm state checkpoint dùng channels-last strides (ví dụ conv đầu [27,1,9,3]), trong khi parameter local thường dùng layout khác. Adapter chuẩn hóa cả stride/layout theo parameter hiện tại bằng copy giữ nguyên giá trị, không chỉ dtype/device; có test riêng đảm bảo moments và step counter không bị reset.
