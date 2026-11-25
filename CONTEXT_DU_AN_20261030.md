# Context bàn giao dự án giao thông

Cập nhật: **30/10/2026**, múi giờ **Asia/Saigon (UTC+7)**.

Tài liệu tổng hợp đoạn chat và đối chiếu các tệp hiện có trong `D:\se363-bigdata`. Đây là ảnh chụp trạng thái tại thời điểm bàn giao, không phải trạng thái live của Kaggle hoặc thiết bị.

## 1. Phạm vi và nguyên tắc tiếp tục

- Dự án chính: phát hiện xe, trạng thái đèn, tracking và hỗ trợ phân tích vi phạm giao thông bằng YOLO/VLM.
- Workspace của cuộc trao đổi này: `D:\se363-bigdata`.
- Tab IDE gần nhất nằm trong `D:\New folder (2)\MFVLR`. Đó là dự án khác; không trộn log train/generation của MFVLR vào báo cáo giao thông và không tự sửa workspace đó.
- Người dùng muốn QA và lưu kết quả trên máy, tránh dựng thêm web. Ưu tiên giao diện QA local, ảnh/video và ghi chú dễ xem, không bắt người dùng sửa JSON thủ công.
- Chỉ báo thành công khi có bằng chứng: log hoàn tất, chỉ số hoặc video/checkpoint thực sự tồn tại. Notebook `COMPLETE` một mình không đủ chứng minh worker train thành công.
- QA một số ảnh không có nghĩa toàn bộ dataset đã được xác nhận. Video qua kiểm tra tự động không tự trở thành GT.
- Không tự xóa dataset, checkpoint, video cần xem lại hoặc hồ sơ QA ngoài phạm vi đã được duyệt. Không tự khởi động lại job train hay dùng GPU chỉ để kiểm tra trạng thái.
- **Yêu cầu mới nhất về thiết bị: đã ngắt kênh ADB điện thoại cổng 5038. Không kết nối lại, quay, mở ứng dụng hoặc chạy script dùng điện thoại nếu người dùng chưa yêu cầu lại.**

## 2. Kết quả train YOLO mới nhất đã xác minh

Đang dùng **YOLO26s (small)** cho xe và đèn. Trước đây đã train/evaluate YOLOv8 nano/small và thử một số trọng số ngoài; không nhầm các kết quả cũ với phiên YOLO26s.

Các chỉ số sau là **validation**, không phải test độc lập, tracking hay benchmark realtime end-to-end.

| Bài toán | Epoch tốt nhất | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Xe | 27 | 83,27% | 71,67% | 80,73% | 64,72% |
| Đèn | 30 | 77,59% | 59,91% | 65,82% | 35,72% |

### Xe

- Notebook mới nhất đã kiểm tra: `v5nhdf/yolo26-auto-resume-20261028-01`.
- Link: https://www.kaggle.com/code/v5nhdf/yolo26-auto-resume-20261028-01
- Kaggle trả `COMPLETE`; tệp completion xác nhận `success=true`, `worker_exit_codes.vehicle=0`.
- Resume từ epoch 16, train thêm **11 epoch**, kết thúc tại **epoch 27**, thời gian phiên train khoảng **8,445 giờ**.
- **Chưa early stopping.** Epoch cuối 27 cũng là epoch tốt nhất trong CSV đã tải. Không tự suy diễn phiên này đã đạt mục tiêu 100 epoch.
- mAP50–95 tăng từ **63,31%** ở epoch 16 lên **64,72%** ở epoch 27, tức **+1,41 điểm phần trăm**.
- Output Kaggle có liệt kê `best.pt` và `last.pt` mới. **Chưa tải checkpoint epoch 27 về máy trong lần báo cáo này**; chỉ tải CSV và metadata.
- Checkpoint xe đã có ở `weights/yolo26_autorestart_20261028/cycle_00/vehicle/checkpoints/` thuộc lần sao lưu epoch 16, không mặc định là epoch 27.
- Khi resume đúng trạng thái train, ưu tiên `last.pt` cùng cấu hình/phiên bản tương thích; không coi fine-tune từ `best.pt` là resume optimizer/epoch đầy đủ.

Nguồn xác minh:

- `runs/project363/yolo_train_report_20261030/REPORT.md`
- `runs/project363/yolo_train_report_20261030/vehicle_results.csv`
- `runs/project363/yolo_train_report_20261030/latest_completion.json`
- `runs/project363/yolo_train_report_20261030/latest_progress.json`

Lưu ý: `latest_progress.json` vẫn có `return_code=null` do là snapshot trước khi worker thoát. Completion và sự kiện `WORKER_EXIT` xác nhận mã 0; không lấy riêng snapshot cũ để kết luận job còn chạy.

### Đèn

- **Early stopping tại epoch 45**, `patience=15`; best ở epoch **30**.
- Checkpoint/log đã tải: `weights/yolo26_autorestart_20261028/cycle_00/light/`.
- `checkpoints/best.pt` là bản tốt nhất; `checkpoints/last.pt` là bản cuối.
- Theo quyết định trước đó, ưu tiên train tiếp xe; phần đèn để kiểm tra dữ liệu và cải thiện sau.
- Báo cáo chẩn đoán: `runs/project363/yolo26_light_diagnostic_20261029/report.json`.
- Validation lại tại imgsz 640 cho AP50 xấp xỉ: red **82,58%**, green **89,08%**, yellow **47,56%**, off **44,04%**. Hai lớp yellow/off còn yếu.
- Thử tăng imgsz từ 640 lên 1280 giúp mAP50 tổng từ khoảng 65,81% lên 67,33%, nhưng mAP50–95 vẫn khoảng 35,6%; không phải giải pháp đủ để kết luận đã sửa được vấn đề.
- Tốc độ ghi trong validation không phải phép đo realtime batch=1 cho toàn pipeline.

## 3. Quota Kaggle — snapshot đã kiểm tra trong chat ngày 30/10

| Tài khoản kết nối | GPU đã dùng / 30 giờ | GPU còn lại |
|---|---:|---:|
| Kaggle 1 | Khoảng 29 giờ 54 phút | Khoảng 5 phút 34 giây |
| Kaggle 2 — `v5nhdf` | Khoảng 26 giờ 7 phút | Khoảng 3 giờ 53 phút |

- Quota GPU đặt trước lúc kiểm tra: **0** trên cả hai tài khoản.
- TPU còn **20 giờ mỗi tài khoản** lúc kiểm tra.
- API trả thời điểm reset `2026-10-31T00:00:00Z`, tương ứng **07:00 ngày 31/10/2026 giờ Việt Nam**.
- Đây không phải cam kết quota còn nguyên: có dự án khác dùng tài khoản Kaggle. Phải đọc quota và job đang chạy lại trước khi submit.
- Công cụ MCP có hai kết nối riêng `kaggle` và `kaggle2`; không đổi token hoặc ghi đè tài khoản đang dùng cho dự án khác.

## 4. Dataset và QA

### Xe

- Các lớp đang dùng: **car, motorcycle, bus, truck**, theo thứ tự ID 0–3 trong `config/vehicle_dataset.yaml`.
- Các vấn đề đã được người dùng quan sát: nhầm car/truck; trọng số ngoài bỏ sót xe máy; nhiều bbox ở quá xa hoặc ảnh mờ khó nhận dạng.
- Người dùng ưu tiên giữ đối tượng nhìn rõ; không lọc máy móc quá chặt chỉ vì ảnh gốc nhỏ nếu nội dung vẫn rõ và bbox đủ lớn.
- Định hướng ảnh đầu ra dự án là **1280×720 (720p)**, không đồng nghĩa `imgsz=720`. Cấu hình train đã xác minh ở bản checkpoint xe dùng `imgsz=640`, batch 16, mục tiêu 100 epoch.
- Có bản `data/YOLO363_readable_crops_20261027/`. Metadata build ghi **165.233 crop xe**; đây không phải số ảnh nguồn độc lập.
- Người dùng đã QA nhiều mẫu keep/reject và chấp nhận bộ lọc. Tuy vậy `status.json`/`summary.json` của bản build vẫn có trạng thái draft yêu cầu QA; không tự đổi thành toàn bộ dataset đã được duyệt chỉ dựa trên lời nhận xét mẫu.
- Nguồn QA: `data/YOLO363_readable_crops_20261027/quality_review/human_review.json`.

### Đèn

- Các lớp hiện tại: **red, yellow, green, off**. Chưa có xác nhận mô hình hiện tại đã hỗ trợ đầy đủ mũi tên/hướng rẽ hay tín hiệu riêng cho xe máy.
- Đã khảo sát, bổ sung và QA nguồn đèn ngoài. Các lớp trạng thái/hướng từ nguồn khác phải kiểm tra mapping, không nhập tự động theo tên gần giống.
- Bản full-scene dùng cho thí nghiệm đèn: `data/Light363_fullscene_20261028/`.
- Theo `summary.json`: **8.653 ảnh train độc lập**, **10.211 lượt ảnh train hiệu dụng** sau lặp/cân bằng, **1.898 ảnh validation**.
- Bbox train: red 9.357, green 9.300, yellow 540, off 650. Các lớp yellow/off ít hơn rõ rệt.
- Bản này giữ toàn cảnh, downscale/pad về 1280×720, không upscale và không crop; giữ annotation nguồn. Validation không lọc hoặc lặp, chỉ đổi tên tệp, giữ nguyên nội dung ảnh/nhãn.
- `test_uploaded=false` trong metadata; không dùng validation thay cho test cuối.
- Các bản filtered/balanced/crops/expansion khác vẫn hiện diện. Đừng suy ra tất cả đang active hoặc xóa bản cũ khi chưa kiểm tra phụ thuộc và backup.

### Tracking và 15 case video

- Giữ dữ liệu tracking phục vụ đánh giá và tài liệu/metadata để mở rộng sau này; không đồng nhất độ chính xác detector với chất lượng tracking.
- Bộ 15 đoạn video biên tập: `data/VLM363_working/external_compilation_15/`.
- Video có tại `clips/case_01.mp4`…`case_15.mp4`; bản kèm QA trong `gt_draft_revision_001/case_01/`…`case_15/` theo README.
- Đây là **15 phần biên tập**, không tự coi mỗi phần là một xe vi phạm hay GT vượt đèn đỏ. Có replay, zoom, overlay, chuyển cảnh; cần QA và phân tách trước khi đánh giá.
- Người dùng đã sửa nhận định về các video nguồn ở batch trước: không có xe vượt vạch/vượt đèn đỏ theo quan sát của họ. Không giữ nhãn vi phạm cũ chỉ vì JSON từng ghi `OBSERVED_CROSSING`.
- Không tự biến file hình học chưa đủ bằng chứng thành ground truth, không bịa frame trước/sau hoặc timestamp.

## 5. iHanoi — kiểm kê và lựa chọn camera

Gốc dữ liệu: `data/iHanoi_capture/`.

- Khảo sát có **87 camera** trong danh mục; chưa xác minh đây là toàn bộ camera Hà Nội.
- Quyết định người dùng hiện có: **7 giữ để theo dõi đèn/vạch**, **38 cần xem lại**, **38 bỏ qua**.
- Có **4 camera bị ẩn theo bộ lọc nền đen**, chưa có lựa chọn QA thủ công; không tự coi chúng đã được người dùng duyệt xóa.
- Người dùng đã yêu cầu **loại cả camera chỉ có giá trị tracking** trong đợt thu thập iHanoi này.
- Nguồn quyết định ưu tiên: `data/iHanoi_capture/city_survey_20261029/human_choices.json`. Metadata capture cũ có thể khác với lựa chọn mới.

### 7 camera đã chọn góc đèn/vạch

1. C069.10-PTZ
2. C057.08-PTZ
3. C040.08-PTZ
4. C047.10-PTZ
5. C042.08-PTZ
6. C183.31-PTZ
7. C148.14-PTZ

Giữ góc camera không đồng nghĩa mỗi lần quay mới đều có chất lượng tốt. Ví dụ C067 từng bị catalog cũ đánh reject nhưng lựa chọn hiện tại là `CAN_XEM_LAI`; không xóa theo quyết định lịch sử đã bị thay thế.

### Dọn dữ liệu đã thực hiện và được duyệt

- Xóa **198 tệp** liên quan **38 camera BO_QUA**, gồm clip, bản sao, preview và log riêng.
- Xóa **38 thư mục rỗng**, giải phóng **265.510.559 byte**, khoảng **266 MB** theo hệ thập phân.
- Giữ 7 camera chọn, 38 camera cần xem lại, 4 camera ẩn chưa quyết định và hồ sơ QA/kiểm kê chung.
- Kiểm tra sau dọn xác nhận tệp cần giữ còn tồn tại, lựa chọn người dùng không đổi và danh mục đã đánh dấu dữ liệu bị xóa.
- Manifest: `runs/project363/ihanoi_cleanup_20261029/manifest.json`.

## 6. Kết quả quay có giám sát — không nhầm với video khảo sát cũ

Yêu cầu người dùng:

- Chỉ quay **một instance/thiết bị**.
- Chỉ bắt đầu khi hình camera đã rõ và được phóng to.
- Giám sát trong khi ghi; phát hiện vùng xám, đứng hình/lag hoặc mất khả năng giám sát thì hủy clip.
- Ưu tiên camera có đèn và vạch. Video đạt kiểm tra phải tách riêng và còn cần người dùng QA.

### BlueStacks

- Phiên `guarded_20261029_185125` thử 13/45 camera: 1 clip giữ, 3 hủy, 9 chưa đủ điều kiện.
- Phiên tiếp theo `guarded_20261029_201426` thử thêm 9 camera, không có clip giữ. Tổng hai đợt là 22/45 đã thử, không phải 22 clip thành công.
- Clip mới duy nhất giữ được từ các đợt guarded đã báo cáo:
  `data/iHanoi_capture/guarded_20261029_185125/C069_10-PTZ.mp4`.
- Clip này khoảng **30 giây**, khoảng **21,3 FPS ghi màn hình**, **9,2 MB**, trạng thái `PENDING_HUMAN_QA`. FPS này không phải FPS gốc của camera.
- Người dùng đã tắt BlueStacks và chuyển sang điện thoại; không tự mở lại emulator.

### Điện thoại

- Điện thoại OPPO Reno12 Pro 5G, model CPH2629, độ phân giải màn hình đã đọc **1080×2412**.
- Android Platform-Tools chính thức được đặt trong `.tools/android/platform-tools/`; kênh ADB riêng đã dùng là cổng **5038**.
- `screenrecord` ghi vào `/sdcard` và thư mục tạm đều bị từ chối quyền ghi. Chế độ H264 thô qua USB cũng bị treo; không dùng lại cách này cho điện thoại.
- Tiến trình cũ PID 23204 không cho ADB gửi tín hiệu dừng. Người dùng đã reboot điện thoại; lần kiểm tra sau đó xác nhận tiến trình cũ hết. Trạng thái block trong thư mục phiên cũ là hồ sơ lịch sử, không chứng minh tiến trình vẫn đang sống sau reboot.
- Đã tải **scrcpy 4.1 chính thức**, kiểm tra SHA-256, đặt tại `.tools/scrcpy/scrcpy-win64-v4.1/`.
- Script chuyển sang scrcpy không ghi âm, không mở cửa sổ, không phát lại, không bật điều khiển scrcpy; điều hướng camera vẫn thông qua thao tác ADB đã kiểm tra.
- Phiên scrcpy `guarded_20261029_221257` đã thử 8 camera: **7 chưa qua kiểm tra trước ghi, 1 clip hủy vì phát hiện đứng hình; không giữ được video mới trên điện thoại**. Phiên này đã kết thúc.
- Cần phân biệt lỗi tải/màn hình/bộ lọc với camera thực sự hỏng. Trước đó bộ lọc đã nhầm bề mặt xám có texture với vùng xám lỗi; đã sửa điều kiện yêu cầu vùng gần như phẳng và bổ sung test, nhưng chưa xác minh được một clip điện thoại thành công sau sửa.
- Không khẳng định phương án mới đã hoạt động chỉ vì công cụ khởi chạy hoặc tệp log được tạo.

Nguồn liên quan:

- `scripts/record_ihanoi_guarded.py`
- `tests/test_ihanoi_guarded.py`
- `data/iHanoi_capture/README_PHONE_CAPTURE.md`
- `data/iHanoi_capture/guarded_20261029_221257/status.json`

Script có chờ tối đa 45 giây mỗi lần mở, yêu cầu 5 giây hình ổn định, kiểm tra đứng hình ngưỡng 2 giây. Đây là heuristic lấy mẫu, không bảo đảm phát hiện mọi lỗi ngay tức thì. Người dùng báo phiên xem app có giới hạn khoảng 90 giây; không giải quyết bằng chờ vô hạn.

Timestamp scrcpy là timestamp encoder của điện thoại, không phải timestamp gốc camera. Video màn hình chưa crop thành 1280×720 và `timebase_verified=false`; không tự dùng làm GT thời gian vượt vạch.

### Trạng thái thiết bị cuối cùng trong chat

**Người dùng yêu cầu ngắt kết nối ADB. Đã dừng server ADB cổng 5038. Không được tự kiểm tra ADB bằng lệnh có thể tự khởi động lại server, không tự kết nối lại điện thoại.**

Không có bằng chứng mới trong chat cho thấy đã quay được thêm clip sau báo cáo thất bại nói trên. Các tác vụ tiếp theo nên tập trung vào train/dataset hoặc xin yêu cầu mới trước khi động tới thiết bị.

## 7. Công cụ, cấu hình và bảo mật

- Python đã dùng: `C:/Users/NHA/miniconda3/envs/my_env/python.exe`.
- Cấu hình train local: `config/`; server: `config/server/`; Kaggle: `config/kaggle/`.
- Trên Kaggle có rclone remote `gdrive2` đã từng thử upload thành công. Không suy ra remote thuộc email nào nếu chưa kiểm tra được danh tính.
- Đã có giai đoạn Drive đầy; có khoảng thời gian người dùng chỉ còn 6 GB. Mong muốn backup là giữ bản mới nhất và best, tránh tích lũy checkpoint mỗi epoch vô hạn. Phải đọc script/config thực tế trước khi khẳng định chính sách hiện đang áp dụng.
- Token OAuth/refresh token và client secret từng xuất hiện trong lịch sử chat. Không chép chúng vào context, log, notebook public hoặc commit; cân nhắc thu hồi/đổi nếu chưa làm.
- Không in giá trị biến môi trường token, nội dung cấu hình rclone hoặc JSON OAuth khi chẩn đoán. Không tự cập nhật token của tài khoản khác.

## 8. Việc nên làm tiếp theo

1. **Xe:** nếu người dùng muốn resume, kiểm tra quota và job đang chạy; tải/xác minh checkpoint epoch 27 cùng CSV/cấu hình, lưu riêng để không ghi đè checkpoint epoch 16 trước khi kiểm chứng. Chưa được tự submit vì việc lập context không phải lệnh train.
2. Dùng `last.pt` cho resume đầy đủ khi tương thích; kiểm tra epoch, class mapping, phiên bản Ultralytics và dữ liệu trước khi chạy. Không bịa rằng đã đủ 100 epoch hoặc đã early stop.
3. **Đèn:** rà soát bbox nhỏ, nhãn/mapping và các lớp yellow/off; đối chiếu domain toàn cảnh/crop và lỗi dự đoán trước khi tăng epoch hoặc thêm dữ liệu hàng loạt.
4. Nếu cần đánh giá triển khai, benchmark riêng batch=1 trên phần cứng đích và pipeline detector + tracking. Không chuyển trực tiếp ms/image validation thành kết luận realtime.
5. **iHanoi:** giữ clip C069 cho người dùng QA. Chỉ tiếp tục thử điện thoại khi được yêu cầu lại; cần một pilot thực sự tạo video hợp lệ rồi mới mở rộng camera.
6. Giữ 15 case biên tập ở trạng thái cần QA cho tới khi có kết luận và bằng chứng theo từng xe/sự kiện. Không tự gán tất cả là vượt đèn đỏ.

## 9. Checklist cho phiên hỗ trợ sau

- Đọc tài liệu này và các file nguồn được dẫn; ưu tiên tệp QA hiện hành hơn metadata lịch sử.
- Báo cáo train: đọc CSV/checkpoint mới nhất và completion thực tế; chỉ rõ chỉ số train/val/test.
- Trạng thái Kaggle/quota trong tài liệu đã cũ ngay khi có job khác chạy: phải truy vấn lại, nhưng truy vấn không đồng nghĩa được phép khởi chạy train.
- **Không chạy ADB khi chưa được người dùng cho phép lại.**
- Chỉ nói đang quay/train nếu tiến trình thực sự đang chạy; chỉ nói có video/model nếu tệp hợp lệ tồn tại và đã kiểm tra.
- Giữ thay đổi tập trung vào dự án giao thông; không trộn sang MFVLR dù IDE đang mở tệp ở dự án đó.
