# YOLO26s trên Kaggle2, xe/đèn song song

- Chủ notebook: v5nhdf, notebook mới private, không ghi đè notebook YOLOv8.
- Accelerator NvidiaTeslaT4: cần 2 GPU visible. Xe GPU0, đèn GPU1;
  mỗi worker dùng device=0 bên trong CUDA_VISIBLE_DEVICES riêng.
- Khởi tạo cả hai từ yolo26s.pt COCO pretrained, không resume YOLOv8.
- Ultralytics 8.4.178, imgsz640, batch16/GPU, AMP, workers2/GPU,
  tối đa 100 epoch, patience15. Dừng ở cuối epoch khi đã đạt 10 giờ/worker;
  submit timeout12 giờ để còn thời gian kết thúc và lưu output.
- Đèn không dùng hue augmentation và mosaic để tránh đổi nhãn màu/thu nhỏ đèn thêm.
- Train xe: 165233 crop; train đèn: 10810 crop. Val xe giữ nguyên 21460 frame,
  val đèn giữ nguyên 1898 frame kết hợp. Test không upload và không dùng chọn model.
- Dataset được upload private, chỉ ZIP ảnh/nhãn và metadata, không kèm secrets.

Checkpoint sau mỗi epoch ghi đè last.pt và best.pt trong:
`/kaggle/working/yolo26s_20261027/<vehicle|light>/checkpoints/`.
Bản copy checkpoint này giữ optimizer trước khi Ultralytics strip bản cuối.
Không lưu toàn bộ lịch sử từng epoch. epoch_status.json có SHA256 và số epoch.

Launcher chạy foreground chờ cả hai worker; không chỉ nohup rồi thoát notebook.
Nếu thiếu hai GPU hoặc một worker lỗi, notebook báo lỗi, không tự đổi CPU.

Chưa có Secret Drive được cấp riêng cho notebook mới: chưa bật backup rclone.
last/best được lưu trong working và trở thành output khi phiên lưu thành công;
không bảo đảm lấy lại được nếu Kaggle thu hồi phiên trước khi lưu output.
Không đính kèm dataset rclone hoặc token/OAuth JSON để lách cấp quyền Secrets.
