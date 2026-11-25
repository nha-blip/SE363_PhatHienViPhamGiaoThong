# Thử nghiệm đèn A/B — 31/10/2026

So sánh hai cấu hình trên tài khoản `nhatquang6`, chạy song song trên hai GPU.

| Thử nghiệm | Dữ liệu train |
|---|---|
| `A_fullscene` | Full-scene hiện có (10.211 lượt ảnh) |
| `B_fullscene_lisa` | Như A, cộng 643 frame LISA; frame có yellow xuất hiện ba lần (tổng 11.030 lượt ảnh) |

Giữ giống nhau: model khởi tạo YOLO26s COCO, seed 363, imgsz 640, batch 16, 100 epoch, patience 15, các tham số tăng cường của lần train đèn trước, Ultralytics 8.4.178.

Đánh giá `best.pt` trên hai tập, không dùng test:

- Validation hiện có: 1.898 ảnh, giữ nguyên nội dung.
- LISA giữ riêng: `daySequence2`, `nightSequence2`, 255 frame.

Giới hạn cần ghi khi báo cáo:

- Không có dữ liệu Việt Nam trong train hay đánh giá; kết quả không cho biết hiệu quả trên camera Việt Nam.
- LISA chưa được QA từng ảnh; một số đèn có thể chưa gán nhãn.
- LISA quay từ trên xe và không có lớp off.
- Một lần chạy với một seed; chênh lệch nhỏ chưa đủ để kết luận.
- Tốc độ validation không phải benchmark realtime.
- Giấy phép LISA là CC BY-NC-SA 4.0: chỉ phi thương mại, cần ghi nguồn và chia sẻ tương tự.
