# Đánh giá đèn A theo §19.2 (26/11/2026)

Script: `scripts/eval_vehicle_6class_20261126.py --names red,yellow,green,off` (IoU 0,5, conf 0,25, ảnh 960). Model: `weights/yolo26_light_ab_20261031/A_fullscene/checkpoints/best.pt`. Đây là số đo detector trên từng ảnh, không phải độ đúng pha đèn của pipeline (95,2% trên iHanoi sau ROI + đọc vị trí bóng + bỏ phiếu).

| Tập | red | yellow | green | off (không tính mục tiêu) |
|---|---|---|---|---|
| Test gốc (890 ảnh Bosch/Kaggle) | 74,7% (803) | 21,1% (90) | 81,3% (973) | 20,5% (234) |
| VN pool train (361 ảnh, A chưa thấy) | 82,3% (254) | 50,0% (34) | 63,0% (100) | — |
| VN pool val (51 ảnh) | 88,2% (34) | 38,9% (18) | 82,4% (17) | — |

## Lỗi chính
- **Bỏ sót đèn, không phải sai màu.** Test: red bị sót 188, green 167. Đọc sai màu chỉ đáng kể ở **vàng → đỏ** (test 27/90, VN 13/52).
- **Test gốc: đèn nhỏ.** Recall theo chiều cao box: <8 px 24%, 8–16 px 62%, 16–32 px 84%, ≥32 px 84%.
- **VN: đèn lớn.** Recall theo chiều cao box: <16 px 87%, 16–32 px 76%, ≥32 px 60% (tại 960). Đổi cỡ ảnh không giúp được: ở 640 và 1280 đều kém hơn 960 ở mọi nhóm kích thước. Lỗi do model chưa học kiểu đèn VN lớn, không do chọn sai cỡ ảnh.
- **Precision trên VN thấp** (28–59%): một phần vì quy tắc gán nhãn VN không gán đèn mũi tên và đèn đếm giây. A vẫn khoanh các đèn đó, nên chúng bị tính là dự đoán sai.

## Hệ quả
- Detector đèn còn xa mục tiêu 95% theo từng lớp. Pipeline bù lại bằng ROI cột đèn, đọc vị trí bóng cho đèn 3 bóng và bỏ phiếu theo thời gian.
- Muốn A tốt hơn cần thêm **ảnh VN đã QA có đèn lớn và đèn vàng**. Các lần trộn dữ liệu ngoài (LISA, DualCam) đều làm hỏng đèn VN (xem `weights/yolo26_light_*/STATUS.md`).
