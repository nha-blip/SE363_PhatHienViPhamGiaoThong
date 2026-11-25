# Cấu hình dữ liệu Light363 mở rộng — 26/10/2026

Đây là cấu hình dữ liệu cho máy hiện tại, chưa chạy train. Các YAML tham chiếu
danh sách ảnh trong data/TrafficLight363_expanded_candidate_20261026 và sử dụng
ảnh/nhãn ở bộ gốc hoặc pool đã lọc. Không sao chép thêm ảnh và không sửa bộ gốc.

| YAML | Train | Validation |
| --- | --- | --- |
| data.yaml | Train mở rộng, ưu tiên vàng/tắt | Validation gốc + nguồn mới |
| data_unique.yaml | Cùng các ảnh train, không lặp | Validation gốc + nguồn mới |
| data_original_val.yaml | Train ưu tiên vàng/tắt | Chỉ validation gốc |
| data_new_holdout.yaml | Train ưu tiên vàng/tắt | Chỉ holdout nguồn mới |

Tất cả YAML giữ test là **890 ảnh test gốc**, không bổ sung ảnh mới vào test.
Đánh giá validation nguồn mới bằng split=val, không gọi nó là test độc lập.

## Thống kê

- Train: 8.653 ảnh riêng, gồm 5.541 gốc + 3.112 mới.
- Train ưu tiên: 10.211 lượt ảnh/epoch; ảnh chứa vàng hoặc tắt xuất hiện tổng
  cộng ba lần, không phải ba bản sao file và không tăng số mẫu độc lập.
- Validation: 1.898 ảnh, gồm 933 gốc + 965 mới; không lặp.
- Test: 890 ảnh gốc, giữ nguyên danh sách và nhãn.

Train riêng có red 9.357, yellow 540, green 9.300, off 650 bbox.
Sau lặp ảnh: yellow có 1.620, off có 1.950 lượt bbox/epoch. Các đối tượng
đỏ/xanh cùng xuất hiện trong ảnh được lặp cũng được lặp; không phải class weight.
Mất cân bằng chưa được loại bỏ hoàn toàn.

## Chính sách chia

Các nhóm ngày/chuỗi/camera là nguyên khối. Mỗi nguồn có đại diện ở train và
validation mới. Chọn nhóm bằng phân bố nhãn/số ảnh, không dùng dự đoán model;
tỷ lệ validation mới là khoảng 23,67%, không ép đúng 20% bằng cách xé chuỗi.
Chi tiết nhóm ở new_source_split.json của thư mục dataset ứng viên.

## Lần train tiếp theo

- Chốt checkpoint khởi đầu rồi mới chạy; YAML ở đây chỉ cấu hình dữ liệu.
- So sánh data_unique.yaml và data.yaml để biết tác dụng của lặp mẫu.
- Giữ cùng seed, kích thước ảnh, batch và thiết lập train khi so sánh.
- Báo cáo riêng AP từng màu trên test gốc và validation nguồn mới; không so
  trực tiếp validation gộp với validation cũ như cùng một tập đánh giá.
- Tập ảnh mới chủ yếu dùng nhãn nguồn đã lọc kỹ thuật, không phải mọi ảnh
  đều đã QA riêng; không tuyên bố toàn bộ là GT đã xác nhận.
- Đường dẫn tuyệt đối dành cho Windows hiện tại; chưa phải gói upload Kaggle/server.
- Kiểm tra điều khoản nguồn trước khi công bố dữ liệu ghép.
