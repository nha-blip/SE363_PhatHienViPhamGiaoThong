# Nguồn dữ liệu ứng viên cho pilot

Ngày khảo sát: 27/09/2026. Đối chiếu với đặc tả và hướng dẫn pilot trong workspace.

Mức kiểm chứng: đã đọc tài liệu nguồn, trang phát hành và đường dẫn tải. Chưa tải/giải mã video để xác minh trực tiếp vạch dừng, đèn, độ liên tục, FPS và điều kiện quan sát. Danh sách này là ứng viên tuyển chọn, chưa phải danh sách video được nghiệm thu.

## 1. RunningRedlight — ưu tiên bối cảnh Việt Nam

[Repository tác giả](https://github.com/namnv78/RunningRedlight) có [thư mục dữ liệu Drive](https://drive.google.com/drive/folders/1fbfE1pcY46cFzlx1Q-uBxoST-CFds6Ur?usp=sharing) và bản ảnh resize 256×256. Công cụ đọc web bị chuyển sang đăng nhập khi mở thư mục Drive; chưa xác minh tải được video gốc.

[Bài báo](https://eudl.eu/pdf/10.1007/978-3-031-08878-0_15) mô tả camera giám sát tại ba giao lộ Việt Nam, xe đi thẳng và clip khoảng 15 giây/10 FPS. Abstract ghi 108 video nhưng phần thực nghiệm ghi 105 train + 14 test: phải kiểm kê bản tải thực tế trước khi trích số lượng.

Đề xuất: lấy video toàn cảnh gốc; không dùng ảnh resize làm nguồn chính. Kiểm tra clip có đủ trước/sau vạch và nguồn video/phiên để chia tập. Nhãn nhị phân và bbox từ tracking cần kiểm chứng lại. Chưa xác minh giấy phép dữ liệu.

## 2. Dataset for Traffic Evaluation — ưu tiên khảo sát video tải trực tiếp

[Zenodo v2](https://zenodo.org/records/14041920) công bố video phục vụ phân tích giao thông và vượt đèn đỏ, khoảng 9,4 GB. Bắt đầu với `2024-11-04 12-00-31.mp4` (12,4 MB), rồi `2024-04-24 10-35-00.mp4` (40,1 MB). Tải từng file ở trang nguồn.

Đề xuất: kiểm tra chúng có phải đoạn liên tục, cùng camera hay khác phiên; không suy ra ngày/đêm chỉ từ tên file. `rlr_occasions.mp4` cần kiểm tra khả năng là tuyển tập sự kiện trước khi đưa vào test liên tục. Preview trả lỗi 429 trong lượt khảo sát. Chưa xác minh quốc gia, nội dung từng video và tên giấy phép từ phần văn bản trang trả về.

## 3. FETV — ứng viên bổ sung đánh giá VLM

[Repository chính thức](https://github.com/MoyoG/FETV) công bố 200 clip từ 9 video nguồn, khoảng 563 MiB; [Drive](https://drive.google.com/drive/folders/1VMbtNo6zzWzjxVztzy842cn7xQ0rv25G?usp=drive_link). Có nhãn sự kiện và mô tả. Trang Drive mở được nhưng công cụ không liệt kê nội dung.

Khác biệt quan trọng: định nghĩa `red_light` cho phép suy ra đèn đỏ từ hành vi giao thông khi không nhìn thấy đèn. Không nhập nguyên nhãn này vào ground truth của nhóm. Đề xuất gán lại theo bằng chứng, giữ liên kết video nguồn và dùng như tập VLM bổ sung. Chưa xác minh được giấy phép dữ liệu trong README đã đọc.

## 4. IMPTC — ứng viên bổ sung thời gian và trạng thái đèn

[Repository](https://github.com/kav-institute/imptc-dataset) mô tả giao lộ cố định tại Đức, chuỗi đồng bộ có quỹ đạo và trạng thái đèn. [Zenodo](https://zenodo.org/records/14811016) có `imptc_samples.tar.gz` 347,7 MB; nên khảo sát gói mẫu trước.

Đề xuất: kiểm tra có ảnh/video gốc riêng từng camera đủ nhìn đèn/vạch hay chỉ video tổng hợp có overlay. Nhãn cảm biến chỉ dùng đối chiếu ground truth, không cấp làm input trong bài đánh giá video-only. Không mặc định có đủ sự kiện vượt đỏ. README nêu nghiên cứu phi thương mại và giấy phép repository Apache-2.0; cần đối chiếu điều khoản của gói dữ liệu.

## 5. Đầu mối Việt Nam bổ sung, chưa phải nguồn video tải ngay

[A Large-Scale Dataset for Real-Time Vehicle Detection in Vietnamese Urban Traffic Scenes](https://www.sciencedirect.com/org/science/article/pii/S1546221826004108) mô tả ảnh CCTV giao lộ Đà Nẵng và nhận dạng đèn. Bài viết nêu dữ liệu nghiên cứu phi thương mại cung cấp theo yêu cầu tới tác giả. Đây là bộ ảnh; cần hỏi có thể cung cấp video gốc và thời gian tương ứng hay không. Chưa gửi liên hệ.

## 6. Nguồn không ưu tiên cho ground truth cốt lõi

- [UIT-ADrone](https://uit-together.github.io/datasets/UIT-ADrone/): drone tại vòng xoay; không khớp camera cố định và chứng minh thời điểm vượt vạch lúc đèn đỏ.
- [Bosch Small Traffic Lights](https://zenodo.org/records/12706046): phù hợp tác vụ đèn riêng; không thay thế dataset sự kiện liên tục của nhóm.
- Video demo đã vẽ bbox, thêm đèn giả lập hoặc chỉ tuyển vi phạm: chỉ dùng tham khảo, không tự xem là video thô/test đại diện.

## 7. Thứ tự tuyển mẫu đề xuất

1. Kiểm tra khả năng truy cập video gốc RunningRedlight. Nếu thiếu video gốc/quyền truy cập, giữ làm đầu mối và chuyển sang Zenodo.
2. Khảo sát hai file Zenodo nhỏ; chọn đoạn liên tục khi xác minh đủ xe–vạch–đèn.
3. Khảo sát FETV cho thử nghiệm VLM và IMPTC cho đối chiếu thời gian; không trộn các mục đích đánh giá thành một chỉ số.
4. Chọn 2–3 video/phiên độc lập, gán mọi lượt trong đoạn; mục tiêu pilot 30–50 lượt. Không bảo đảm đủ ba trạng thái chỉ bằng số lượt: ghi phân bố thực tế, bổ sung nguồn nếu thiếu.

Phiếu tuyển từng video cần có: nguồn/điều khoản sử dụng, checksum, camera và phiên gốc, độ phân giải/PTS, độ liên tục, vạch quan sát được, đèn đúng hướng quan sát được, ngữ cảnh trước/sau, điều kiện, overlay, trạng thái tuyển và lý do. Chỉ xác nhận đủ điều kiện sau khi xem trực tiếp. Ngày/đêm/mưa là điều kiện khảo sát, không phải thuộc tính đã được xác nhận cho toàn bộ shortlist.
