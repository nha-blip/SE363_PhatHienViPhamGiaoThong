# Hướng dẫn gán nhãn pilot

Áp dụng cùng `Dataset specification - Traffic Violation Understanding.md`. Mục tiêu pilot là kiểm tra quy ước, chi phí gán nhãn và nguồn dữ liệu; chưa phải benchmark cuối.

## 1. Chuẩn bị nguồn

- Lưu video gốc, URL, quốc gia, điều kiện sử dụng, checksum và quan hệ với các clip cắt.
- Chọn khoảng liên tục, không chỉ cắt các vi phạm nổi bật. Ghi lý do tuyển chọn.
- Loại nguồn bị cắt dựng hoặc thay đổi góc mà không thể giữ cấu hình ổn định; nếu chia cảnh, lưu quan hệ với nguồn gốc.
- Camera cố định; phải xác định được vùng tiếp cận, vạch dừng và đèn điều khiển hướng mục tiêu. Video chất lượng thấp có thể tạo nhãn thiếu bằng chứng, nhưng nguồn không thể cấu hình hướng/đèn không phù hợp test chính.
- Ghi điều kiện ánh sáng/thời tiết/chất lượng theo đoạn; không suy đoán điều không thấy.

## 2. Cấu hình camera

Trên frame tham chiếu, đánh dấu vùng tiếp cận, vạch dừng, hướng đi, vùng đèn và ánh xạ đèn với hướng đi. Giữ phiên bản cấu hình. Định nghĩa ranh giới vượt là mép vạch phía xe tiếp cận gặp trước, để đo lúc phần đầu xe bắt đầu vượt qua vạch; kiểm tra tính khả thi trong pilot. Nếu vạch mờ hoặc không thấy phần đầu xe, ghi bất định, không âm thầm thay bằng tâm bbox.

Đánh dấu thêm `observation_roi` bao gồm phần sau vạch cần quan sát để phân biệt đi thẳng với rẽ. Thời điểm vượt vạch không tự động là thời điểm kết thúc quan sát lượt.

Không chỉnh cấu hình dựa trên kết luận mô hình hoặc nhãn hành vi test. Cấu hình thủ công camera là input được phép biết trước.

## 3. Gán timeline đèn

Gán trạng thái từng khoảng thời gian cho đèn liên quan: đỏ, vàng, xanh, không rõ, tắt hoặc khác. Dùng timestamp video/PTS. Khi chuyển màu giữa hai frame, lưu khoảng từ quan sát cuối của trạng thái cũ tới quan sát đầu của trạng thái mới. Đoạn đèn bị che không được tự điền màu bằng nội suy chưa có bằng chứng.

Gán `signal_interval_id` cho mỗi khoảng trạng thái và `transition_id` cho mỗi lần chuyển màu, liên kết chu kỳ liên quan. Các khoảng trạng thái chắc chắn không chồng lấn; khoảng chuyển chưa phân định được giữ riêng. Xét cả lúc bắt đầu lẫn kết thúc đỏ khi đối chiếu xe vượt, không so với một lần chuyển đỏ bất kỳ trong video.

## 4. Gán mọi lượt tiếp cận

1. Gán ID phương tiện chuẩn trong video và loại phương tiện.
2. Ghi thời điểm bắt đầu/kết thúc quan sát lượt, lý do kết thúc, hướng đi, phạm vi và tình trạng cắt cụt. Tiếp tục xem sau vượt vạch để xác định hướng đi.
3. Gán bbox tại các frame mốc: tiếp cận, trước/sau vượt vạch nếu có, xác định hướng đi và bằng chứng. Lưu trong `vehicle_keyframes` với ID xe, frame, timestamp và che khuất. Bbox `[x_min, y_min, x_max, y_max]` theo pixel ảnh gốc, bao phần nhìn thấy; không tạo bbox giả ở frame xe biến mất.
4. Xác định lần cuối phần đầu xe chưa vượt và lần đầu đã vượt; lưu khoảng thời gian giữa hai quan sát. Nếu thiếu bằng chứng về lần vượt, để thời gian trống và ghi lý do.
5. Đối chiếu timeline đúng đèn; gán kết luận hành vi.
6. Ghi dữ kiện chuẩn và frame hỗ trợ, không xem dự đoán mô hình trước khi gán nhãn ban đầu.

ID chuẩn không đổi theo ID dự đoán. Xe dừng rồi đi tiếp là cùng lượt; che khuất tạm thời không kết thúc lượt nếu có thể nối chắc chắn. Nếu chưa xác định xe sau che khuất có phải xe trước đó, lưu mảnh quan sát với `identity_uncertain`, không nối theo suy đoán hoặc đếm thành hai lượt hoàn chỉnh. Kết thúc lượt khi rời `observation_roi`, hết đoạn hoặc mất dấu không thể nối; ghi `end_reason`. Xe rời vùng rồi quay lại là lượt mới. Nếu xe lùi rồi vượt lại trong cùng lượt, lưu từng lần trong `crossing_observations` với `crossing_id` riêng.

## 5. Quy tắc kết luận

| Tình huống | Nhãn |
| --- | --- |
| Xe vượt vạch trong khoảng chứng minh đèn liên quan đã đỏ | `red_light_crossing` |
| Toàn bộ khoảng vượt nằm trong trạng thái xác minh không đỏ của chu kỳ liên quan, hoặc xe được quan sát đủ và không vượt trong đoạn đánh giá | `no_red_light_crossing` |
| Khoảng vượt vạch chồng lấn khoảng chuyển đỏ, không phân định thứ tự | `insufficient_evidence` |
| Không rõ đèn liên quan, vạch, phần đầu xe hoặc lần vượt bị che | `insufficient_evidence` |
| Xe rẽ hoặc loại xe ngoài phạm vi | `scope_status = out_of_scope`, `assessment = null`, ghi lý do |
| Chưa xác định được hướng hoặc loại xe | `scope_status = scope_uncertain`, `assessment = null`, ghi lý do |
| Lượt bị cắt ở đầu/cuối video | Ghi `boundary_truncated`, chỉ kết luận nếu phần còn lại đủ bằng chứng |

Ranh giới thời gian tiếp xúc nhau mà thứ tự chưa được chứng minh cũng là thiếu bằng chứng. Xe đã ở trong giao lộ khi đèn đỏ không đủ để gán vượt vạch lúc đỏ. Nhãn âm tính đối với xe dừng chỉ mô tả lượt/đoạn đã quan sát, không khẳng định xe không vi phạm về sau.

Nếu có nhiều lần vượt trong cùng lượt, nhãn lượt dương tính khi ít nhất một lần xác định đỏ; nếu không có dương tính nhưng có lần chưa rõ, nhãn lượt là thiếu bằng chứng. Giữ nhãn từng lần để đánh giá sự kiện một-một.

Ví dụ giả định cùng một đèn có xanh được xác minh trong phần bên trong khoảng 8,00–9,96 giây, chuyển đỏ trong [9,96; 10,00], đỏ được xác minh trong phần bên trong khoảng 10,00–20,00 giây. Ranh giới chuyển màu vẫn xét theo quy tắc bất định ở trên. Các khoảng dưới đây không có mất quan sát:

| Khoảng vượt (giây) | Nhãn | Căn cứ |
| --- | --- | --- |
| [12,00; 12,04] | `red_light_crossing` | Toàn bộ khoảng nằm trong đỏ đã xác minh |
| [9,80; 9,84] | `no_red_light_crossing` | Toàn bộ khoảng nằm trong xanh đã xác minh |
| [9,94; 10,02] | `insufficient_evidence` | Chồng lấn khoảng chuyển màu |
| [10,00; 10,04] | `insufficient_evidence` nếu không có thêm bằng chứng phân định thứ tự | Tiếp xúc ranh giới chuyển màu |

Đây là ví dụ kỹ thuật về nhãn quan sát, không là kết luận pháp lý.

## 6. Dấu hiệu ngoại lệ

Gán hai nhóm: người điều tiết (`traffic_controller`) và tín hiệu/biển phụ (`supplementary_signal_or_sign`). Mỗi nhóm có trạng thái thấy/không thấy/chưa rõ, quan hệ với hướng mục tiêu, mô tả điều nhìn được và frame hỗ trợ.

Không suy ra người xuất hiện đang ra hiệu cho xe mục tiêu; không suy ra nội dung biển quá nhỏ để đọc. Không thấy dấu hiệu trong khung hình không chứng minh ngoài khung hình không có. Kết luận hành vi vượt vạch lúc đỏ được lưu riêng và không bị đổi thành âm tính từ dấu hiệu này. Output phải diễn đạt rõ hành vi quan sát được, không tuyên bố ngoại lệ pháp lý đã áp dụng.

## 7. Dữ kiện chuẩn cho giải thích

Ghi các dữ kiện được chứng minh: phương tiện mục tiêu, hướng đi, khoảng vượt vạch, trạng thái đúng đèn, thứ tự thời gian, dấu hiệu ngoại lệ và hạn chế quan sát. Mỗi dữ kiện có frame/timeline hỗ trợ. Không bắt buộc một câu văn tham chiếu duy nhất.

Rubric pilot kiểm tra từng mục: đúng đối tượng, đúng quan hệ thời gian, đúng đèn, dẫn bằng chứng, không thêm thông tin thiếu căn cứ, diễn đạt đúng giới hạn. Ghi đạt/không đạt/không áp dụng và lý do; đây là checklist khảo sát, thang điểm báo cáo cuối chốt sau pilot.

### Kiểm tra gói bằng chứng VLM

Sau khi có pipeline, lưu đúng clip, frame/crop, thứ tự, timestamp và dữ kiện CV gửi cho VLM trong `evidence_packages`. Người kiểm tra xem đúng gói, không lấy kết luận VLM làm căn cứ; ghi `evidence_sufficiency`, `package_assessment`, phần thiếu và dữ kiện CV sai trong `package_reviews`. Đối chiếu nhãn toàn video ở bước riêng để phân loại nguyên nhân lỗi. Không ghi đè nhãn toàn video bằng nhãn gói.

Ví dụ: video gốc xác nhận xe vượt lúc đỏ nhưng clip ứng viên chỉ bắt đầu khi xe đã qua vạch. Ground truth toàn video vẫn dương tính; nhãn gói có thể là thiếu bằng chứng. VLM trả thiếu bằng chứng là phù hợp với gói, nhưng nếu pipeline không có cảnh báo dương tính đúng thì sự kiện vẫn tính FN. Không cung cấp nhãn gói hoặc nhãn toàn video cho mô hình khi đánh giá.

## 8. Kiểm tra chéo và hoàn tất

- Người thứ hai kiểm tra toàn bộ test và các mẫu khó. Lưu nhãn ban đầu, nhãn sau duyệt và lý do sửa.
- Mẫu chưa giải quyết bất đồng chưa được coi là nhãn chuẩn đã duyệt; chuyển thành thiếu bằng chứng nếu nguyên nhân thực sự là thiếu quan sát.
- Kiểm tra frame tham chiếu tồn tại, khoảng thời gian hợp lệ, ID xe nhất quán, cấu hình đúng phiên bản và timeline đèn không mâu thuẫn.
- Kiểm tra `assessment = null` cho mẫu ngoài/chưa rõ phạm vi; khoảng vượt liên kết đúng đèn và chu kỳ; bbox có hệ tọa độ nhất quán. Các mảnh chưa giải quyết ID được lưu và thống kê riêng.
- Ghi giờ công tuyển nguồn, cấu hình, gán và kiểm tra riêng.
- Tóm tắt số nguồn độc lập, số lượt, số nhãn từng nhóm, điều kiện, tỷ lệ bất định và bất đồng. Pilot dự kiến 2–3 nguồn và khoảng 30–50 lượt, điều chỉnh theo video thực tế.
- Video dùng chỉnh prompt/ngưỡng thuộc tập phát triển. Tất cả clip cùng video/phiên và nội dung tải lại phải cùng nhóm chia tập.

### Kiểm tra tính điểm sau khi có dự đoán

- Ghép dương tính một-một theo xe, lần vượt và thời gian. Ghép đúng là TP; lần vượt dương tính không có dự đoán ghép đúng là FN, kể cả VLM từ chối hoặc không tạo ứng viên.
- Cảnh báo dương tính thừa trong phần đã gán đủ và chấm được là FP, gồm cảnh báo trùng. Chỉ khử bản ghi phát lại kỹ thuật có cùng ID, không khử các sự kiện khác ID cùng cảnh báo một lần vượt.
- Nhãn thiếu bằng chứng, ngoài/chưa rõ phạm vi và ID chưa giải quyết không bị ép thành âm tính; loại khỏi mẫu số precision/recall chính và báo số lượng/cảnh báo riêng. Kiểm tra cảnh báo chưa ghép để phân biệt FP với không thể chấm.
- Ghi quy tắc, ngưỡng và phiên đánh giá. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); mẫu số của chỉ số nào bằng 0 thì chỉ số đó ghi N/A, kèm số đếm. Báo thêm kết quả thẩm định theo nhãn gói, không dùng nó thay kết quả toàn pipeline.
- Ví dụ có 2 lần vượt dương tính: lần đầu có 2 cảnh báo dương tính đúng thời gian, lần sau VLM từ chối. Kết quả là TP=1, FP=1 do trùng, FN=1; precision=0,5, recall=0,5, F1=0,5. Lưu nguyên nhân FN để phân biệt thiếu bằng chứng đầu vào với lỗi suy luận.

## 9. Những gì chốt sau pilot

Quy mô chính thức, phân bổ điều kiện, tỷ lệ phát triển/test, độ dài clip trước/sau ứng viên, ngưỡng ghép một-một, sai số thời gian, thang điểm giải thích và chất lượng tối thiểu của nguồn. Ngưỡng được chốt bằng tập phát triển trước khi đánh giá test.

Chưa có video được thu thập hoặc nhãn thực tế trong workspace tại thời điểm viết hướng dẫn này.
