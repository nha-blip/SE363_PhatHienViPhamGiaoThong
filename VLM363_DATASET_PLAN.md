# PROJECT 363 — Phương án dataset VLM

Ngày lập: 09/10/2026. Trạng thái: PROPOSED — chưa tạo dataset, chưa chạy VLM.

## 1. Căn cứ và kết luận

Đã đọc `Nghiên Cứu VLM Giao Thông(2).md` và đối chiếu ba tài liệu hiện có:

- `Dataset specification - Traffic Violation Understanding.md`.
- `Dataset sources shortlist - Traffic Violation Understanding.md`.
- `Pilot annotation guide - Traffic Violation Understanding.md`.

Tài liệu nghiên cứu mô tả VLM là lớp thẩm định ngữ nghĩa theo sự kiện: nhận video/keyframe và dữ kiện CV, xem xét bối cảnh, thứ tự thời gian, dấu hiệu ngoại lệ và giải thích có bằng chứng. VLM không thay thế YOLO/ByteTrack và không cần một dataset bbox mọi frame mới để bắt đầu sử dụng pretrained.

**Phạm vi được chủ dự án duyệt ngày 12/10/2026: `car`, `motorcycle`, `bus`, `truck` đi thẳng, quan sát hành vi vượt vạch lúc đèn đỏ.** Bán tải trong batch_001 được xác nhận thuộc `truck`, nằm trong phạm vi. Không mở đồng thời sang sai làn, mũ bảo hiểm, tai nạn hay phân định ngoại lệ pháp lý. Các nhiệm vụ đó cần quy ước và dữ liệu riêng sau pilot.

Tạm dừng mở rộng dataset tracking; giữ nguyên video, GT, bản nháp và quyết định QA đã có. Không xóa hoặc sửa nguồn để chuyển sang VLM.

## 2. Dataset cần phục vụ việc gì?

| Thành phần | Đơn vị dữ liệu | Mục đích |
| --- | --- | --- |
| Ground truth toàn pipeline | Một lượt tiếp cận vạch của một xe trong đoạn video liên tục | Đánh giá tạo ứng viên, bỏ sót, cảnh báo sai và cảnh báo trùng |
| Gói bằng chứng VLM | Clip/keyframe có timestamp, mục tiêu, cấu hình camera và dữ kiện CV dự đoán | Đánh giá VLM có kết luận đúng với thông tin thực nhận không |
| Bộ dữ kiện kiểm chứng | Đèn liên quan, khoảng vượt vạch, frame chứng minh, hạn chế quan sát | Kiểm tra giải thích không thêm thông tin thiếu căn cứ |
| Kho Legal-RAG riêng | Văn bản chính thức, phiên bản, điều/khoản/điểm và thời gian hiệu lực | Tra cứu quy định; không dùng trí nhớ VLM làm căn cứ luật |

Giai đoạn đầu dùng dataset để đánh giá pretrained và điều chỉnh prompt trên tập phát triển, **không mặc định fine-tune**. Chỉ quyết định fine-tune khi đã có lỗi lặp lại, nhãn đủ chất lượng và baseline đo được.

## 3. Nguồn dữ liệu: dùng lại nhưng phải tuyển đúng mục tiêu

| Nguồn trong dự án | Vai trò đề xuất | Điều kiện / giới hạn |
| --- | --- | --- |
| `data/Dataset_Traffic_Evaluation/` | Ưu tiên khảo sát video gốc cho sự kiện đèn đỏ | Xác minh quốc gia, quyền sử dụng, giải mã, vạch và đúng đèn điều khiển; chưa mặc định đủ điều kiện |
| `data/Vehicle_Tracking/VNTraffic/` | Thử đóng gói bằng chứng, định vị xe, quan sát che khuất | Có bbox/ID không đồng nghĩa có nhãn vượt đỏ; phải xem video gốc và kiểm chứng đèn–hướng–vạch |
| `data/Vehicle_Tracking/AICC22-Custom/` | Mẫu bổ sung quan sát/che khuất/giới hạn CV | Không nhập ID dự đoán làm GT sự kiện; chưa xác nhận đủ bằng chứng vượt đỏ |
| Video Bellevue | Bổ sung các trường hợp xe che nhau, đèn/cột che xe, khó xác định đối tượng | Bối cảnh nước ngoài phải báo riêng; không áp luật Việt Nam lên video này |
| Vehicle363 và TrafficLight363 bản YOLO | Tái sử dụng detector để tạo dữ kiện đầu vào | Ảnh đơn, nhãn xe/đèn không chứng minh thứ tự thời gian hoặc đèn nào điều khiển xe nào |

Shortlist có RunningRedlight, FETV, IMPTC và nguồn Zenodo. Đây là đầu mối cần xác minh truy cập, video gốc và điều khoản dữ liệu, không phải nguồn đã nghiệm thu. Không tải hàng loạt trước khi kiểm tra mẫu.

Đọc video gốc thay vì video QA đã vẽ ID/bbox làm đầu vào mặc định. Overlay có thể tiết lộ nhãn hoặc làm nhiễu mô hình. Crop phụ trợ phải giữ liên kết ảnh gốc; luôn giữ toàn cảnh để thấy xe–vạch–đèn.

## 4. Pilot nhỏ trước, mở rộng sau

### Bước A — Tuyển nguồn

- Lập danh mục video: đường dẫn, SHA-256, nguồn, quốc gia, giấy phép, camera/phiên, thời lượng, độ phân giải, FPS và PTS/timebase.
- Kiểm tra khả năng giải mã các đoạn dự kiến dùng; loại hoặc ghi riêng nguồn hỏng/cắt dựng/thiếu thông tin.
- Xem trực tiếp vạch dừng, phần đầu xe, hướng đi và đèn điều khiển hướng đó. Không nhìn rõ quan hệ đèn–hướng thì không tuyển làm test chính.
- Đánh dấu `ELIGIBLE`, `SUPPLEMENTARY_ONLY`, `REVIEW_REQUIRED`, `REJECTED`, kèm lý do.

### Bước B — Gán pilot

Theo đặc tả hiện có: **2–3 video/phiên độc lập, khoảng 30–50 lượt tiếp cận** trong các đoạn liên tục. Đây là mục tiêu khảo sát chi phí, không bảo đảm có đủ mẫu dương tính.

Gán tất cả lượt thuộc phạm vi trong đoạn đã chọn, bao gồm xe dừng, xe đi bình thường, xe bị che và lượt thiếu bằng chứng. Không chỉ chọn clip mà YOLO/quy tắc đã báo nghi vấn: cách đó không đo được bỏ sót của pipeline.

Muốn quan sát thêm ca dương tính hiếm có thể bổ sung bộ thử thách được tuyển riêng; ghi rõ thiên lệch tuyển chọn, không trộn để báo tỷ lệ vi phạm tự nhiên.

### Bước C — Đóng gói cho VLM

Thử trên tập phát triển clip khoảng 4–12 giây và 6–12 keyframe có thứ tự; đây là cấu hình thử, chưa phải chuẩn cố định. Mở rộng clip nếu chưa đủ ngữ cảnh trước/sau hoặc chưa xác định được hướng đi.

Giữ PTS/timestamp gốc. Tăng mật độ frame gần chuyển đèn và vượt vạch; ảnh lấy thưa không thể bảo đảm độ chính xác mili-giây. Nếu chờ frame sau sự kiện trong streaming, tính thời gian chờ vào latency.

Mục tiêu được tham chiếu bằng bbox tại frame mốc và định danh ổn định trong gói. ID ByteTrack là dữ kiện dự đoán có thể sai; không tự dùng làm ID chuẩn.

### Bước D — Đánh giá rồi mới mở rộng

So sánh baseline CV/quy tắc với CV/quy tắc + VLM trên cùng ứng viên. Đo cả kết quả toàn pipeline và chất lượng thẩm định riêng. Dùng giờ công pilot, phân bố nhãn và khả năng tuyển video để quyết định quy mô tiếp theo; chưa chốt số nghìn mẫu hoặc lịch hoàn thành khi chưa tuyển nguồn.

## 5. Nhãn và quy tắc quan sát

Giữ nguyên taxonomy kỹ thuật đã có:

- `red_light_crossing`: chứng minh phần đầu xe vượt vạch trong khoảng đúng đèn đã được xác minh đỏ.
- `no_red_light_crossing`: đủ bằng chứng không xảy ra hành vi mục tiêu trong khoảng đánh giá.
- `insufficient_evidence`: không rõ xe, vạch, đèn liên quan hoặc thứ tự thời gian.

Nhãn này **không phải phán quyết vi phạm pháp luật**. Xe đã nằm trong giao lộ khi đèn đỏ không đủ chứng minh vượt vạch lúc đỏ.

Lưu khoảng thời gian bất định cho lần vượt và lần chuyển màu. Khoảng chồng lấn hoặc tiếp xúc mà không xác định được thứ tự nhận `insufficient_evidence`; không đoán một timestamp chính xác. Xét cả bắt đầu và kết thúc khoảng đỏ của đúng chu kỳ.

Xe rẽ/ngoài nhóm mục tiêu: `out_of_scope`, nhãn hành vi `null`. Chưa xác định hướng/loại xe: `scope_uncertain`, không ép âm tính. Lượt bị cắt đầu/cuối ghi `boundary_truncated`.

Chỉ gán bbox xe tại frame mốc cần chứng minh, không vẽ lại toàn video. Timeline đèn dùng chung cho các lượt cùng đèn. Không yêu cầu GT quỹ đạo dày đặc để làm pilot VLM.

Dấu hiệu người điều tiết và tín hiệu/biển phụ lưu riêng với trạng thái thấy/không thấy/chưa rõ và frame hỗ trợ. Không tự suy ra hiệu lệnh, tính hợp pháp hoặc đổi nhãn hành vi từ dương thành âm. Các ngoại lệ khác trong tài liệu nghiên cứu chưa thuộc taxonomy pilot.

## 6. Schema và phân tách dữ liệu

Tái sử dụng schema của đặc tả, không tạo bộ định nghĩa cạnh tranh:

| Nhóm | Nội dung chính |
| --- | --- |
| `sources` | Nguồn, quốc gia, quyền dùng, checksum, camera/phiên, timebase, split |
| `camera_configs` | ROI tiếp cận/quan sát, vạch, hướng, đèn, ánh xạ đèn–hướng, phiên bản |
| `observation_segments` | Khoảng liên tục, điều kiện và mức hoàn tất nhãn |
| `signal_timelines`, `signal_transitions` | Khoảng màu chắc chắn và khoảng chuyển bất định |
| `approaches` | ID xe chuẩn, phạm vi, lần vượt, assessment, che khuất, dữ kiện/bằng chứng, người gán/duyệt |
| `vehicle_keyframes` | Bbox pixel ảnh gốc `[x_min,y_min,x_max,y_max]` tại frame mốc |
| `evidence_packages` | Chính xác clip/crop/frame/thứ tự/timestamp/dữ kiện dự đoán VLM nhận |
| `package_reviews` | Nhãn gói thực nhận, phần thiếu, dữ kiện CV sai; lưu riêng nhãn toàn video |

Ví dụ: video gốc chứng minh vượt lúc đỏ nhưng gói bắt đầu sau khi xe đã qua vạch. Nhãn toàn video vẫn dương, nhãn gói có thể thiếu bằng chứng. VLM từ chối có thể đúng với gói nhưng pipeline vẫn bỏ sót nếu không có cảnh báo đúng.

Ground truth xe/đèn/thời gian/kết luận không được truyền vào input đánh giá chính. Cấu hình camera được phép biết trước; dữ kiện CV phải là dự đoán. Thí nghiệm input chuẩn chỉ dùng chẩn đoán riêng và ghi rõ ORACLE, không báo như hiệu năng thực tế.

Output VLM gồm assessment, dữ kiện quan sát, frame/timestamp hỗ trợ, giải thích ngắn và giới hạn. Không yêu cầu bản ghi chuỗi suy nghĩ nội bộ dài; cần bằng chứng kiểm chứng được.

## 7. Chia tập và QA

- Chia theo video/phiên gốc trước khi chỉnh prompt. Tệp tách từ cùng phiên, clip chồng lấn hoặc bản tải lại cùng nội dung thuộc cùng tập.
- Theo đặc tả hiện có, cùng camera có thể xuất hiện trong development/test nếu phiên độc lập; báo đây là kiểm tra camera quen, không khẳng định tổng quát hóa camera mới. Holdout camera riêng là mở rộng khi đủ nguồn.
- Không ấn định tỷ lệ train/val/test vì pilot dùng pretrained và số nguồn còn ít. Prompt/quy tắc/ngưỡng chỉnh trên development; test giữ độc lập.
- Golden samples dùng làm ví dụ truy xuất/prompt thuộc development, không lấy đáp án test làm ví dụ.
- Một người gán độc lập trước khi xem kết luận mô hình; người thứ hai kiểm tra toàn bộ test và mẫu khó. Lưu nhãn ban đầu, sửa đổi và bất đồng.
- Kiểm tra timestamp/frame tồn tại, bbox hợp lệ, đúng đèn/chu kỳ, ID xe nhất quán, timeline không mâu thuẫn và bằng chứng đủ hỗ trợ nhãn.
- Kiểm tra quyền sử dụng và dữ liệu nhạy cảm; nếu làm bản che mặt/biển số, lưu provenance và không che mất tín hiệu cần đánh giá.

## 8. Đánh giá

**Toàn pipeline:** precision/recall/F1 theo sự kiện ghép một-một đúng xe và thời gian; báo cảnh báo trùng, ứng viên bị bỏ sót và nguyên nhân FN. Không tạo ứng viên hoặc VLM từ chối trên sự kiện dương tính vẫn có thể là FN.

**VLM riêng:** ma trận ba nhãn theo `package_assessment`, tỷ lệ từ chối/bao phủ, đúng đối tượng/đèn/thứ tự thời gian, bằng chứng và giới hạn giải thích. Tách lỗi CV, cắt clip và lỗi thẩm định.

**Nhóm chưa chấm được:** thiếu bằng chứng, ngoài/chưa rõ phạm vi, chưa giải quyết ID phải báo riêng; không biến thành âm tính để làm đẹp precision/recall. Mẫu số bằng 0 báo N/A kèm số đếm.

**Vận hành:** JSON đúng schema, latency p50/p95 gồm chờ ngữ cảnh sau, bộ nhớ GPU và hàng đợi trên phần cứng thực tế. Chốt ngưỡng ghép, rubric và mục tiêu chất lượng sau pilot, trước test; không lấy số quảng bá trong tài liệu làm kết quả dự án.

## 9. Legal-RAG và phần cứng

Legal-RAG là nhánh dữ liệu riêng. Lưu URL nguồn chính thức, hash tài liệu, ngày hiệu lực/hết hiệu lực, phiên bản sửa đổi, quốc gia và điều/khoản/điểm; kiểm chứng văn bản áp dụng tại thời điểm sự kiện hoặc ghi rõ chế độ demo theo ngày chọn. Không đóng cứng mức phạt trong nhãn hình ảnh; chưa có người kiểm tra pháp lý thì chỉ xuất thông tin tham khảo.

Tài liệu nghiên cứu giả định RTX 3060 12 GB, còn máy được kiểm tra trong dự án là RTX 4050 Laptop khoảng 6 GB. Không mặc định Qwen2-VL-7B + YOLO chạy đồng thời được. Giữ dataset trung lập model, thử bộ gói nhỏ đo VRAM/latency trước khi chọn checkpoint, lượng hóa và nơi chạy. Không tự nâng cấp model chỉ vì repository đã đổi phiên bản.

Một số phát biểu trong tài liệu như “khắc phục hoàn toàn ảo giác” và “tính hợp pháp tuyệt đối” không phải tiêu chí nghiệm thu được chứng minh. Lượt này không kiểm chứng toàn bộ bài báo/số liệu tham khảo; kế hoạch không phụ thuộc các tuyên bố tuyệt đối đó.

## 10. Cấu trúc dự kiến và sản phẩm bàn giao

Đây là đề xuất đường dẫn, **chưa tạo payload hoặc bộ nhãn**:

```text
data/VLM363_working/
    metadata/source_manifest.jsonl
    metadata/camera_configs.jsonl
    metadata/observation_segments.jsonl
    clips/
    keyframes/
    annotations/signal_timelines.jsonl
    annotations/signal_transitions.jsonl
    annotations/approaches.jsonl
    annotations/vehicle_keyframes.jsonl
    evidence_packages/
    qa/package_reviews.jsonl
    qa/review_history.jsonl
    splits/
runs/project363/vlm/
    source_suitability_report.md
    pilot_annotation_report.md
    baseline_evaluation/
data/LegalRAG363_working/
    documents/
    chunks/
    metadata/
```

Thứ tự triển khai: (1) danh mục và tuyển video, (2) cấu hình camera + gán 30–50 lượt, (3) QA, (4) gói bằng chứng và baseline pretrained, (5) phân tích lỗi + chốt quy mô. Kho pháp lý làm riêng; không để nó chặn pilot quan sát hành vi.

Pilot sẵn sàng khi nguồn/quyền dùng được ghi rõ, frame/timebase hợp lệ, cấu hình đèn–hướng được xác minh, mọi lượt trong đoạn tuyển đã gán, nhãn test/mẫu khó được duyệt và split không rò rỉ. Nếu chưa có mẫu vượt đỏ đủ bằng chứng thì báo thiếu nguồn, không tuyên bố dataset đủ cho đánh giá hành vi mục tiêu.

## 11. Tài liệu đối chiếu bên ngoài

- [TrafficVLM — bài báo tác giả](https://arxiv.org/abs/2404.09275): mô tả video giao thông theo các pha sự kiện và ngữ cảnh không gian/thời gian; đây không phải bộ nhãn vi phạm pháp lý Việt Nam có sẵn.
- [Repository Qwen hiện tại](https://github.com/QwenLM/Qwen3-VL): đường dẫn Qwen2-VL đã chuyển hướng khi kiểm tra; cần khóa đúng model/revision và preprocessing khi triển khai, không dựa vào tên đường dẫn cũ.

**Bước tiếp theo đề xuất:** khảo sát video gốc trong `data/Dataset_Traffic_Evaluation/` và VNTraffic, tạo báo cáo tuyển nguồn có ảnh minh chứng xe–vạch–đèn; chưa chạy model hoặc gán nhãn hàng loạt trước bước này.
