# Đặc tả chi tiết hệ thống 363 — Phân tích giao thông bằng CV, VLM và dữ liệu phân tán

**Phiên bản:** 1.1 — 25/11/2026 (1.0 — 30/10/2026). Thay đổi ở bản 1.1: taxonomy xe 6 lớp (§3.1, FR-06, §15.3), cách đọc trạng thái đèn (§9.4), ngưỡng nghiệm thu detector (§19.2, §21).  
**Trạng thái:** Bản thiết kế tổng hợp để rà soát và triển khai; chưa phải biên bản nghiệm thu hoặc phạm vi đã được giảng viên phê duyệt.  
**Dự án:** SE363 — Traffic Violation Understanding.

## 1. Mục đích và căn cứ

Tài liệu mô tả kiến trúc, yêu cầu chức năng, quy tắc xử lý, hợp đồng dữ liệu, lưu trữ, giao diện và cách đánh giá hệ thống 363. Mục tiêu là chuyển các tài liệu nghiên cứu thành các hạng mục có thể triển khai và kiểm thử.

Tài liệu sử dụng các nguồn nội bộ sau:

| Mã | Tài liệu | Vai trò |
| --- | --- | --- |
| S1 | [Nghiên Cứu VLM Giao Thông(2).md](<Nghiên Cứu VLM Giao Thông(2).md>) | Kiến trúc ban đầu: Kafka, Spark, YOLO, ByteTrack, VLM, Legal RAG |
| S2 | [deep-research-report (2).md](<deep-research-report (2).md>) | Các đề xuất hoàn thiện kiến trúc, dịch vụ, dữ liệu và vận hành |
| S3 | [Dataset specification - Traffic Violation Understanding.md](<Dataset specification - Traffic Violation Understanding.md>) | Phạm vi pilot, định nghĩa hành vi, ground truth và đánh giá |
| S4 | [CONTEXT_DU_AN_20261030.md](CONTEXT_DU_AN_20261030.md) | Ghi nhận tình hình thực nghiệm, dataset và các quyết định trong quá trình làm dự án |

Khi tài liệu có khác biệt, bản này áp dụng phạm vi pilot và định nghĩa nhãn của S3; sử dụng S1 để giữ định hướng hệ thống; sử dụng S2 cho đề xuất triển khai kỹ thuật. Các con số thực nghiệm trong S4 chỉ phản ánh những phiên chạy đã ghi nhận.

Các từ **phải**, **nên**, **có thể** lần lượt biểu thị yêu cầu bắt buộc trong phạm vi đã chọn, khuyến nghị và khả năng mở rộng. API, bảng dữ liệu, mã lỗi và cấu hình đề xuất trong bản này là hợp đồng thiết kế, không khẳng định đã tồn tại trong mã nguồn.

## 2. Mục tiêu hệ thống

Hệ thống tiếp nhận video giao thông, phát hiện và theo dõi phương tiện, xác định quan hệ giữa phương tiện–vạch dừng–đèn điều khiển, tạo hồ sơ nghi vượt đèn đỏ kèm bằng chứng. VLM hỗ trợ xem xét ngữ cảnh và giải thích; người đánh giá kiểm tra kết quả trước khi chốt hồ sơ.

Các mục tiêu cụ thể:

1. Xác định được phương tiện mục tiêu, hướng đi và khoảng thời gian vượt vạch.
2. Phân biệt xe vượt lúc đỏ với xe đã qua vạch trước khi đèn đỏ hoặc xe dừng trước vạch.
3. Lưu video, keyframe và dữ kiện đủ để người khác kiểm tra lại kết luận.
4. Cho phép trả về thiếu bằng chứng khi xe, đèn, vạch hoặc thứ tự thời gian không rõ.
5. Tách xử lý CV liên tục khỏi tác vụ VLM tốn tài nguyên.
6. Có thể phát lại dữ liệu, theo dõi lỗi và tái lập kết quả theo phiên bản mô hình/cấu hình.
7. Đánh giá riêng độ chính xác detector, tracking, sự kiện và tốc độ toàn hệ thống.

Đầu ra là kết luận hành vi quan sát được và hồ sơ hỗ trợ kiểm duyệt. Hệ thống không tự ban hành quyết định xử phạt.

## 3. Phạm vi và các giai đoạn

### 3.1. Phạm vi pilot bắt buộc

| Hạng mục | Phạm vi |
| --- | --- |
| Hành vi | Nghi vượt vạch dừng khi đèn điều khiển hướng đi đã đỏ |
| Phương tiện | `car`, `motorcycle`, `bus`, `truck`, `pickup`, `minibus` (định nghĩa ở §15.3) |
| Hướng di chuyển | Đi thẳng trong vùng đã cấu hình |
| Nguồn đánh giá | Video liên tục từ góc camera cố định, có trục thời gian và cấu hình cảnh |
| Độ phân giải mục tiêu | Khung hình 1280×720; kích thước đầu vào mô hình được cấu hình riêng |
| Điều kiện | Ưu tiên bối cảnh Việt Nam; báo cáo ngày/đêm/mưa và nguồn ngoài Việt Nam riêng |
| Hiệu chuẩn | Người dùng khai báo vùng tiếp cận, vùng quan sát, vạch dừng, đèn và quan hệ điều khiển |
| Kết luận | Ba trạng thái hành vi và các lý do bất định |
| Kiểm duyệt | Người đánh giá xem bằng chứng, ghi quyết định và lý do |

### 3.2. Phạm vi kiến trúc đầy đủ

Kiến trúc phải dành giao diện tích hợp cho RTSP, Kafka, Spark Structured Streaming, VLM, nhận dạng biển số và Legal RAG. Việc đưa một thành phần vào sơ đồ không đồng nghĩa thành phần đó đã được triển khai hoặc đạt nghiệm thu.

Kafka/Spark là hạng mục của hệ thống dữ liệu phân tán cần thể hiện trong bản tích hợp phục vụ môn học. Có thể phát triển và kiểm thử CV độc lập trước khi kết nối đầy đủ các thành phần.

### 3.3. Chưa thuộc nghiệm thu pilot

- Suy luận rẽ trái/rẽ phải, đèn mũi tên, đèn riêng cho xe máy khi chưa có taxonomy và dữ liệu tương ứng.
- Đi ngược chiều, sai làn, không đội mũ bảo hiểm, chở quá số người và dừng đỗ.
- Tự suy tốc độ theo km/h khi chưa hiệu chuẩn không gian và thời gian.
- Nhận dạng một xe xuyên nhiều camera.
- Tự kết luận ngoại lệ pháp lý chỉ từ sự xuất hiện của người điều tiết hoặc biển phụ.
- Cam kết vận hành mọi camera hoặc mọi điều kiện thời tiết với cùng độ chính xác.

Các mục này là phần mở rộng, cần đặc tả và bộ đánh giá riêng.

## 4. Thuật ngữ và nguyên tắc

| Thuật ngữ | Định nghĩa |
| --- | --- |
| Detection | Bbox, lớp và điểm tin cậy do detector dự đoán trên một frame |
| Track | Chuỗi quan sát được tracker gán cùng định danh trong một phiên camera |
| Approach | Một lượt phương tiện tiếp cận và đi qua hoặc dừng trong vùng quan sát |
| Candidate | Sự kiện nghi vấn do quy tắc tạo ra, chưa phải ground truth |
| Evidence package | Clip, frame, cấu hình cảnh và dữ kiện dự đoán thực sự được đưa cho VLM/người xem |
| Ground truth (GT) | Nhãn đã được người gán và kiểm tra theo quy ước dataset |
| Event time | Thời gian trong nguồn video, hoặc thời gian UTC ánh xạ từ nguồn đã xác minh |
| Processing time | Thời gian hệ thống xử lý bản ghi |
| `unknown` | Không đủ quan sát để xác định trạng thái |
| `off` | Quan sát được đèn không phát sáng; không đồng nghĩa mất dữ liệu |

Không dùng GT làm đầu vào mô hình trong đánh giá chính. Không dùng output mô hình làm GT nếu chưa qua kiểm chứng. Không thay đổi nhãn quan sát hành vi chỉ để phù hợp với lời giải thích của VLM.

## 5. Kiến trúc tổng thể

### 5.1. Đối chiếu hai phương án trong tài liệu

S1 mô tả chuỗi thu nạp qua Kafka, Spark điều phối xử lý rồi gọi Served AI. S2 đề xuất giữ luồng giải mã–CV gần nguồn, đưa sự kiện và tham chiếu bằng chứng qua Kafka, sử dụng Spark cho xử lý sự kiện theo thời gian và tổng hợp.

**Phương án triển khai đề xuất trong bản này là phương án S2**, nhằm giữ detector/tracker hoạt động độc lập với hàng đợi VLM và các micro-batch phân tích. Đây là lựa chọn thiết kế cần được rà soát theo yêu cầu môn học, không phải tuyên bố S1 đã được triển khai như vậy.

```text
Video file / RTSP
        |
        v
Ingest + kiểm tra thời gian/chất lượng + bộ đệm video
        |
        v
YOLO xe + YOLO trạng thái đèn
        |
        v
ByteTrack + timeline đèn + cấu hình cảnh
        |
        v
Quy tắc không gian/thời gian
        |
        +--> Sự kiện ứng viên --> Evidence Service --> MinIO
        |                              |
        |                              v
        |                        Hàng đợi VLM
        |                              |
        |                    Kết quả ngữ nghĩa có cấu trúc
        |                              |
        +------------------------------+--> Case Service
                                                |
                                   ALPR / Legal RAG khi bật
                                                |
                                                v
                                      PostgreSQL + Review UI

Kafka: vận chuyển sự kiện, metadata, URI và hash giữa các dịch vụ
Spark: tiêu thụ sự kiện, xử lý cửa sổ thời gian, tổng hợp và phát lại
```

### 5.2. Các ranh giới kiến trúc

- **Luồng liên tục:** ingest, quality gate, detector, tracker, timeline đèn và rule engine.
- **Luồng theo sự kiện:** tạo bằng chứng, VLM, ALPR, Legal RAG và kiểm duyệt.
- **Luồng phân tích:** Kafka và Spark tổng hợp chỉ số theo camera/thời gian, hỗ trợ replay và đánh giá.
- **Luồng quản trị:** quản lý camera, cấu hình hình học, mô hình, tài khoản và nhật ký.

Không gửi toàn bộ video dưới dạng payload JSON qua Kafka trong phương án đề xuất. Media nằm trong bộ đệm và object storage; message mang tham chiếu cùng checksum. Detector/tracker có thể trao đổi frame trong cùng tiến trình hoặc cơ chế nội bộ thích hợp; chưa bắt buộc tách thành microservice riêng cho mỗi thao tác.

## 6. Thành phần và trách nhiệm

| Thành phần | Đầu vào | Trách nhiệm | Đầu ra |
| --- | --- | --- | --- |
| Ingest Service | Video/RTSP và cấu hình nguồn | Giải mã, đánh số frame, giữ PTS, reconnect, quản lý phiên | Frame, segment, health event |
| Quality Gate | Frame và ROI | Phát hiện đen/xám/mờ/đứng hình, theo dõi khả năng quan sát | Quality flags, đoạn không dùng được |
| Perception Service | Frame/ROI đủ điều kiện | Chạy model xe và đèn, chuyển tọa độ về ảnh nguồn | Detections và model provenance |
| Tracking Service | Detections theo thứ tự | ByteTrack, vòng đời track, quỹ đạo, khoảng mất dấu | Tracks có ID theo phiên |
| Signal Service | Detections trong ROI đèn | Ghép đèn với `signal_id`, tổng hợp trạng thái theo thời gian | Timeline và khoảng chuyển màu |
| Scene/Rule Service | Tracks, timeline và SceneConfig | Xác định lượt, hướng đi, vượt vạch và bất định | Normal/candidate/ambiguous observations |
| Evidence Service | Trigger và bộ đệm | Trích clip trước/sau, keyframe, overlay và manifest | URI, hash, độ đầy đủ bằng chứng |
| VLM Service | Gói bằng chứng | Nhận xét hành vi, ngữ cảnh và giới hạn quan sát | Kết quả JSON có dẫn chiếu frame |
| ALPR Service | Crop/clip phương tiện | Phát hiện biển số, OCR, tổng hợp theo thời gian | Chuỗi đọc được, phương án và độ tin cậy |
| Legal RAG Service | Hành vi, loại xe, thời điểm, nguồn luật | Truy xuất điều khoản phù hợp theo hiệu lực | Tài liệu, điều/khoản và trạng thái tìm kiếm |
| Case Service | Kết quả các nhánh | Quản lý hồ sơ, trạng thái và phiên bản phân tích | Hồ sơ chuẩn hóa |
| Review API/UI | Hồ sơ, bằng chứng và người dùng | Xem, lọc, ghi nhận xét, duyệt và yêu cầu phân tích lại | Review revision và audit |
| Analytics Service | Kafka events | Spark aggregation, thống kê và kiểm tra backlog | Chỉ số theo camera/cửa sổ |

## 7. Tác nhân và quyền

| Vai trò | Quyền chính |
| --- | --- |
| Operator | Đăng ký nguồn, bật/tắt xử lý nguồn, xem health và tiến độ |
| Annotator | Gán bbox/timeline/hành vi trên bộ dữ liệu được phân công |
| Reviewer | Xem bằng chứng, chấp nhận/bác bỏ/yêu cầu thêm thông tin, ghi lý do |
| ML engineer | Đăng ký model, cấu hình thử nghiệm, chạy benchmark và đề xuất phát hành |
| Administrator | Quản lý người dùng, kích hoạt cấu hình/model, chính sách lưu trữ |
| Legal curator | Quản lý phiên bản tài liệu pháp lý nếu bật Legal RAG |

Một người có thể giữ nhiều vai trò trong đồ án, nhưng hệ thống vẫn phải lưu người thực hiện và thời điểm. Trong đánh giá test, người kiểm tra nhãn phải được phân biệt với quá trình sinh dự đoán.

## 8. Yêu cầu chức năng

Các mã dưới đây thuộc bản đặc tả này, không đồng nhất số thứ tự FR trong S2.

| Mã | Yêu cầu | Mức | Tiêu chí kiểm tra |
| --- | --- | --- | --- |
| FR-01 | Đăng ký video, checksum, độ phân giải, timebase và phiên nguồn | Phải | Phát lại được đúng nguồn đã đăng ký |
| FR-02 | Tiếp nhận RTSP và reconnect theo cấu hình | Giai đoạn tích hợp | Mất một nguồn không dừng các nguồn khác |
| FR-03 | Lưu source time, processing time và sai lệch thời gian riêng | Phải | Không dùng thời gian xử lý thay thời điểm vượt vạch |
| FR-04 | Kiểm tra chất lượng theo ROI và theo đoạn | Phải | Đoạn hỏng có cờ và lý do, không âm thầm coi là bình thường |
| FR-05 | Quản lý SceneConfig có phiên bản và kiểm tra hình học | Phải | Không kích hoạt cấu hình thiếu vạch/đèn/quan hệ điều khiển |
| FR-06 | Phát hiện sáu lớp xe (§15.3) và lưu confidence | Phải | Lớp từ checkpoint được ánh xạ đúng taxonomy; checkpoint 4 lớp cũ chỉ chạy được khi cấu hình khai báo rõ là 4 lớp |
| FR-07 | Ước lượng trạng thái đèn và khoảng chuyển màu | Phải | Có `unknown` khi không quan sát được |
| FR-08 | Theo dõi xe theo camera/session | Phải | ID không bị hiểu là định danh toàn hệ thống |
| FR-09 | Quản lý lượt xe, phạm vi hướng đi và khoảng che khuất | Phải | Xe dừng rồi đi tiếp không tự tạo hai lượt |
| FR-10 | Phát hiện vượt vạch theo phần đầu xe và chiều tiến | Phải | Xe ở sẵn sau vạch không tạo sự kiện vượt giả |
| FR-11 | Đối chiếu crossing interval với đúng signal interval | Phải | Các trường hợp giao thời không bị ép dương/âm |
| FR-12 | Tạo và lưu gói bằng chứng có hash | Phải | Tìm được frame trước/sau và cảnh gốc liên quan |
| FR-13 | VLM trả kết quả có cấu trúc, không chặn CV | Phải trong bản VLM | Timeout/lỗi JSON có trạng thái riêng |
| FR-14 | Lưu hành vi, ngoại lệ quan sát và kết quả người xem riêng | Phải | Có thể truy lại từng nguồn kết luận |
| FR-15 | Lưu hồ sơ và xử lý replay không trùng | Phải | Gửi lại cùng event không tăng số hồ sơ |
| FR-16 | UI có video, bbox, nhãn, timeline và cột ghi chú | Phải | Reviewer hoàn tất QA mà không sửa JSON trực tiếp |
| FR-17 | Xuất kết quả đánh giá và dữ liệu QA cục bộ | Phải | Có manifest, phiên bản nhãn và đường dẫn media |
| FR-18 | Xuất sự kiện Kafka và tổng hợp bằng Spark | Phải trong bản phân tán | Đối soát được số sự kiện trước/sau replay |
| FR-19 | Nhận dạng biển số | Mở rộng | Cho phép không đọc được và không suy đoán ký tự |
| FR-20 | Legal RAG theo ngày hiệu lực | Mở rộng | Kết quả phải có nguồn; không tìm thấy thì ghi rõ |
| FR-21 | Quản lý model/config và rollback | Phải | Hồ sơ cũ giữ nguyên phiên bản đã dùng |
| FR-22 | Theo dõi lỗi, GPU, FPS, queue và dung lượng | Phải | Nhìn thấy bottleneck và các đoạn không xử lý |
| FR-23 | Phân quyền, nhật ký và xóa theo chính sách | Phải | Hành động sửa/xóa có actor và lý do |

## 9. Quy tắc thời gian, hình học và hành vi

### 9.1. Thời gian

Frame phải có `source_id`, `session_id`, `frame_id`, `source_pts` hoặc `source_time_s`. `capture_ts_utc` chỉ được điền khi có ánh xạ UTC đáng tin cậy; nếu chưa xác minh, để `null` và `timebase_verified=false`.

Không suy thời gian nguồn từ FPS danh nghĩa khi có PTS hợp lệ. Video quay màn hình phải ghi rõ `capture_method`, vì FPS encoder và thời gian quay màn hình không chứng minh FPS/thời gian của camera gốc. Khi nguồn nhảy thời gian hoặc reconnect, mở discontinuity/phiên mới phù hợp và không nối quỹ đạo qua khoảng mất dữ liệu một cách tự động.

### 9.2. Hình học cảnh

SceneConfig tối thiểu gồm:

- Kích thước ảnh tham chiếu, frame tham chiếu và hệ tọa độ.
- `approach_roi`, `observation_roi`, hai điểm của vạch dừng và phía trước/sau vạch.
- Vector hướng đi được xét; quan hệ giữa vùng tiếp cận và `signal_id`.
- ROI của từng đèn điều khiển và trạng thái xác nhận mapping.
- Phiên bản, người kiểm tra, thời điểm kích hoạt và checksum nguồn tham chiếu.

Bbox và hình học chuẩn lưu theo pixel ảnh nguồn. Nếu resize, letterbox, crop hoặc quay ảnh, phải lưu phép biến đổi và đảo biến đổi khi xuất kết quả. Không kéo giãn ảnh sang 1280×720 làm thay đổi tỉ lệ hình học. Camera PTZ đổi góc khiến SceneConfig cũ mất hiệu lực; phải dừng quy tắc cho đến khi cấu hình lại.

### 9.3. Xác định vượt vạch

Mốc quan sát là phần đầu phương tiện bắt đầu vượt mép vạch phía xe tiếp cận. Có thể dùng một điểm/biên ước lượng từ bbox theo góc camera để tạo ứng viên, nhưng phải hiệu chuẩn và đánh giá sai số; điểm giữa bbox không mặc nhiên đại diện cho đầu xe.

Với đường vạch có hai điểm A, B và điểm tham chiếu P, dấu tích có hướng `(Bx−Ax)(Py−Ay)−(By−Ay)(Px−Ax)` giúp xác định phía của P. Đổi dấu chỉ là một điều kiện hình học: phải kết hợp chiều tiến, ROI, biên sai số, sự ổn định track và khả năng quan sát phần đầu xe.

Khoảng vượt `[crossing_time_min, crossing_time_max]` được giới hạn bởi quan sát trước/sau đáng tin cậy. Mất frame hoặc che khuất làm khoảng này rộng hơn; không được tự điền thời điểm chính xác giả. Giữ nhiều crossing trong cùng lượt nếu xe lùi rồi tiến lại.

### 9.4. Đèn và mapping

Detector đèn nhận diện các lớp `red`, `yellow`, `green`, `off`. Tầng hệ thống bổ sung `unknown` và `other` để biểu diễn quan sát chưa đủ hoặc tín hiệu chưa hỗ trợ; các giá trị này không nhất thiết là lớp train.

Màu của từng đèn điều khiển được đọc trong ROI đèn khai báo ở SceneConfig, không dựa vào việc tự gom box trên toàn ảnh. Thứ tự ưu tiên:

1. **Theo vị trí bóng sáng:** áp dụng cho đèn ba bóng có khai báo `lamp_layout` (`vertical3`: đỏ trên, vàng giữa, xanh dưới; `horizontal3`: đỏ trái, vàng giữa, xanh phải). ROI được chia ba vùng; vùng có bóng sáng trội hơn hẳn hai vùng còn lại cho ra màu. Quy tắc này đọc được đèn vàng VN có màu cam mà detector dễ nhầm sang đỏ.
2. **Theo detector + bộ phân loại màu:** box của detector đèn nằm trong ROI được đọc lại màu bằng bộ phân loại ảnh cắt (gồm lớp `none`). Không có box thì phân loại toàn ROI. Kết quả `none` hoặc không đọc được không tạo màu.
3. **Không khai báo `lamp_layout`** cho đèn có bóng mũi tên hoặc đèn đếm giây; những đèn này đi theo bước 2. Đèn có mũi tên rẽ được đánh dấu `has_turn_arrow`: màu xanh đọc được từ đèn đó không chứng minh quyền đi thẳng.

Màu theo từng frame được bỏ phiếu theo thời gian (cửa sổ 2 s, chỉ dùng frame đã qua); không có quan sát trong 6 s thì trạng thái là `unknown`. Chuyển trực tiếp xanh→đỏ mà không thấy vàng thì 3,5 s đầu của khoảng đỏ được coi là chưa chắc màu, nên crossing rơi vào đó là `insufficient_evidence`. Các tham số trên là giá trị đã thử trên dữ liệu pilot, phải ghi theo phiên bản cấu hình.

Trạng thái phải được gắn với đèn điều khiển đúng hướng xe. Không chọn đèn gần xe nhất chỉ dựa trên khoảng cách pixel. Temporal smoothing phải giữ lịch sử quan sát và độ trễ; không được hồi tố một màu qua khoảng chuyển không rõ rồi coi đó là dữ kiện chắc chắn.

### 9.5. Bảng quyết định hành vi

| Tình huống | Kết quả |
| --- | --- |
| Toàn bộ khoảng vượt nằm trong khoảng đỏ rõ ràng của đúng đèn | `red_light_crossing` |
| Khoảng vượt nằm hoàn toàn trong trạng thái xác minh không đỏ | `no_red_light_crossing` cho lần vượt đó |
| Xe dừng trước vạch, lượt được quan sát đầy đủ, không có lần vượt đỏ | `no_red_light_crossing` |
| Khoảng vượt chạm/chồng khoảng chuyển màu, không xác định thứ tự | `insufficient_evidence` |
| Không rõ đầu xe, vạch, đèn liên quan hoặc thời gian | `insufficient_evidence` |
| Xe đã ở trong giao lộ khi bắt đầu clip | Không suy ra vượt đỏ; đánh dấu cắt cụt/thiếu bằng chứng |
| Xe rẽ hoặc loại xe ngoài phạm vi | `scope_status=out_of_scope`, `assessment=null` |
| Không xác định được hướng đi hoặc loại xe | `scope_status=scope_uncertain`, `assessment=null` |

Đối với lượt có nhiều lần vượt: dương tính nếu có ít nhất một lần vượt đỏ đủ bằng chứng; nếu không có lần dương tính nhưng có lần chưa rõ thì kết luận thiếu bằng chứng. `off` không tự chứng minh quyền đi hoặc kết luận pháp lý.

Các nhãn trên mô tả hành vi quan sát. Sự xuất hiện của người điều tiết hoặc biển/tín hiệu phụ được lưu riêng với trạng thái `present`, `not_observed`, `unclear` và mức liên quan tới hướng xe. Không tự đổi hành vi vượt đỏ thành âm tính vì có một dấu hiệu ngoại lệ.

## 10. Evidence, VLM, ALPR và Legal RAG

### 10.1. Gói bằng chứng

Mỗi gói phải chứa clip cảnh đầy đủ, các keyframe có timestamp, vị trí xe mục tiêu, trạng thái chất lượng, SceneConfig, dữ kiện CV dự đoán và phiên bản mô hình. Crop xe/đèn chỉ bổ sung để xem chi tiết, không thay toàn bộ bối cảnh.

Phải lưu media gốc và overlay riêng. Manifest liệt kê URI, SHA-256, kích thước, khoảng thời gian, frame ID, phép biến đổi tọa độ, các phần thiếu và `evidence_version`. Hash phát hiện thay đổi file nhưng không tự chứng minh tính xác thực của nguồn camera.

`pre_context_s`, `post_context_s` và số keyframe được chọn trên dev. Thời gian chờ frame sau sự kiện phải tính trong latency. Khi bộ đệm thiếu đoạn trước/sau, gói mang `incomplete=true`; không tạo dữ kiện thay thế.

### 10.2. Hợp đồng VLM

VLM nhận câu hỏi hẹp về xe mục tiêu, quan sát vượt vạch, tín hiệu điều khiển, dấu hiệu ngoại lệ và mức đầy đủ của bằng chứng. Dữ kiện từ detector/tracker được đánh dấu `predicted`, không đưa nhãn GT vào prompt.

Output tối thiểu:

| Trường | Nội dung |
| --- | --- |
| `assessment` | Một trong ba nhãn hành vi khi thuộc phạm vi |
| `supporting_frame_ids` | Chỉ được tham chiếu frame có trong gói |
| `observed_facts` | Các quan sát ngắn gọn có dẫn chiếu |
| `uncertainty_reasons` | Những phần không nhìn thấy hoặc không chắc chắn |
| `exception_observations` | Người điều tiết/tín hiệu phụ và mức liên quan |
| `cv_disagreements` | Điểm bất đồng với dữ kiện CV dự đoán |
| `explanation` | Giải thích ngắn, không bổ sung chi tiết ngoài bằng chứng |
| `model_version`, `prompt_version` | Phiên bản dùng để tái lập |

Kết quả phải qua kiểm tra schema, enum và frame references. Timeout, JSON sai hoặc tham chiếu không hợp lệ chuyển sang lỗi phân tích/yêu cầu kiểm tra. Không sửa facts gốc để khớp output VLM; giữ cả hai và đánh dấu mâu thuẫn.

### 10.3. ALPR và Legal RAG

ALPR được thực hiện bởi detector biển số và OCR chuyên dụng nếu bật. Ghi `unreadable` nếu biển quá nhỏ/mờ, giữ các phương án chưa chắc chắn; không bắt buộc có biển số mới lưu được sự kiện nghi vấn.

Legal RAG nhận hành vi quan sát, loại xe, thời điểm và bối cảnh. Kho tài liệu phải có nguồn, phiên bản, điều/khoản, thời điểm hiệu lực và trạng thái kiểm duyệt. Kết quả trả nguồn tra cứu, không tự sinh mức phạt thiếu căn cứ. Khi không biết ngày sự kiện hoặc không có tài liệu phù hợp, trả `DATE_UNKNOWN` hoặc `LEGAL_MATCH_NOT_FOUND`.

Bản đặc tả không xác nhận nội dung pháp luật hiện hành hay ấn định mức phạt. Nghiệm thu nhánh RAG cần bộ câu hỏi và nguồn pháp lý được kiểm chứng riêng tại thời điểm triển khai.

## 11. Hợp đồng dữ liệu và Kafka

### 11.1. Envelope chung

Mọi event cần `schema_version`, `message_id`, `event_type`, `produced_at_utc`, `camera_id`, `session_id`, `source_id`, `correlation_id`, `producer_version` và `payload`. Những event thuộc hồ sơ phải có `event_id` và `analysis_revision`.

`message_id` định danh thông điệp; `event_id` định danh hồ sơ logic; không dùng chúng thay thế nhau. Khi retry, giữ khóa idempotency. Khi phân tích lại, tạo revision mới liên kết hồ sơ cũ, không âm thầm ghi đè.

### 11.2. Topic đề xuất

| Topic | Partition key | Nội dung |
| --- | --- | --- |
| `camera.health.v1` | `camera_id` | Online/offline, chất lượng, thời gian và gián đoạn |
| `video.segment.v1` | `camera_id` | URI, hash, timebase, thời gian đầu/cuối |
| `detections.v1` | `camera_id` | Detections được chọn theo chính sách xuất dữ liệu |
| `tracks.v1` | `camera_id` | Track updates, lifecycle và quality flags |
| `candidate.violation.v1` | `camera_id` | Crossing, signal, lý do tạo ứng viên |
| `evidence.ready.v1` | `event_id` | Manifest và trạng thái đầy đủ |
| `vlm.request.v1`, `vlm.result.v1` | `event_id` | Yêu cầu/kết quả VLM có revision |
| `legal.request.v1`, `legal.result.v1` | `event_id` | Yêu cầu/kết quả RAG khi bật |
| `review.decision.v1` | `event_id` | Quyết định người xem và revision |
| `pipeline.deadletter.v1` | `camera_id` hoặc `event_id` | Message lỗi, nguyên nhân và số lần thử |

Không giả định có thứ tự toàn cục giữa mọi partition/topic. Consumer kiểm tra phiên và revision; kết quả cũ đến muộn không được thay kết quả mới. Topic detections/tracks có thể tắt hoặc giảm tần suất cho deployment nhỏ, nhưng phải ghi rõ profile đã dùng khi benchmark.

### 11.3. Vai trò Spark

Spark đọc metadata/events để tổng hợp số lượt, số ứng viên, tỷ lệ thiếu bằng chứng, chất lượng camera và độ trễ theo cửa sổ thời gian. Watermark và ngưỡng dữ liệu đến muộn là cấu hình được chọn theo thực nghiệm; dữ liệu trễ quá ngưỡng phải có bộ đếm và đường xử lý riêng.

Job lưu checkpoint bền vững. Ghi PostgreSQL bằng khóa duy nhất và thao tác idempotent; không tuyên bố exactly-once toàn pipeline chỉ vì sử dụng checkpoint. Phải kiểm thử restart, replay và lỗi giữa ghi dữ liệu/commit offset.

## 12. Mô hình lưu trữ

### 12.1. PostgreSQL

| Bảng logic | Khóa/quan hệ chính | Nội dung |
| --- | --- | --- |
| `sources` | `source_id` | Nguồn, URI, hash, timebase, điều kiện sử dụng |
| `camera_sessions` | `session_id`, `camera_id` | Thời gian phiên, profile, gián đoạn |
| `scene_configs` | `scene_config_id`, `version` | Hình học, mapping, người duyệt, thời gian hiệu lực |
| `model_versions` | `model_version_id` | Artifact hash, lớp, runtime, dataset manifest |
| `processing_runs` | `run_id` | Model/config/rule/prompt phiên bản và môi trường chạy |
| `approaches` | `approach_id`, `run_id` | Track liên quan, hướng đi, phạm vi, cắt cụt |
| `signal_intervals` | `interval_id`, `signal_id`, `run_id` | Trạng thái, thời gian và nguồn quan sát |
| `crossings` | `crossing_id`, `approach_id` | Khoảng vượt, signal interval, bất định |
| `cases` | `event_id` | Hồ sơ logic và trạng thái review |
| `analysis_revisions` | `event_id`, `revision` | Kết quả CV/rule/VLM/ALPR/RAG theo lần chạy |
| `evidence_objects` | `object_id`, `evidence_version` | URI, hash, vai trò và khoảng thời gian |
| `review_revisions` | `event_id`, `review_revision` | Reviewer, quyết định, lý do, phiên bản đã xem |
| `audit_events` | `audit_id` | Actor, thao tác, đối tượng, thời gian, thay đổi |
| `legal_documents`, `legal_provisions` | ID tài liệu/điều khoản | Nguồn, hiệu lực và version nếu bật RAG |

GT nằm trong namespace/bảng annotation riêng, có `label_version`, `annotator_id`, `reviewer_id`, `review_status`. Không dùng chung cột cho GT và dự đoán.

Các ràng buộc bắt buộc: khoảng thời gian đầu không lớn hơn cuối; bbox có diện tích dương và nằm trong kích thước tham chiếu; phiên bản được tham chiếu phải tồn tại; trạng thái đèn xác định không chồng lấn mâu thuẫn trong cùng timeline; trạng thái ngoài phạm vi có assessment rỗng và lý do.

### 12.2. MinIO hoặc object storage tương đương

Cấu trúc logic đề xuất:

```text
sources/{source_id}/original/
segments/{camera_id}/{session_id}/
evidence/{event_id}/{evidence_version}/original/
evidence/{event_id}/{evidence_version}/overlays/
evidence/{event_id}/{evidence_version}/manifest.json
models/{model_version_id}/
reports/{run_id}/
```

PostgreSQL lưu metadata và liên kết; không lưu toàn bộ ảnh/video dưới dạng BLOB cho luồng chính. Đường dẫn object không chứa token. UI nhận URL có thời hạn hoặc qua API được phân quyền.

## 13. Trạng thái hồ sơ và độ tin cậy giao dịch

Trạng thái kỹ thuật đề xuất:

```text
DETECTED -> EVIDENCE_PENDING -> ANALYSIS_PENDING -> REVIEW_READY
                |                    |
                v                    v
        EVIDENCE_INCOMPLETE     ANALYSIS_FAILED
                |                    |
                +---- bổ sung / retry ----+
```

Trạng thái kiểm duyệt tách riêng: `PENDING`, `ACCEPTED`, `REJECTED`, `NEEDS_MORE_INFO`. Chấp nhận hồ sơ nghĩa là người xem chấp nhận kết luận ghi trong revision, không phải tự động xử phạt. Hồ sơ thiếu bằng chứng vẫn phải xem được để xử lý lỗi, nhưng không hiển thị như hồ sơ đã được xác minh đầy đủ.

Mỗi quyết định review lưu revision phân tích và evidence hash mà người xem đã kiểm tra. Phân tích mới không tự kế thừa phê duyệt cũ. Thay đổi quyết định tạo review revision mới.

Case Service nên dùng outbox giao dịch hoặc cơ chế tương đương để bảo đảm thay đổi DB và event cần phát không bị lệch khi crash. Upload media phải hoàn thành và kiểm tra hash trước khi phát `evidence.ready`. Retry có giới hạn; vượt giới hạn vào dead-letter, không mất hồ sơ im lặng.

## 14. API và giao diện kiểm duyệt

### 14.1. API đề xuất

| API | Chức năng |
| --- | --- |
| `POST /api/v1/sources` | Đăng ký video hoặc nguồn RTSP |
| `GET /api/v1/cameras/{id}/health` | Health, chất lượng, lag |
| `POST /api/v1/cameras/{id}/scene-configs` | Tạo draft hình học |
| `POST /api/v1/scene-configs/{id}/activate` | Kích hoạt phiên bản đã kiểm tra |
| `POST /api/v1/runs` | Tạo phiên xử lý với cấu hình cố định |
| `GET /api/v1/runs/{id}` | Tiến độ, lỗi và thống kê |
| `GET /api/v1/cases` | Lọc theo camera, thời gian, trạng thái, loại xe |
| `GET /api/v1/cases/{id}` | Hồ sơ và các revision |
| `GET /api/v1/cases/{id}/evidence` | Manifest và quyền xem media |
| `POST /api/v1/cases/{id}/reviews` | Quyết định, ghi chú, revision đã xem |
| `POST /api/v1/cases/{id}/reanalysis` | Phân tích lại với revision mới |
| `GET /api/v1/models` | Danh sách model và taxonomy |
| `POST /api/v1/models/{id}/activate` | Kích hoạt model có kiểm tra tương thích |
| `GET /api/v1/audit` | Truy vấn nhật ký theo quyền |

Request thay đổi dữ liệu mang `Idempotency-Key`; request review mang `expected_revision` để tránh hai người ghi đè. API danh sách có phân trang. Lỗi trả `code`, `message`, `request_id`, thông tin có thể khắc phục; không đưa token, đường dẫn bí mật hoặc stack trace ra UI.

### 14.2. Nội dung UI

UI phải có danh sách sự kiện, bộ lọc, cột ghi chú, vùng phát video, tua frame, bật/tắt overlay, ảnh phóng to xe/đèn và timeline crossing–signal. Người xem có thể chuyển giữa toàn cảnh và crop mà không mất vị trí thời gian.

Trong chế độ QA dataset, GT và PRED đặt cạnh nhau, tên lớp và nguồn nhãn hiển thị rõ. Trong kiểm duyệt vận hành, không hiển thị dự đoán như GT. Mọi bbox cần màu tương phản, độ dày và cỡ chữ phù hợp; crop phóng to không được gọi là ảnh được phục hồi chi tiết thật.

Với quy trình QA hiện tại, ưu tiên công cụ chạy cục bộ và xuất file kết quả trên máy. Dashboard trong kiến trúc đầy đủ là giao diện phục vụ hệ thống; không yêu cầu tạo website công khai cho từng đợt QA.

## 15. Dataset, QA và huấn luyện

### 15.1. Phiên bản và chia tập

Mỗi bản dataset có manifest nguồn, checksum, mapping lớp, phép biến đổi, chính sách lọc, thống kê ảnh/bbox và kết quả QA. Ảnh crop, frame gần nhau, bản sao và clip cùng nguồn phải được nhóm trước khi chia train/dev/test.

Không trộn bản nguồn và crop của cùng ảnh sang hai split. Có thể dùng cùng camera ở dev/test theo S3 nếu phiên nguồn độc lập; phải công bố đây là kiểm tra cùng camera khác phiên. Khả năng tổng quát sang camera mới cần test riêng theo camera giữ lại.

### 15.2. Chất lượng hình ảnh và bbox

Giữ mẫu có thể nhận diện rõ đối tượng/trạng thái; ngưỡng không chỉ dựa vào độ phân giải toàn ảnh. Đo kích thước bbox ở ảnh nguồn và sau phép resize thực tế của mô hình. Đèn nhỏ còn phân biệt được màu có tiêu chí khác xe cần phân biệt car/truck/motorcycle.

Không đơn giản xóa nhãn bbox nhỏ rồi để nguyên đối tượng trong ảnh train: đối tượng có thể bị học thành nền. Các lựa chọn hợp lệ phải được ghi rõ: loại cả ảnh, tạo crop không chứa vùng không gán nhãn, hoặc dùng cơ chế ignore đã xác minh được hỗ trợ trong pipeline. Không mặc định YOLO label TXT tiêu chuẩn có vùng ignore.

Train có thể kết hợp full-scene và crop đã QA, nhưng cần thử nghiệm tỉ lệ trên dev và bảo toàn ngữ cảnh cho phần phát hiện sự kiện. Validation/test chính phải đại diện video triển khai, không chỉ gồm crop rõ đã chọn. Không lọc bỏ các trường hợp khó của test để tăng điểm; báo cáo nhóm đọc được/không đọc được và lý do riêng.

### 15.3. Nhãn và kiểm tra

- Sáu lớp xe (chốt 21/11/2026), mã lớp 0–5:

  | Mã | Lớp | Gồm |
  | --- | --- | --- |
  | 0 | `car` | Ô tô con, SUV, MPV/minivan 7 chỗ |
  | 1 | `motorcycle` | Xe máy, xe máy điện |
  | 2 | `bus` | Xe khách và xe buýt trên 16 chỗ |
  | 3 | `truck` | Xe tải có cabin riêng và thùng, xe ben, đầu kéo, container |
  | 4 | `pickup` | Xe bán tải (cabin + thùng hở phía sau) |
  | 5 | `minibus` | Xe từ 16 chỗ trở xuống dạng thùng (Hiace, Transit, Solati…) và mọi loại van, kể cả van chở hàng |

  Mã 0–3 giữ nguyên so với taxonomy 4 lớp cũ, nên nhãn cũ chuyển sang bằng cách tách `truck` thành `truck`/`pickup`/`minibus`; `bus` cũ cần soát lại các xe ≤16 chỗ. Mọi thành phần còn giả định 4 lớp (tên lớp, nhóm lớp khi nối track, luật, cấu hình tracking) phải cập nhật trước khi kích hoạt model 6 lớp.
- Bốn lớp màu đèn hiện tại; `unknown` được xử lý ở tầng dữ liệu/hệ thống khi không đủ quan sát.
- Bbox đánh dấu phần nhìn thấy theo quy ước đã công bố; không tự ước lượng vật thể bị che hoàn toàn.
- Bộ test sự kiện gán mọi lượt đủ phạm vi trong đoạn video liên tục, gồm xe tuân thủ.
- Nhãn toàn video `assessment` và nhãn gói `package_assessment` được gán, lưu và đánh giá riêng.
- Pilot QA mẫu không tự chứng minh toàn bộ dataset đã duyệt từng ảnh.

### 15.4. Huấn luyện và phát hành model

Chạy YOLO26s hoặc phiên bản khác thông qua cùng giao diện detector. Phiên bản mới được chọn bằng benchmark của dự án, không chỉ theo tên đời model. YOLOv8 được giữ làm baseline khi artifact và tập đánh giá tương thích còn sẵn.

Mỗi run lưu phiên bản thư viện, model khởi tạo, seed, imgsz, batch, augmentation, optimizer, số epoch, early stopping, class mapping và dataset manifest. `last.pt` phục vụ tiếp tục phiên train nếu còn trạng thái cần thiết; `best.pt` phục vụ chọn model theo tiêu chí validation đã định nghĩa. Khả năng resume phải được kiểm tra trên checkpoint thực tế.

Theo ưu tiên dung lượng của dự án, bản backup mặc định giữ latest và best theo từng task/run, không tự lưu vĩnh viễn checkpoint mọi epoch. Chỉ thay bản backup mới sau khi xác minh upload hoàn tất. Xe và đèn dùng thư mục riêng; không ghi đè chéo.

Test chỉ dùng sau khi chọn cấu hình trên dev. Không dùng kết quả test để chỉnh ngưỡng rồi tiếp tục gọi cùng tập đó là test độc lập.

## 16. Hiệu năng, realtime và tài nguyên

Các mục tiêu thời gian dưới đây kế thừa hướng đề xuất của S2, **chưa phải kết quả đạt được**. Phải chốt profile camera, FPS đầu vào, tốc độ xử lý tối thiểu và giới hạn chấp nhận trước nghiệm thu.

| Mã | Chỉ tiêu thiết kế | Cách kiểm tra |
| --- | --- | --- |
| NFR-01 | Perception lag P95 <500 ms trên profile đã chọn | Đo từ frame sẵn sàng ở ingest tới track/rule input sẵn sàng |
| NFR-02 | Candidate latency P95 <1 s sau mốc sự kiện | Đo theo nguồn thời gian đã ánh xạ; báo riêng sai số crossing |
| NFR-03 | VLM service latency P95 mục tiêu <10 s | Từ worker nhận gói đến output hợp lệ; báo queue wait riêng |
| NFR-04 | Không tăng backlog không giới hạn | Soak test ở tải mục tiêu và tải vượt ngưỡng |
| NFR-05 | Không mất âm thầm hồ sơ sau retry/replay | Đối soát ID và các trạng thái lỗi |
| NFR-06 | Mọi candidate có provenance | Kiểm tra model/config/source/evidence version |
| NFR-07 | Lỗi một nguồn không dừng nguồn khác | Fault injection theo camera |

Tổng độ trễ hồ sơ phải bao gồm ingest/decode, CV, rule, chờ post-context, tạo evidence, chờ VLM, suy luận và ghi hồ sơ. Không đồng nhất 1/inference-time với FPS realtime toàn pipeline. Báo cáo số frame bỏ, khoảng trống quan sát và tỷ lệ sự kiện thiếu bằng chứng.

Benchmark realtime dùng batch=1 hoặc batch đúng profile triển khai, warm-up, đồng bộ GPU khi đo, cùng input size và cùng precision/runtime giữa các model. Ghi phần cứng, nhiệt/tải nền, số camera, P50/P95/P99, FPS xử lý và hệ số thời gian thực cho phát lại video.

Baseline phần cứng trong tài liệu là RTX 3060 12 GB. Đề xuất VLM concurrency=1, ưu tiên ngân sách GPU cho detector. Hai model xe/đèn trên một GPU phải được đo chung; không giả định cộng hai FPS đo riêng là tốc độ hệ thống. Số camera tối đa và FPS được chốt bằng benchmark, không ấn định khi chưa đo.

## 17. Xử lý lỗi và vận hành

| Sự cố | Hành vi yêu cầu |
| --- | --- |
| Camera mất kết nối | Health offline, reconnect có backoff, đánh dấu đoạn thiếu |
| Frame đen/xám/đứng | Dừng dùng đoạn đó làm bằng chứng chắc chắn, ghi quality event |
| Chất lượng thấp nhưng vẫn có cảnh | Gắn cờ, không tự coi không có xe/vi phạm |
| Camera đổi góc/zoom | Vô hiệu SceneConfig liên quan đến khi kiểm tra lại |
| Detector OOM | Giảm tải theo profile dự phòng đã benchmark hoặc tạm dừng nguồn, ghi gap |
| Track mất/đổi ID gần vạch | Đánh dấu bất định. Bản 1.1: được nối có điều kiện (≤0,6 s, gần vạch, cùng nhóm xe, không qua khoảng mất frame) để không mất bằng chứng, nhưng mọi crossing chạm khoảng nối mang `review_required` và lý do, không được coi là kết luận chắc chắn khi chưa có người xem |
| VLM timeout | Giữ evidence, retry giới hạn, cho người xem xử lý |
| Kafka tạm lỗi | Ghi hàng đợi bền vững cục bộ có giới hạn và phát lại khi phục hồi |
| PostgreSQL tạm lỗi | Giữ spool bền vững, không báo hồ sơ đã lưu thành công |
| MinIO lỗi | Giữ staging nếu còn chỗ; không phát evidence ready khi chưa hoàn tất |
| Hết dung lượng/spool đầy | Cảnh báo và dừng thu nhận có kiểm soát, không giả vờ tiếp tục đầy đủ |
| Schema/model không tương thích | Từ chối kích hoạt, ghi mã lỗi |

Giám sát tối thiểu: source FPS, processed FPS, frame gaps, quality rejection, queue depth/age, Kafka lag, VLM latency, GPU memory/utilization, lỗi theo loại, dung lượng và số hồ sơ chờ kiểm duyệt.

## 18. An toàn dữ liệu và quản lý vòng đời

Thông tin truy cập camera, Kaggle, OAuth và rclone phải nằm trong cơ chế secrets phù hợp, không trong repository, manifest công khai hay log. Tài liệu này không chứa thông tin xác thực.

Áp dụng quyền theo vai trò, truy cập media có kiểm soát và audit cho kích hoạt cấu hình, sửa nhãn, duyệt hồ sơ, export và xóa dữ liệu. Chính sách lưu raw video, evidence, debug detections và báo cáo được cấu hình riêng; chưa tự đặt số ngày khi chưa thống nhất nhu cầu và dung lượng.

Xóa dữ liệu phải dựa trên manifest có thể kiểm tra, giữ quan hệ provenance cần thiết và trạng thái đã xóa. Không xóa dữ liệu nguồn hoặc artifact duy nhất chỉ vì tên thư mục cũ. Quay màn hình ứng dụng chỉ thu phần phục vụ dự án, không ghi tài khoản/màn hình cá nhân.

Yêu cầu vận hành hiện tại: ADB điện thoại đã được người dùng yêu cầu ngắt. Việc triển khai đặc tả không bao gồm tự kết nối lại điện thoại hoặc tự thu camera.

## 19. Kế hoạch kiểm thử và nghiệm thu

### 19.1. Các tầng đánh giá

| Tầng | Chỉ số cần báo cáo |
| --- | --- |
| Detector xe | Recall từng lớp (§19.2), precision, mAP50, mAP50–95, AP từng lớp, confusion car/truck/pickup/minibus và bus/minibus, FN motorcycle |
| Detector đèn | AP từng màu, recall theo kích thước bbox, confusion và `unknown` ở tầng hệ thống |
| Tracking | IDF1/HOTA khi có GT tương ứng, ID switches, mất dấu gần vạch |
| Hình học/thời gian | Sai số crossing, mapping đèn, tỷ lệ khoảng bất định |
| Candidate | Precision/recall sự kiện trên video liên tục, cảnh báo giả theo thời lượng |
| VLM | Chất lượng ba nhãn, từ chối đúng, frame-reference hợp lệ, mâu thuẫn với evidence |
| Pipeline | Recall sự kiện cuối, false alerts/hour, tỷ lệ thiếu bằng chứng, latency và backlog |
| Review | Khả năng truy lại quyết định, thời gian xử lý và tỷ lệ cần thêm thông tin |

Nếu tập tracking chỉ có bbox ở keyframe, không báo HOTA/IDF1 toàn quỹ đạo như thể có GT đủ mọi frame. Đánh giá chính phải công bố số mẫu, số sự kiện dương/âm/không rõ và điều kiện nguồn.

### 19.2. Quy tắc tính điểm sự kiện

Ghép dự đoán và GT một-một theo nguồn, phương tiện và khoảng thời gian với dung sai chốt trên dev. Nhiều cảnh báo cho một sự kiện không được tính nhiều TP; phần trùng được thống kê riêng theo quy tắc đánh giá đã khóa.

Trên các sự kiện dương tính xác định, hệ thống trả `insufficient_evidence` hoặc không sinh cảnh báo dương tính vẫn là bỏ sót trong phép đo recall đầu-cuối. Đồng thời báo cáo riêng chất lượng từ chối của VLM trên nhãn `package_assessment` để phân biệt lỗi cắt clip với lỗi suy luận.

Mẫu GT không đủ bằng chứng không được tự chuyển thành âm tính để tính confusion matrix nhị phân. Báo cáo nhóm này riêng và thêm đánh giá ba trạng thái.

**Ngưỡng nghiệm thu detector (chốt 02/11/2026):** recall từng lớp **≥ 95%** cho cả detector xe và detector đèn.

- Recall lớp c = số vật thể thật lớp c được phát hiện với IoU ≥ 0,5 **và gọi đúng lớp c**, chia cho **tổng số vật thể thật lớp c**. Vật thể bị bỏ sót và vật thể bị gọi sai lớp đều tính là không đạt. Đây không phải mAP.
- Xe: áp dụng cho từng lớp trong sáu lớp ở §15.3.
- Đèn: áp dụng cho `red`, `yellow`, `green`. Lớp `off` vẫn được train nhưng không tính vào ngưỡng, báo cáo riêng.
- Đo ở đúng kích thước ảnh và ngưỡng confidence mà pipeline sử dụng; luôn báo precision đi kèm vì recall phụ thuộc ngưỡng.
- Báo riêng theo nguồn: tập test của dataset và tập camera Việt Nam chưa dùng để train.

Ngưỡng precision đi kèm, ngưỡng recall sự kiện và cảnh báo giả theo giờ chưa được phê duyệt: phải điền trước khi chạy test nghiệm thu, không chọn sau khi nhìn kết quả.

### 19.3. Ca kiểm thử bắt buộc

| Mã | Ca kiểm thử | Kết quả mong đợi | Liên kết |
| --- | --- | --- | --- |
| TC-01 | Xe dừng trước vạch lúc đỏ | Không tạo kết luận vượt đỏ | FR-09–11 |
| TC-02 | Xe vượt rõ sau khi đèn đỏ | Candidate có crossing và đúng đèn | FR-10–12 |
| TC-03 | Xe qua vạch trước đỏ, còn trong giao lộ khi đỏ | Không suy vượt đỏ từ vị trí hiện tại | FR-11 |
| TC-04 | Crossing chồng khoảng chuyển màu | Thiếu bằng chứng | FR-07, FR-11 |
| TC-05 | Đèn đỏ thuộc hướng khác | Không dùng đèn sai để kết luận | FR-05, FR-07 |
| TC-06 | Mất dấu/che đầu xe ngay vạch | Bất định, không tạo thời điểm giả | FR-08–11 |
| TC-07 | Clip thiếu frame trước/sau | Evidence incomplete, bảo toàn lỗi | FR-12 |
| TC-08 | VLM trả JSON sai/timeout | Có trạng thái lỗi và review, CV tiếp tục | FR-13 |
| TC-09 | Phát lại cùng event hoặc crash khi ghi | Một hồ sơ logic, audit đầy đủ | FR-15, FR-18 |
| TC-10 | Người dùng không có quyền duyệt | API từ chối thao tác | FR-23 |
| TC-11 | Hai reviewer gửi cùng revision | Xung đột được phát hiện | FR-16 |
| TC-12 | PTZ đổi góc hoặc nguồn nhảy thời gian | Cấu hình/track không tiếp tục như cũ | FR-03, FR-05 |
| TC-13 | Đối tượng nhỏ bị lọc khỏi nhãn nhưng còn trong ảnh | Dataset audit phát hiện nguy cơ nhãn thiếu | Mục 15 |
| TC-14 | VLM thấy người điều tiết | Lưu dấu hiệu riêng, không tự đổi GT hành vi | FR-14 |
| TC-15 | Tăng tải vượt khả năng GPU | Cảnh báo và giảm tải có ghi gap | NFR-04, Mục 17 |

### 19.4. Hồ sơ nghiệm thu

Gói nghiệm thu gồm mã nguồn/commit, cấu hình deployment, model hashes, dataset manifests, báo cáo QA, báo cáo định lượng, log benchmark, danh sách lỗi chưa giải quyết và video demo có thể tái lập. Các mục mở rộng chưa triển khai phải được ghi rõ; không dùng kết quả detector đơn lẻ thay nghiệm thu toàn pipeline.

## 20. Kế hoạch triển khai theo mốc

| Mốc | Công việc | Điều kiện hoàn tất |
| --- | --- | --- |
| M1 — Chốt dữ liệu | Khóa taxonomy, split, QA, policy filter và scene test | Manifest tái lập, không rò rỉ nguồn giữa split |
| M2 — CV trên video | Xe/đèn, tracking, timebase, scene/rule, replay | Qua TC-01–06 và TC-12 trên bộ pilot |
| M3 — Hồ sơ và QA | Evidence, PostgreSQL/MinIO, review cục bộ/API | Xem lại được media và toàn bộ quyết định |
| M4 — Dữ liệu phân tán | Kafka events, Spark aggregates, checkpoint/outbox | Replay/crash không tăng trùng hồ sơ |
| M5 — VLM | Gói input, schema output, hàng đợi, xử lý lỗi | So sánh CV+rules với CV+rules+VLM trên cùng test |
| M6 — Đánh giá hệ thống | Realtime, độ chính xác, tải, lỗi và demo | Báo cáo đủ profile, latency, recall và giới hạn |
| M7 — Mở rộng | ALPR, Legal RAG, hướng rẽ/tín hiệu đặc biệt | Có dữ liệu, hợp đồng và nghiệm thu riêng từng nhánh |

Các mốc có thể phát triển song song khi không phá phụ thuộc dữ liệu. Huấn luyện trên Kaggle và triển khai inference trên máy cá nhân là hai môi trường; artifact phải được kiểm tra tương thích trước phát hành.

## 21. Hiện trạng và các quyết định còn mở

Theo bản context ngày 30/10/2026, dự án đã có các đợt chuẩn bị dataset/QA, huấn luyện YOLOv8/YOLO26s, thử nghiệm tracking và thu video camera. Điều đó chưa xác nhận Kafka, Spark, VLM, ALPR, RAG và dashboard đã tích hợp hoàn chỉnh. Bản đặc tả này không thực hiện kiểm toán mã nguồn từng dịch vụ và không tạo hoặc khởi chạy các dịch vụ đó.

Các quyết định cần chốt bằng thử nghiệm hoặc rà soát yêu cầu:

| Quyết định | Cách chốt |
| --- | --- |
| Giữ luồng sự kiện S2 hay bắt buộc Spark điều phối AI như S1 | Đối chiếu tiêu chí môn học và benchmark cả chi phí điều phối |
| Model xe/đèn cuối cùng | AP theo lớp + recall sự kiện + realtime trên cùng profile |
| Input size và ROI đèn | Thử trên dev; giữ ánh xạ tọa độ và chi phí toàn pipeline |
| FPS, số camera đồng thời | Soak test phần cứng đích và ngưỡng frame gap |
| VLM cụ thể và precision | Đo VRAM, latency, chất lượng trên gói evidence thực |
| Pre/post context, keyframes | Đo độ đầy đủ bằng chứng và latency trên dev |
| Ngưỡng chất lượng/bbox/confidence | QA mẫu đại diện và phân tích lỗi theo lớp/nguồn |
| Ngưỡng precision, recall sự kiện, false alerts để nghiệm thu | Thống nhất trước test chính (recall từng lớp của detector đã chốt ở §19.2) |
| Retention và dung lượng | Ước tính theo lưu lượng, quyền truy cập và nhu cầu giữ bằng chứng |
| ALPR/RAG có bắt buộc ở bản nộp không | Chốt phạm vi với chủ dự án/giảng viên |

## 22. Ma trận truy vết tài liệu

| Nội dung bản đặc tả | Căn cứ | Điều chỉnh trong bản này |
| --- | --- | --- |
| Kiến trúc CV + tracking + VLM + Big Data | S1, S2 | Trình bày rõ hai phương án luồng dữ liệu |
| Bốn lớp xe, đi thẳng, suspected red-light | S3 §1 | Dùng làm phạm vi pilot thay danh mục mọi vi phạm; bản 1.1 mở rộng thành sáu lớp xe theo quyết định của chủ dự án (21/11/2026) |
| Ba nhãn và khoảng crossing/chuyển màu | S3 §2–5 | Đưa thành quy tắc xử lý và ca kiểm thử |
| Signal mapping và cấu hình cảnh | S3, S2 | Quản lý version, transform và PTZ invalidation |
| Evidence, VLM, ALPR, Legal RAG | S1, S2 | Tách kết luận quan sát và quyết định pháp lý; ALPR/RAG là nhánh mở rộng pilot |
| Kafka/Spark/DB/object storage | S2 | Chi tiết key, revision, idempotency và lỗi |
| API, schema và state machine | S2 và thiết kế bổ sung | Hợp đồng đề xuất, không mô tả API đã triển khai |
| Dataset dễ nhìn, QA cục bộ, checkpoint tiết kiệm | S4 và yêu cầu trong trao đổi | Giữ ngữ cảnh full-scene, tránh bbox không nhãn thành nền |
| Latency mục tiêu | S2 | Phân biệt compute, queue, evidence wait và đầu-cuối |
| Tiến độ thực tế | S4 | Giữ riêng khỏi yêu cầu và tiêu chí nghiệm thu |

