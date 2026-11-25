# Đặc tả dataset — Traffic Violation Understanding

Trạng thái: bản đã chỉnh theo review; chưa phải phạm vi được giảng viên phê duyệt. Quy tắc gán nhãn và tính điểm dưới đây áp dụng cho pilot; các ngưỡng số được chốt trên tập phát triển trước test.

## 1. Các quyết định đã chốt

- Hành vi: nghi vượt đèn đỏ; `car`, `motorcycle`, `bus`, `truck` đi thẳng trong vùng cấu hình (phạm vi chủ dự án duyệt ngày 12/10/2026; bán tải thuộc `truck`).
- Nguồn: video có sẵn; ưu tiên Việt Nam, bổ sung quốc gia khác và báo cáo riêng theo bối cảnh.
- Input hệ thống: video liên tục từ camera cố định, timestamp và cấu hình camera thủ công.
- Điều kiện: ngày, đêm, mưa và nhiều mức chất lượng. Đây là mở rộng so với scope ban ngày trong báo cáo 26/09/2026; không mặc định cam kết chất lượng ngang nhau giữa các điều kiện.
- VLM: nhận clip/keyframe và dữ kiện CV dự đoán, thẩm định và giải thích.
- Dùng pretrained; tập phát triển phục vụ chỉnh prompt/quy tắc/ngưỡng; không mặc định fine-tune.
- Một đơn vị nhãn: một lượt tiếp cận vạch dừng của một phương tiện.
- Mốc vượt vạch: phần đầu phương tiện bắt đầu vượt qua vạch dừng theo hướng đi đã cấu hình; ghi khoảng thời gian bất định.
- Gán timeline trạng thái cho từng đèn điều khiển hướng đi, dùng chung cho các lượt xe.
- Test: các khoảng video liên tục được chọn, gán mọi lượt thuộc phạm vi; không chỉ gán ứng viên do mô hình tạo ra.
- Chia tập theo phiên/video nguồn; cùng camera có thể nằm ở cả phát triển và test.
- Một người gán, người còn lại kiểm tra toàn bộ test và các mẫu khó.
- Pilot trước để đo chi phí gán nhãn và chốt quy mô.
- Test chính theo các điều kiện thu thập đủ; điều kiện khác là test bổ sung, báo cáo riêng.
- Bổ sung nhãn dấu hiệu ngoại lệ quan sát được: người điều tiết và tín hiệu/biển phụ liên quan hướng đi. Đánh giá nhận biết dấu hiệu; không kết luận ngoại lệ pháp lý được áp dụng.
- Kết luận hành vi quan sát được và nhãn dấu hiệu ngoại lệ được lưu, đánh giá riêng. Không đổi hành vi thành âm tính chỉ vì thấy dấu hiệu ngoại lệ.
- Gán bounding box ở các frame mốc, liên kết bằng ID phương tiện chuẩn; không gán quỹ đạo đầy đủ mọi frame.
- VLM dùng clip quanh thời điểm ứng viên; thử độ dài trên tập phát triển. Frame sau sự kiện phải chờ đến trong streaming và thời gian chờ được tính vào latency.

## 2. Input, output và ground truth

| Thành phần | Input khi chạy | Output dự đoán | Ground truth để đánh giá |
| --- | --- | --- | --- |
| Toàn pipeline | Video liên tục, thời gian, cấu hình camera | Sự kiện nghi vượt đèn đỏ, đối tượng, thời gian, bằng chứng, giải thích | Các lượt tiếp cận được người kiểm tra độc lập |
| CV và quy tắc | Frame có thứ tự, cấu hình vạch/vùng/đèn | Track, trạng thái đèn, khoảng vượt vạch, ứng viên | ID phương tiện chuẩn, timeline đèn chuẩn, khoảng vượt vạch chuẩn |
| VLM | Bằng chứng có timestamp, đối tượng mục tiêu, dữ kiện CV dự đoán | Kết luận ba trạng thái, dữ kiện hỗ trợ, frame tham chiếu, giải thích, giới hạn, dấu hiệu ngoại lệ | Nhãn toàn video cho đánh giá pipeline; nhãn theo gói bằng chứng thực nhận để chẩn đoán VLM |

Ground truth không được truyền vào input mô hình trong đánh giá chính. Bản ghi nguồn video và cấu hình camera được phân biệt với nhãn kết quả. Cấu hình camera là input được phép biết trước; kết luận/thời gian/trạng thái đèn chuẩn là dữ liệu đánh giá.

Phân biệt `assessment` của lượt xe trên toàn video với `package_assessment` của gói bằng chứng thực sự đưa cho VLM. Nếu video chứng minh hành vi nhưng clip bị cắt mất thời điểm vượt vạch, nhãn toàn video vẫn dương tính; gói có thể là `insufficient_evidence`. Khi đó VLM từ chối có thể phù hợp với gói nhận được, nhưng pipeline vẫn bỏ sót sự kiện nếu không xuất cảnh báo dương tính. Dữ kiện CV dự đoán không được coi là bằng chứng chuẩn. Nhãn gói do người xem đúng clip/keyframe, cấu hình và dữ kiện đầu vào thực nhận kiểm chứng, không suy diễn từ kết luận VLM; lưu riêng sai lệch của CV và thiếu sót của gói.

## 3. Đơn vị mẫu và phạm vi

Một lượt bắt đầu khi phương tiện vào vùng tiếp cận. `approach_end_s` là thời điểm kết thúc quan sát lượt, tách khỏi thời điểm vượt vạch: tiếp tục xem phần sau vạch đủ để xác định hướng đi, rồi kết thúc khi xe rời vùng quan sát đã cấu hình. Các lý do kết thúc khác là hết đoạn được gán nhãn hoặc mất dấu không thể nối lại. ID chuẩn do người gán xác định, không dùng trực tiếp track ID dự đoán.

Xe dừng rồi đi tiếp trong cùng vùng vẫn là một lượt. Che khuất tạm thời không kết thúc lượt nếu có bằng chứng nối lại cùng xe; ghi khoảng che khuất. Nếu không thể nối chắc chắn, lưu các mảnh quan sát có `identity_uncertain`, không tự gộp hoặc coi chúng là hai lượt hoàn chỉnh độc lập. Các trường hợp chưa giải quyết được được báo cáo riêng, không đưa vào mẫu số sự kiện xác định. Xe thực sự rời vùng rồi quay lại tạo lượt mới; lần lùi qua vạch rồi vượt lại trong cùng lượt cần ghi riêng trong `crossing_observations` để không bỏ mất lần vượt lúc đỏ.

- `scope_status`: `in_scope`, `out_of_scope`, `scope_uncertain`.
- Rẽ và phương tiện ngoài hai nhóm mục tiêu: ngoài phạm vi, lưu lý do.
- Không xác định được hướng đi hoặc loại phương tiện: ghi phạm vi chưa rõ, không tự ép thành âm tính.
- Lượt còn dở ở đầu/cuối video: đánh dấu `boundary_truncated`; không tự kết luận không có hành vi.
- Test phải giữ các lượt khó/thiếu bằng chứng nằm trong video đã chọn.
- Xe tuân thủ và xe dừng trước vạch vẫn là mẫu đánh giá.
- Chỉ `in_scope` mới nhận một trong ba nhãn hành vi. `out_of_scope` hoặc `scope_uncertain` có `assessment = null` và lý do; thống kê riêng.

## 4. Ground truth ba trạng thái

Đây là nhãn hành vi theo quy ước kỹ thuật của dataset, không phải phán quyết pháp lý.

| Giá trị | Ý nghĩa |
| --- | --- |
| `red_light_crossing` | Có bằng chứng phần đầu xe vượt vạch khi đèn điều khiển hướng đi đã đỏ, theo quy ước quan sát đã chốt |
| `no_red_light_crossing` | Có đủ bằng chứng lượt xe không thực hiện hành vi mục tiêu trong khoảng đánh giá |
| `insufficient_evidence` | Không đủ bằng chứng về xe, vạch, đèn liên quan hoặc thứ tự thời gian để kết luận |

Không suy ra hành vi chỉ từ việc xe đang nằm trong giao lộ lúc đèn đỏ. Xe qua vạch trước đèn đỏ là tình huống âm tính nếu thứ tự được chứng minh. Nhóm chọn nhãn dấu hiệu ngoại lệ quan sát được, chỉ gồm người điều tiết và tín hiệu/biển phụ liên quan hướng đi. Không đánh giá việc ngoại lệ pháp lý thực sự được áp dụng. Kết luận hành vi quan sát được được giữ riêng; sự hiện diện của dấu hiệu không tự chuyển hành vi vượt vạch lúc đỏ thành âm tính. Nhãn `red_light_crossing` không đồng nghĩa với kết luận vi phạm pháp luật.

Lưu thời gian bằng giây theo trục thời gian video gốc, kèm frame ID/PTS khi có. Timestamp hệ thống xử lý được lưu riêng. Mỗi lần vượt liên kết với đúng `signal_id` và các khoảng trạng thái/chuyển màu của chu kỳ liên quan. Không so với một lần chuyển đỏ bất kỳ trong video.

Khoảng vượt vạch chồng lấn hoặc tiếp xúc khoảng chuyển màu mà không chứng minh được thứ tự: `insufficient_evidence`. Chỉ gán dương tính khi toàn bộ khoảng vượt nằm trong khoảng đã xác minh đỏ; chỉ gán âm tính cho lần vượt khi toàn bộ khoảng nằm trong trạng thái đã xác minh không đỏ. Khoảng đỏ cũng phải có điểm kết thúc được xét, không chỉ kiểm tra xe vượt sau lúc bắt đầu đỏ. Xe không vượt có thể nhận âm tính cho phần quan sát đầy đủ trước vạch; giữ cờ cắt cụt nếu chưa theo dõi hết lượt. Nếu có nhiều lần vượt, lưu từng lần và lấy dương tính khi có ít nhất một lần xác định đỏ; nếu không có dương tính nhưng có lần chưa rõ thì nhãn lượt là thiếu bằng chứng.

## 5. Schema logic

### sources

`source_id`, `source_url`, `country`, `usage_terms_reference`, `retrieved_at`, `original_video_id`, `camera_id`, `session_id`, `file_uri`, `checksum`, `duration_s`, `resolution`, `timebase`, `nominal_fps`, `split`.

Giữ quan hệ giữa clip và video gốc. Video tải lại/clip trùng nội dung phải cùng nhóm chia tập. FPS danh nghĩa không thay thế PTS nếu video có tốc độ khung hình biến đổi.

### camera_configs

`camera_id`, `config_version`, `reference_frame_id`, `coordinate_system`, `approach_roi`, `observation_roi`, `stop_line`, `travel_direction`, `signal_id`, `signal_roi`, `signal_to_approach_mapping`.

`observation_roi` gồm vùng tiếp cận và phần sau vạch cần xem để xác định hướng đi. Ranh giới vượt là mép vạch phía xe tiếp cận gặp trước; kiểm chứng khả năng quan sát phần đầu xe trong pilot, không thay bằng tâm bbox khi bị che.

### observation_segments

`segment_id`, `source_id`, `start_s`, `end_s`, `lighting`, `weather`, `visibility`, `image_quality`, `selection_reason`, `annotation_completeness`.

Gán điều kiện theo đoạn khi video thay đổi điều kiện. Cho phép `unknown`; không đoán thời tiết không nhìn rõ.

### signal_timelines

`signal_interval_id`, `source_id`, `signal_id`, `cycle_id`, `state`, `start_s`, `end_s`, `evidence_frame_ids`, `review_status`.

Mỗi hàng là khoảng trạng thái được xác minh, không chồng lấn các trạng thái khác của cùng đèn. Lưu chuyển màu ở `signal_transitions`: `transition_id`, `source_id`, `signal_id`, `from_interval_id`, `to_interval_id`, `transition_time_min`, `transition_time_max`, `evidence_frame_ids`. Khoảng chuyển màu chưa phân định là khoảng bất định, không đồng thời gán chắc chắn cho cả hai màu. Khoảng mất quan sát là `unknown`; không tự điền bằng nội suy.

Trạng thái gồm `red`, `yellow`, `green`, `unknown`, `off`, `other`; đèn mũi tên hoặc tín hiệu đặc biệt cần ghi loại và quan hệ hướng đi. Không gán đỏ từ một đèn không điều khiển phương tiện mục tiêu.

### approaches

`sample_id`, `source_id`, `camera_config_version`, `gt_vehicle_id`, `vehicle_type`, `approach_start_s`, `approach_end_s`, `scope_status`, `scope_reason`, `boundary_truncated`, `crossing_observed`, `crossing_time_min`, `crossing_time_max`, `signal_id`, `assessment`, `uncertainty_reasons`, `evidence_frame_ids`, `verified_facts`, `exception_annotations`, `annotator_id`, `reviewer_id`, `review_status`, `label_version`.

Bổ sung `segment_id`, `end_reason` (`left_observation_roi`, `segment_end`, `lost_unresolved`), `identity_uncertain`, `occlusion_intervals`, `crossing_observations`. Mỗi phần tử `crossing_observations` có `crossing_id`, `time_min`, `time_max`, `signal_interval_ids`, `transition_ids`, `assessment` và frame trước/sau. Các trường thời gian vượt ở cấp lượt chỉ là giá trị tóm tắt khi có đúng một lần vượt; để `null` nếu có nhiều lần và dùng danh sách làm nguồn chuẩn. Đánh giá sự kiện dùng `crossing_id`; đánh giá lượt dùng `sample_id`.

`crossing_observed`: `yes`, `no`, `unknown`. Không vượt vạch hoặc không quan sát được thời điểm: thời gian để `null`, ghi lý do. ID phương tiện chuẩn hỗ trợ ánh xạ dự đoán bằng bounding box ở các frame mốc: tiếp cận, trước/sau vượt vạch nếu có và frame bằng chứng. Lưu frame ID, timestamp, hệ tọa độ, bbox và mức che khuất; không dùng bbox mô hình chưa kiểm chứng làm nhãn chuẩn.

`exception_annotations`: gồm hai loại `traffic_controller` và `supplementary_signal_or_sign`. Mỗi loại lưu `observation_status` (`present`, `not_observed`, `unclear`), `relevance_to_target_direction` (`observed`, `unclear`, `not_applicable`), `evidence_frame_ids`, `observed_details` và `uncertainty_reason`. Cho phép đồng thời nhiều dấu hiệu. `not_observed` chỉ có nghĩa không thấy trong bằng chứng đã xem, không chứng minh không tồn tại ngoài khung hình. Không thêm trường xác nhận tính hợp pháp; không suy ra hiệu lệnh nếu không nhìn rõ. Dấu hiệu phương tiện đặc biệt và tránh chướng ngại vật chưa thuộc taxonomy pilot.

### vehicle_keyframes

`keyframe_id`, `sample_id`, `gt_vehicle_id`, `source_id`, `frame_id`, `timestamp_s`, `frame_role`, `bbox_xyxy`, `occlusion_status`, `review_status`.

BBox theo pixel ảnh gốc: `[x_min, y_min, x_max, y_max]`, gốc ở góc trên trái; lưu kích thước ảnh trong nguồn. Gán hộp bao phần nhìn thấy, không suy đoán phần bị che. `frame_role` có thể gồm tiếp cận, trước/sau vượt, xác định hướng đi và bằng chứng. Không yêu cầu nhãn mọi frame; ánh xạ xe được kiểm tra trên các frame mốc, không tự suy ra toàn bộ quỹ đạo chuẩn.

### evidence_packages và package_reviews

`evidence_packages`: `package_id`, `run_id`, `candidate_id`, `source_id`, `clip_start_s`, `clip_end_s`, `clip_uri`, `frame_ids`, `image_transforms`, `cv_payload_uri`, `camera_config_version`, `content_checksum`, `available_at`.

Lưu đúng crop, resize, thứ tự frame và dữ kiện CV mà VLM nhận để tái lập đầu vào. `package_reviews`: `package_id`, `matched_sample_id`, `evidence_sufficiency` (`sufficient`, `insufficient`, `unassessable`), `package_assessment`, `missing_evidence`, `cv_discrepancies`, `verified_facts`, `exception_annotations`, `reviewer_id`. Nhãn gói có thể khác nhãn toàn video; không ghi đè nhãn toàn video. `unassessable` dùng khi không xác định được đối tượng/phạm vi để chấm, `package_assessment = null`.

### predictions (tách khỏi nhãn chuẩn)

`prediction_id`, `run_id`, `method_id`, `candidate_id`, `package_id`, `event_id`, `source_id`, `camera_id`, `session_id`, `pred_track_id`, `pred_crossing_time`, `pred_signal_state`, `assessment`, `explanation`, `evidence_frame_ids`, `pred_vehicle_keyframes`, `exception_annotations`, `limitations`, `model_version`, `prompt_version`, `config_version`, `processing_timestamps`.

`pred_vehicle_keyframes` chứa frame ID, timestamp, bbox và hệ tọa độ có ánh xạ về ảnh gốc. `run_id` định danh phiên thử nghiệm; `method_id` phân biệt baseline/VLM. `candidate_id` liên kết cùng ứng viên giữa hai phương án; `package_id` có thể trống với baseline không dùng gói VLM. `exception_annotations` dùng cấu trúc của nhãn chuẩn nhưng chứa dự đoán. Ánh xạ dự đoán–nhãn chuẩn được lưu riêng trong kết quả đánh giá, không đưa ngược vào input.

Lưu kết quả baseline và VLM riêng; không ghi đè kết luận dự đoán bằng nhãn kiểm duyệt.

## 6. Chia dữ liệu và đánh giá

- Chia theo video/phiên thu thập trước khi chỉnh prompt/quy tắc. Mọi clip từ cùng nguồn gốc hoặc nội dung trùng nằm chung một tập.
- Video cùng phiên bị chia thành nhiều tệp vẫn phải cùng tập. Nếu không xác định được phiên, ghi hạn chế và nhóm bảo thủ theo nguồn.
- Test được giữ độc lập; chỉnh hệ thống dựa trên lỗi test thì phải ghi nhận và cần test độc lập khác cho đánh giá cuối.
- Ghép một-một dự đoán dương tính với lần vượt dương tính theo đúng xe, lượt và thời gian. Một nhãn nhận tối đa một dự đoán, một dự đoán nhận tối đa một nhãn. Lưu `run_id`, `prediction_id`, `sample_id`, `crossing_id`, trạng thái ghép và lý do.
- Dùng bbox ở frame mốc để kiểm tra đúng xe. Sai số thời gian là khoảng cách từ timestamp dự đoán tới khoảng thời gian chuẩn (bằng 0 nếu nằm trong khoảng). Ngưỡng ghép và cách phá hòa khi nhiều ứng viên đủ điều kiện được chốt trên tập phát triển trước test; ngưỡng không được dùng để đổi nhãn bất định thành nhãn chắc chắn.
- True positive (TP): cảnh báo dương tính ghép đúng. False negative (FN): lần vượt dương tính không có cảnh báo ghép đúng, kể cả không tạo ứng viên, VLM bác bỏ hoặc từ chối kết luận. False positive (FP): cảnh báo dương tính không ghép được trong phần dữ liệu đủ nhãn/đủ bằng chứng, gồm cảnh báo trên xe âm tính và cảnh báo trùng sau lần ghép đầu. Dự đoán đúng xe nhưng sai thời gian quá ngưỡng có thể tạo cả FP và FN.
- Tính precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN); mẫu số của chỉ số nào bằng 0 thì chỉ số đó báo N/A, kèm số đếm. Cảnh báo trùng vừa được tính FP vừa thống kê riêng. Chỉ loại bản ghi phát lại kỹ thuật có cùng định danh idempotent; không loại các event khác ID cùng cảnh báo một sự kiện.
- Ground truth `insufficient_evidence`, `scope_uncertain`, `out_of_scope` hoặc chưa giải quyết ID nằm ngoài mẫu số precision/recall chính; báo riêng số mẫu, số cảnh báo và lý do. Xác định nhóm loại trừ từ nhãn chuẩn, không từ kết luận mô hình. Cảnh báo chưa ghép phải được kiểm tra vị trí/đối tượng để phân biệt FP trên vùng đã gán đầy đủ với cảnh báo không thể chấm; không tự loại mọi cảnh báo chưa ghép.
- Đánh giá VLM và baseline trên cùng bộ ứng viên, đồng thời báo kết quả toàn pipeline dựa trên nhãn toàn video. Báo riêng chẩn đoán VLM dựa trên `package_assessment` và độ đầy đủ của gói để phân biệt lỗi trích xuất bằng chứng, dữ kiện CV sai và lỗi thẩm định.
- Báo tỷ lệ từ chối trên các kết quả được gọi thẩm định, tách theo nhãn thật và độ đầy đủ gói; việc không tạo ứng viên không phải một lần VLM từ chối. Báo ma trận ba trạng thái trên mẫu có kết luận chuẩn, cùng tỷ lệ bao phủ bộ ứng viên. Việc chỉ chấm trên các mẫu mô hình chịu kết luận không được thay thế recall toàn pipeline.
- Rubric giải thích: đúng đối tượng, đúng thứ tự thời gian, đúng đèn liên quan, dẫn bằng chứng phù hợp, không thêm dữ kiện thiếu căn cứ, nêu giới hạn khi cần. Thang điểm chưa chốt.
- Báo cáo riêng theo quốc gia, ánh sáng, thời tiết và chất lượng; không kết luận cho nhóm điều kiện có quá ít mẫu.

## 7. Quy trình pilot

1. Tuyển khoảng 2–3 video/phiên độc lập; lưu nguồn gốc, quyền sử dụng và cấu hình camera.
2. Chọn khoảng video liên tục; mục tiêu khảo sát khoảng 30–50 lượt tiếp cận, không thay đổi nhãn để đạt số lượng.
3. Gán timeline đèn, mọi lượt xe và các frame bằng chứng; ghi thời gian lao động thực tế.
4. Người thứ hai kiểm tra nhãn test và toàn bộ mẫu khó; lưu bất đồng và lý do sửa.
5. Kiểm tra độ rõ vạch/đèn, khoảng bất định, tính nhất quán ID, tình huống cắt cụt và các điều kiện quan sát.
6. Sửa hướng dẫn gán nhãn, chốt quy mô và ngưỡng đánh giá rồi mới gán chính thức.

Pilot chưa đại diện đầy đủ cho mọi điều kiện ngày/đêm/mưa. Test chính dựa trên những điều kiện có đủ dữ liệu; điều kiện còn lại thuộc test bổ sung. Tiêu chuẩn thế nào là đủ sẽ chốt sau pilot, không chỉ dựa vào tổng số frame. Video dùng để thử prompt hoặc chỉnh thuật toán thuộc tập phát triển.

## 8. Quyết định còn mở

- Kiểm tra tính nhất quán của nhãn dấu hiệu ngoại lệ trong pilot; taxonomy và việc tách khỏi kết luận hành vi đã chốt.
- Độ dài ngữ cảnh trước/sau sự kiện: thử trên tập phát triển rồi chốt trước test; đã thống nhất chờ frame tương lai và tính thời gian chờ vào latency.
- Ngưỡng ghép sự kiện, thang điểm rubric, quy mô và tỷ lệ chia tập sau pilot.
