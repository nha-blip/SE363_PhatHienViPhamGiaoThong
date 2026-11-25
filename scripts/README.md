# Script organization — Project 363

Ngày kiểm tra: 09/10/2026.

## Quyết định

- Chuyển 43 script Python ở gốc vào `scripts/`; giữ 6 script Python đã có tại đây.
- Không xóa script nghiệp vụ: không được gọi thường xuyên không có nghĩa là không cần. Audit, builder, sửa GT và QA là các bước tái lập/provenance; một số bước chỉ chạy một lần nhưng vẫn phải giữ.
- `build_vehicle363_v1.py` là entrypoint mỏng gọi `finish_vehicle363_v1.py`, không phải bản sao nội dung cần xóa.
- Giữ thư viện trong `src/tracking/` và kiểm thử trong `tests/`.
- Cập nhật đường dẫn gốc, import nội bộ và tài liệu lệnh. Không sửa lịch sử trong `runs/`, dataset/manifest hoặc notebook outputs; tên script cũ trong báo cáo nay nằm dưới `scripts/`.
- Không chạy train, builder hoặc cleanup dữ liệu để thử script. Không dừng tiến trình khác, không xóa weights hoặc nguồn.

## Cách chạy

Từ `D:\se363-bigdata`:

```powershell
python scripts/survey_vlm_video_sources.py --report
python scripts/run_tracking_pilot.py --help
python scripts/resume_vehicle_local.py --help
python -m scripts.run_tracking_pilot --help
python -m pytest tests/tracking tests/training -q -p no:cacheprovider
```

`--report` chỉ tổng hợp artifact khảo sát đã có. Resume không có `--help`/`--check-only` sẽ khởi động train. Các script build/apply/cleanup và một số QA không có dry-run: không chạy để thử import hoặc `--help` khi chưa kiểm tra parser. Không chạy lại builder trên frozen release. Một số bước cũ cần payload frozen đã được chủ dự án xóa: giữ script không đồng nghĩa chạy lại được ngay.

## Script chuyển từ gốc

| File hiện tại | Vai trò / lý do giữ |
| --- | --- |
| `scripts/analyze_tracking_id_candidates.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/apply_track68_derived_resolution.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/apply_vehicle363_duplicate_policy.py` | Cleanup/QA và quyết định duplicate; giữ để tái lập |
| `scripts/audit_duplicate_resolution.py` | Audit/kiểm chứng/provenance |
| `scripts/audit_hcmc_class_semantics.py` | Audit/kiểm chứng/provenance |
| `scripts/audit_mio_mapping.py` | Audit/kiểm chứng/provenance |
| `scripts/audit_tracking_sources.py` | Audit/kiểm chứng/provenance |
| `scripts/audit_vehicle363_cleanup.py` | Audit/kiểm chứng/provenance |
| `scripts/audit_vehicle_datasets.py` | Audit/kiểm chứng/provenance |
| `scripts/build_tracking_temporal_pilot.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/build_vehicle363_v1.py` | Builder/release; giữ để tái lập |
| `scripts/build_vntraffic_working_gt.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/clean_bellevue_draft_boxes.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/create_tracking_conflict_qa.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/create_tracking_neighbor_qa.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/dryrun_final_duplicate_resolution.py` | Cleanup/QA và quyết định duplicate; giữ để tái lập |
| `scripts/expand_tracking_pilot_sources.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/finalize_tracking_source_audit.py` | Audit/kiểm chứng/provenance |
| `scripts/finish_vehicle363_v1.py` | Builder/release; giữ để tái lập |
| `scripts/freeze_vehicle363_v1.py` | Builder/release; giữ để tái lập |
| `scripts/kaggle_training_config.py` | Cấu hình và huấn luyện; giữ quy trình tiếp tục |
| `scripts/kaggle_train_drive.py` | Cấu hình và huấn luyện; giữ quy trình tiếp tục |
| `scripts/normalize_tracking_human_review.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/prepare_aicc_tracking_event_qa.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/prepare_ne12th_initial_qa.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/prepare_track68_bbox_proposal.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/prepare_tracking_class_qa.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/prepare_tracking_integration.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/prepare_vehicle363_final_qa.py` | Cleanup/QA và quyết định duplicate; giữ để tái lập |
| `scripts/propose_ne12th_initial_boxes.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/render_remaining_tracking_qa.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/resolve_ne12th_duplicate_ids.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/resume_vehicle_local.py` | Cấu hình và huấn luyện; giữ quy trình tiếp tục |
| `scripts/run_expanded_tracking_review.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/run_tracking_experiments.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/run_tracking_pilot.py` | Tracking/GT derived/QA; giữ dù tạm dừng |
| `scripts/scan_project_cleanup_review.py` | Audit/kiểm chứng/provenance |
| `scripts/simulate_duplicate_conflict_drop.py` | Cleanup/QA và quyết định duplicate; giữ để tái lập |
| `scripts/survey_vlm_video_sources.py` | Khảo sát video VLM; đang dùng |
| `scripts/verify_frozen_payload_cleanup.py` | Audit/kiểm chứng/provenance |
| `scripts/verify_hcmc_daynight.py` | Audit/kiểm chứng/provenance |
| `scripts/verify_release_zip_deletion.py` | Audit/kiểm chứng/provenance |
| `scripts/verify_vehicle363_cleanup_apply.py` | Audit/kiểm chứng/provenance |

## Script vốn có trong scripts

- `audit_kaggle_traffic_lights.py`: audit nguồn đèn Kaggle.
- `convert_bosch_to_yolo.py`, `create_bosch_sequence_split.py`, `qa_bosch_dataset.py`: chuyển đổi/chia/QA Bosch.
- `build_kaggle_tl_yolo_v1.py`, `build_trafficlight363_v1.py`: builder nguồn và release đèn.

Tên file giữ nguyên để dễ tra provenance; vị trí dataset/config/output không đổi.

## Gói YOLO server

- `python -m scripts.package_yolo_server`: kiểm tra pairing/số lượng split và định dạng nhãn working, đóng gói xe/đèn thành hai ZIP64, kèm config generator local, SHA256SUMS và manifest. Không chỉnh annotation hoặc chạy train. Từ chối ghi đè export có sẵn.
- `config/server/configure_server.py`: sau giải nén trên server, sinh hai YAML dataset và bốn YAML train nano/small theo vị trí project thật.

## Bổ sung annotation VLM

- `prepare_external_vlm_gt.py build` / `seed-targets`: tạo ảnh evidence local và bbox seed thủ công cho 15 case bản tin; giữ nhãn sự kiện null/chờ QA, không tự duyệt GT. Revision đã tạo không chạy build lại.

- `split_traffic_compilation.py`: khảo sát chuyển cảnh và tách bản tin thành 15 phần nội dung có âm thanh bằng FFmpeg local; lưu manifest frame/hash, không tự gán GT. Kết quả tại `data/VLM363_working/external_compilation_15/`.

- QA hiện dùng file local, không tạo thêm web: xem `data/VLM363_working/QA_WORKFLOW.md`.
- `prepare_vlm_local_vehicle_review.py survey`: tạo timeline lấy mẫu 1 giây; `build-cases`: tạo ảnh bbox tham chiếu và clip từng ứng viên từ seed thủ công. Không chạy tracker/model, không tự duyệt GT.
- Batch 002: `data/VLM363_working/expansion_queue/batch_002/vehicle_review_revision_001/README_QA.md`; file nhập QA là `human_vehicle_review.json` cùng thư mục.

- `build_vlm_annotation_batch.py`: tạo lô đề xuất nhãn thủ công và clip/keyframe derived, không chạy model hoặc tự duyệt GT.
- `vlm_annotation_review.html`: template trang duyệt; export quyết định JSON, không tự ghi đè annotation.
- Lô đầu: `runs/project363/vlm/annotation_batch_001/index.html`; nhãn tại `data/VLM363_working/annotation_batches/batch_001/`.
- Builder từ chối ghi đè lô đã tạo; bản sửa phải dùng revision mới, không chạy lại `--execute` trên lô hiện tại.

## Kết quả kiểm chứng tổ chức script

- 44 tests tracking/training đạt; thư mục tạm kiểm thử nằm trong workspace.
- 68 file Python scripts/src/tests qua kiểm tra cú pháp; 7 module đại diện qua kiểm tra import và ROOT.
- Entry point chính chạy được `--help` dạng đường dẫn trực tiếp và `python -m` mà không khởi động model.
- Manifest tổ chức: `runs/project363/code_cleanup/script_organization_report.json`.
- Manifest frozen lịch sử không sửa. Script hiện tại đã đổi đường dẫn/import nên hash có thể khác bản ghi release; audit vẫn báo khác hash, không bỏ qua kiểm chứng đó.

