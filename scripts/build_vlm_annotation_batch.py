import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / 'data/VLM363_working/annotation_batches/batch_001'
QA = ROOT / 'runs/project363/vlm/annotation_batch_001'
LABELS = {'red_light_crossing', 'no_red_light_crossing', 'insufficient_evidence'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def validate_proposal(sample, metadata):
    width, height = metadata['width'], metadata['height']
    fps, count = metadata['nominal_fps'], metadata['reported_frame_count']
    if sample['scope_status'] not in {'in_scope', 'out_of_scope', 'scope_uncertain'}:
        raise ValueError('Invalid scope status')
    proposed = sample['package_assessment_proposed']
    if sample['scope_status'] != 'in_scope' and proposed is not None:
        raise ValueError('Out-of-scope cases must not have an assessment')
    if proposed is not None and proposed not in LABELS:
        raise ValueError('Invalid assessment')
    start, end = sample['clip']
    if not 0 <= start < end <= count / fps + 1e-6:
        raise ValueError('Invalid source clip range')
    seen = set()
    for frame_id, box in sample['frames']:
        if frame_id in seen or not 0 <= frame_id < count:
            raise ValueError('Invalid or repeated frame')
        if not start <= frame_id / fps < end:
            raise ValueError('Evidence frame outside package')
        if len(box) != 4 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
            raise ValueError('Invalid coordinates')
        left, top, right, bottom = box
        if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
            raise ValueError('Invalid bounding box')
        seen.add(frame_id)


def build():
    proposal_path = AREA / 'manual_proposals.json'
    proposals = json.loads(proposal_path.read_text(encoding='utf-8'))
    source_manifest = ROOT / 'runs/project363/vlm/source_survey/source_manifest.json'
    source_records = json.loads(source_manifest.read_text(encoding='utf-8'))
    sources = {row['survey_id']: row for row in source_records}
    samples = proposals['samples']
    if len({sample['sample_id'] for sample in samples}) != len(samples):
        raise ValueError('Duplicate sample IDs')
    if QA.exists() or (AREA / 'annotations').exists():
        raise FileExistsError('Batch outputs already exist; use a new revision instead of overwriting')
    selected = {sample['survey_id'] for sample in samples}
    hashes = {}
    for survey_id in sorted(selected):
        source = sources[survey_id]
        hashes[survey_id] = digest(ROOT / source['source'])
        if hashes[survey_id] != source['sha256_before']:
            raise ValueError(f'Source changed since survey: {survey_id}')
    for sample in samples:
        validate_proposal(sample, sources[sample['survey_id']]['decoder'])
    QA.mkdir(parents=True, exist_ok=False)
    frames_directory = AREA / 'keyframes'
    clips_directory = AREA / 'clips'
    frames_directory.mkdir(exist_ok=False)
    clips_directory.mkdir(exist_ok=False)
    packages, keyframes, clip_cache, signal_observations = [], [], {}, []
    for sample in samples:
        source = sources[sample['survey_id']]
        metadata = source['decoder']
        fps = metadata['nominal_fps']
        record = {**sample, 'package_id': sample['sample_id'] + '_package',
                  'target_id': sample['sample_id'] + '_target',
                  'source': source['source'], 'source_sha256': hashes[sample['survey_id']],
                  'source_width': metadata['width'], 'source_height': metadata['height'],
                  'nominal_fps': fps, 'assessment': None, 'review_status': 'UNREVIEWED_EVENT_DRAFT',
                  'annotator_id': 'assistant_manual_visual', 'reviewer_id': None,
                  'legal_assessment': 'NOT_EVALUATED', 'is_ground_truth': False,
                  'full_video_assessment': None, 'timebase_verified': False,
                  'crossing_observations': [], 'crossing_time_min': None, 'crossing_time_max': None,
                  'boundary_truncated': True, 'camera_config_status': 'PENDING_GEOMETRY_AND_TARGET_MAPPING',
                  'signal_id': 'canmore_facing_camera_signal' if sample['sample_id'] == 'vlm_001' else None,
                  'evidence': []}
        capture = cv2.VideoCapture(str(ROOT / source['source']))
        for frame_id, box in sample['frames']:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            decoded, frame = capture.read()
            if not decoded:
                raise RuntimeError(f'Failed evidence frame {sample["sample_id"]}/{frame_id}')
            actual = round(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if actual != frame_id:
                raise RuntimeError('Decoder seek returned unexpected frame')
            path = frames_directory / f'{sample["sample_id"]}_{frame_id:08d}.jpg'
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path, quality=95)
            evidence = {'frame_id': frame_id, 'timestamp_s_nominal': frame_id / fps,
                        'decoder_timestamp_ms': capture.get(cv2.CAP_PROP_POS_MSEC),
                        'bbox_xyxy': box, 'coordinate_system': 'native_source_pixels',
                        'image': path.relative_to(ROOT).as_posix(), 'image_sha256': digest(path),
                        'annotation_method': 'assistant_manual_box', 'review_status': 'UNREVIEWED'}
            record['evidence'].append(evidence)
            keyframes.append({'sample_id': sample['sample_id'], 'target_id': record['target_id'], **evidence})
        start = round(sample['clip'][0] * fps)
        end = min(metadata['reported_frame_count'], round(sample['clip'][1] * fps))
        clip_key = (sample['survey_id'], start, end)
        if clip_key not in clip_cache:
            clip_path = clips_directory / f'{sample["survey_id"]}_{start:08d}_{end:08d}.mp4'
            preview_width = min(metadata['width'], 960)
            preview_height = int(metadata['height'] * preview_width / metadata['width']) // 2 * 2
            writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                    (preview_width, preview_height))
            if not writer.isOpened():
                raise RuntimeError('Could not create derived QA clip')
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            mapping = []
            for frame_id in range(start, end):
                decoded, frame = capture.read()
                if not decoded:
                    raise RuntimeError(f'Decode failed at {frame_id}')
                writer.write(cv2.resize(frame, (preview_width, preview_height)))
                mapping.append({'source_frame_id': frame_id, 'source_s_nominal': frame_id / fps,
                                'derived_frame_id': frame_id - start,
                                'decoder_timestamp_ms': capture.get(cv2.CAP_PROP_POS_MSEC)})
            writer.release()
            verifier = cv2.VideoCapture(str(clip_path))
            count = 0
            while True:
                decoded, frame = verifier.read()
                if not decoded:
                    break
                count += 1
            verifier.release()
            if count != end - start:
                raise RuntimeError('Derived clip frame-count mismatch')
            mapping_path = clip_path.with_suffix('.frame_map.json')
            write_json(mapping_path, mapping)
            clip_cache[clip_key] = {'clip_uri': clip_path.relative_to(ROOT).as_posix(),
                                   'clip_sha256': digest(clip_path), 'start_frame_id': start,
                                   'end_frame_exclusive': end, 'decoded_frames': count,
                                   'frame_map': mapping_path.relative_to(ROOT).as_posix(),
                                   'codec': 'mp4v', 'width': preview_width, 'height': preview_height}
        record['media'] = clip_cache[clip_key]
        capture.release()
        for frame_id, state in sample.get('signal_observations', []):
            signal_observations.append({'sample_id': sample['sample_id'], 'signal_id': record['signal_id'],
                                        'frame_id': frame_id, 'timestamp_s_nominal': frame_id / fps,
                                        'state_observed': state, 'review_status': 'UNREVIEWED_SNAPSHOT',
                                        'not_a_continuous_interval': True,
                                        'roi_xyxy': [675, 338, 706, 389]})
        packages.append(record)
    for survey_id in selected:
        if digest(ROOT / sources[survey_id]['source']) != hashes[survey_id]:
            raise RuntimeError('Source hash changed during annotation build')
    write_jsonl(AREA / 'annotations/evidence_packages.draft.jsonl', packages)
    write_jsonl(AREA / 'annotations/vehicle_keyframes.draft.jsonl', keyframes)
    write_jsonl(AREA / 'annotations/signal_observations.draft.jsonl', signal_observations)
    write_json(AREA / 'metadata/source_provenance.json', {
        'source_manifest_sha256': digest(source_manifest), 'manual_proposals_sha256': digest(proposal_path),
        'sources': [sources[survey_id] for survey_id in sorted(selected)],
        'source_hashes_verified_before_after': hashes, 'source_files_modified': False})
    write_json(AREA / 'metadata/camera_configs.draft.json', [
        {'survey_id': survey_id, 'stop_line': None, 'signal_to_approach_mapping': None,
         'review_status': 'PENDING_TARGET_SPECIFIC_GEOMETRY', 'source_quality_approved': True,
         'source_quality_approval_is_not_geometry_approval': True}
        for survey_id in sorted(selected)])
    write_json(AREA / 'qa/human_review_template.json', {
        'batch_id': 'batch_001', 'is_ground_truth': False,
        'reviews': [{'sample_id': row['sample_id'], 'decision': 'UNREVIEWED',
                     'final_scope_status': None, 'final_package_assessment': None,
                     'target_bbox_confirmed': False, 'signal_mapping_confirmed': False,
                     'stop_line_confirmed': False, 'notes': ''} for row in packages]})
    payload = {'batch_id': 'batch_001', 'draft_sha256': digest(proposal_path), 'samples': packages}
    write_json(QA / 'annotation_payload.json', payload)
    template = (ROOT / 'scripts/vlm_annotation_review.html').read_text(encoding='utf-8')
    page_payload = json.loads(json.dumps(payload))
    for row in page_payload['samples']:
        row['media']['url'] = Path(os.path.relpath(ROOT / row['media']['clip_uri'], QA)).as_posix()
        for evidence in row['evidence']:
            evidence['url'] = Path(os.path.relpath(ROOT / evidence['image'], QA)).as_posix()
    (QA / 'index.html').write_text(template.replace('__ANNOTATION_PAYLOAD__',
        json.dumps(page_payload, ensure_ascii=False).replace('</', '<\\/')), encoding='utf-8')
    counts = Counter(row['package_assessment_proposed'] or 'out_of_scope' for row in packages)
    report = {'status': 'DRAFT_REVIEW_REQUIRED', 'sample_count': len(packages),
              'source_count': len(selected), 'keyframe_count': len(keyframes),
              'distinct_derived_clips': len(clip_cache), 'proposed_label_counts': dict(counts),
              'approved_event_labels': 0, 'is_ground_truth': False, 'is_training_ready': False,
              'annotation_completeness': 'PARTIAL_SELECTED_CASES_ONLY',
              'source_files_modified': False, 'models_run': False,
              'invalid_bbox_count': 0, 'duplicate_sample_frame_keys': 0,
              'all_source_and_derived_clip_checks_passed': True}
    write_json(QA / 'annotation_report.json', report)
    lines = ['# VLM363 — Annotation batch 001', '', 'Ngày: 09/10/2026. Trạng thái: DRAFT_REVIEW_REQUIRED.', '',
             f'{len(packages)} trường hợp, {len(selected)} nguồn, {len(keyframes)} bbox mốc thủ công, {len(clip_cache)} clip derived.',
             '', 'Nhãn đề xuất theo gói bằng chứng, KHÔNG phải GT toàn video. Chưa duyệt nhãn sự kiện, chưa dùng train/test.',
             'Không chạy YOLO/VLM/ByteTrack; không sửa video hoặc GT nguồn. ID target là tham chiếu cục bộ của mẫu VLM, không thay ID tracking.',
             '', '## Nhãn đề xuất', '', '| Mẫu | Nội dung | Nhãn gói đề xuất |', '| --- | --- | --- |']
    lines.extend(f'| {row["sample_id"]} | {row["title"]} | {row["package_assessment_proposed"] or "out_of_scope / assessment=null"} |' for row in packages)
    lines.extend(['', '## Phạm vi và hạn chế', '',
                  '- Đây là lô khởi đầu gồm trường hợp được chọn; chưa gán mọi lượt trong các đoạn liên tục hoặc 25 video. Không dùng để báo recall pipeline.',
                  '- Camera geometry còn pending: cần xác nhận vạch và đèn điều khiển đúng xe. Các mẫu chiều ngược lại không bị ép âm/dương từ đèn đối diện.',
                  '- SUV chờ rồi đi sau xanh có đề xuất no_red_light_crossing; vẫn cần kiểm chứng vạch/ánh xạ, không phải nhãn đã duyệt.',
                  '- Mẫu bán tải đề xuất truck/out_of_scope theo phạm vi car/motorcycle; có thể sửa taxonomy khi QA.',
                  '- Frame nguồn 0-based; giây danh nghĩa = frame/FPS; chưa kiểm chứng PTS/timebase. Không tự gán timestamp vượt vạch chính xác hoặc nội suy đèn.',
                  '- Chưa gán nhãn đêm và không dùng ba file checksum lỗi. Nguồn quốc tế báo riêng, không áp luật Việt Nam.',
                  '- Xác nhận trước của chủ dự án là chất lượng nguồn, không tự duyệt nhãn/bbox các mẫu này.',
                  '', '## Xem và duyệt', '',
                  'Mở index.html. Bbox xanh lá là target do assistant gán thủ công, không phải YOLO. Chọn từng frame, xem clip và nhãn đề xuất.',
                  'Nhập quyết định, nhãn gói, kiểm tra bbox/đèn/vạch và ghi chú. Export human_review.json rồi đưa file export vào dự án để kiểm tra/apply ở bước riêng.',
                  'Browser lưu nháp cục bộ nhưng không tự ghi file trên đĩa. Xuất/duyệt không tự biến batch thành GT hay thay đổi nguồn.',
                  'MP4 codec mp4v có thể không chạy trong một số browser; dùng liên kết mở/tải và trình phát local.',
                  '', 'Nhãn tại data/VLM363_working/annotation_batches/batch_001/annotations/. Không tạo split và không dùng bản nháp làm đáp án test.'])
    (QA / 'annotation_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Build bounded manual VLM label proposals; never promote to ground truth')
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if not arguments.execute:
        parser.error('Use --execute to create the first draft batch; no model is run')
    cv2.setNumThreads(1)
    build()


if __name__ == '__main__':
    main()
