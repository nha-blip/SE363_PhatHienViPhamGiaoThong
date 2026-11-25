import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/VLM363_working/external_compilation_15'
OUTPUT = BASE / 'gt_draft_revision_001'
CLASSES = ['truck', 'motorcycle', 'motorcycle', 'car', 'car',
           'motorcycle', 'motorcycle', None, 'car', 'motorcycle',
           'car', 'motorcycle', 'motorcycle', 'motorcycle', 'motorcycle']
SEEDS = {
    'case_01': [(3837, [302, 82, 59, 27], 'Xe trắng phía xa; cần đối chiếu danh tính qua replay.')],
    'case_02': [(4583, [89, 119, 72, 145], 'Người áo đen, xe máy phía trái.')],
    'case_03': [(5330, [214, 201, 61, 121], 'Người áo xanh, xe máy bên trái nhóm ba xe.')],
    'case_04': [(6158, [217, 115, 48, 48], 'SUV có bánh dự phòng sau; không phải minivan gần camera.')],
    'case_05': [(7104, [373, 204, 67, 60], 'Ô tô đỏ; cần kiểm tra đi thẳng hay rẽ.')],
    'case_06': [(8182, [265, 161, 58, 120], 'Người áo hồng, xe máy sát lề phải; cần kiểm tra hướng đi.')],
    'case_07': [(8896, [326, 84, 37, 79], 'Xe máy đã đi qua vùng bóng râm.')],
    'case_08': [(9701, [201, 113, 164, 199], 'Xe chở hàng nhỏ; class chưa phân xử giữa van/car/truck.')],
    'case_09': [(10240, [263, 95, 38, 36], 'Ô tô giữa làn phía xa; cần đối chiếu replay, không chọn xe đỗ bên phải.')],
    'case_10': [(11908, [85, 140, 70, 90], 'Ứng viên người áo đỏ/cam bên trái, bị xe phía sau che một phần.'),
                (11908, [440, 204, 87, 156], 'Ứng viên người áo vàng/cam, mũ đỏ; đáy ảnh và presenter che một phần.'),
                (11908, [598, 246, 42, 108], 'Ứng viên áo hồng, mũ trắng sát mép phải; bị cắt bởi khung hình.')],
    'case_11': [(12621, [286, 52, 145, 119], 'Ô tô tối màu sát lề phải; cần kiểm tra rẽ.')],
    'case_12': [(13497, [58, 212, 44, 96], 'Người áo đen trên xe máy bên trái.')],
    'case_13': [(13955, [80, 68, 74, 143], 'Người mũ đỏ/trắng, áo sáng trên xe máy bên trái.')],
    'case_14': [(14469, [75, 132, 55, 120], 'Xe máy có kiện hàng vàng phía sau, không phải người áo vàng ở đáy ảnh.')],
    'case_15': [(15119, [125, 182, 79, 158], 'Người áo đen, mũ bạc/trắng trên xe máy giữa ảnh.')],
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def evidence_indices(count):
    return sorted({round((count - 1) * fraction / 10) for fraction in range(1, 10)})


def build():
    manifest_path = BASE / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source = ROOT / manifest['source']
    if digest(source) != manifest['source_sha256']:
        raise ValueError('Source hash differs from split manifest')
    for case in manifest['cases']:
        if digest(ROOT / case['clip']) != case['clip_sha256']:
            raise ValueError(f"Clip hash mismatch: {case['case_id']}")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    cases = []
    reviews = []
    for case_index, case in enumerate(manifest['cases']):
        directory = OUTPUT / case['case_id']
        directory.mkdir()
        capture = cv2.VideoCapture(str(ROOT / case['clip']))
        evidence = []
        sheet = Image.new('RGB', (960, 609), 'black')
        painter = ImageDraw.Draw(sheet)
        try:
            for tile_index, clip_frame in enumerate(evidence_indices(case['decoded_frames'])):
                capture.set(cv2.CAP_PROP_POS_FRAMES, clip_frame)
                success, frame = capture.read()
                if not success or frame.shape[:2] != (360, 640):
                    raise ValueError(f"Cannot read {case['case_id']} frame {clip_frame}")
                source_frame = case['source_start_frame_zero_based'] + clip_frame
                path = directory / f'source_{source_frame:06d}.png'
                picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                picture.save(path)
                column, row = tile_index % 3, tile_index // 3
                sheet.paste(picture.resize((320, 180)), (column * 320, row * 203 + 23))
                painter.text((column * 320 + 4, row * 203 + 4),
                             f'source {source_frame} | clip {clip_frame} | {source_frame / 25:.2f}s', fill='white')
                evidence.append({'clip_frame_id': clip_frame, 'source_frame_id': source_frame,
                                 'source_nominal_seconds': source_frame / 25,
                                 'image': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)})
        finally:
            capture.release()
        sheet.save(directory / 'timeline.jpg')
        target_count = 3 if case['case_id'] == 'case_10' else 1
        targets = []
        for target_number in range(1, target_count + 1):
            target_id = f"{case['case_id']}_target_{target_number:02d}"
            targets.append({'target_id': target_id, 'identity_confirmed': False,
                            'class': None, 'class_candidate': CLASSES[case_index],
                            'bbox_keyframes': [], 'motion_direction': None,
                            'scope_status': None, 'stop_line': None, 'signal_roi': None,
                            'signal_controls_target': None, 'signal_timeline': [],
                            'crossing_status': None, 'crossing_frame_before': None,
                            'crossing_frame_after': None, 'signal_state_at_crossing': None,
                            'assessment': None, 'human_corrected': False,
                            'annotation_status': 'PENDING_GEOMETRY_AND_EVENT_REVIEW'})
            reviews.append({'case_id': case['case_id'], 'target_id': target_id,
                            'decision': 'UNREVIEWED', 'target_identity_confirmed': False,
                            'bbox_confirmed': False, 'final_class': None,
                            'final_motion_direction': None, 'final_scope_status': None,
                            'signal_mapping_confirmed': False, 'stop_line_confirmed': False,
                            'crossing_status': None, 'crossing_frame_before': None,
                            'crossing_frame_after': None, 'signal_state_at_crossing': None,
                            'final_assessment': None, 'notes': ''})
        cases.append({'case_id': case['case_id'], 'description': case['description'],
                      'source': manifest['source'], 'source_sha256': manifest['source_sha256'],
                      'source_frame_range_half_open': [case['source_start_frame_zero_based'],
                                                       case['source_end_frame_exclusive']],
                      'clip': case['clip'], 'clip_sha256': case['clip_sha256'],
                      'sequence_id': 'external_compilation_15', 'content_group_id': 'external_compilation_15',
                      'frame_id_base': 0, 'fps': 25, 'image_size': [640, 360],
                      'coordinate_system': 'native_source_pixels',
                      'edited_footage': True, 'continuous_event_timebase_verified': False,
                      'evidence_frames': evidence, 'targets': targets})
    document = {'schema_version': '1.0', 'status': 'DRAFT_NOT_GROUND_TRUTH',
                'is_ground_truth': False, 'case_count': len(cases),
                'candidate_target_count': len(reviews), 'cases': cases}
    write_json(OUTPUT / 'annotations.draft.json', document)
    write_json(OUTPUT / 'human_event_review.json',
               {'status': 'UNREVIEWED', 'is_ground_truth': False,
                'annotation_sha256': digest(OUTPUT / 'annotations.draft.json'), 'reviews': reviews})
    if digest(source) != manifest['source_sha256']:
        raise ValueError('Source changed during preparation')
    write_json(OUTPUT / 'provenance.json',
               {'source': manifest['source'], 'source_sha256': manifest['source_sha256'],
                'raw_source_modified': False, 'split_manifest_sha256': digest(manifest_path),
                'script_sha256': digest(Path(__file__)), 'no_model_run': True,
                'no_new_web_pages': True, 'licensing_verified': False,
                'input_leakage_risks': ['news captions', 'presenter', 'signal inset', 'zoom and replay'],
                'allowed_split_unit': 'whole source content group; no random split between these clips'})
    write_json(OUTPUT / 'qa_report.json',
               {'status': 'REVIEW_REQUIRED', 'case_count': len(cases),
                'candidate_target_count': len(reviews), 'evidence_image_count': sum(len(case['evidence_frames']) for case in cases),
                'confirmed_event_count': 0, 'working_gt_ready': False,
                'source_files_modified': False,
                'remaining': ['identify each target', 'bbox and geometry', 'straight/turning scope',
                              'signal timeline and crossing interval', 'human event approval',
                              'edited footage and leakage suitability review']})
    print(f'{len(cases)} cases; {len(reviews)} candidate targets; REVIEW_REQUIRED: {OUTPUT}')


def seed_targets():
    annotations_path = OUTPUT / 'annotations.draft.json'
    document = json.loads(annotations_path.read_text(encoding='utf-8'))
    if any(target['bbox_keyframes'] for case in document['cases'] for target in case['targets']):
        raise ValueError('Seeds already present; create a new revision for further edits')
    overview = Image.new('RGB', (960, 1218), 'black')
    overview_draw = ImageDraw.Draw(overview)
    target_index = 0
    for case in document['cases']:
        for target, (source_frame, bbox, note) in zip(case['targets'], SEEDS[case['case_id']]):
            evidence = next(frame for frame in case['evidence_frames'] if frame['source_frame_id'] == source_frame)
            image_path = ROOT / evidence['image']
            target['bbox_keyframes'] = [{'source_frame_id': source_frame, 'bbox_xywh': bbox,
                                        'coordinate_system': 'native_source_pixels',
                                        'annotation_method': 'manual_visual_seed', 'human_confirmed': False,
                                        'image_sha256': evidence['sha256']}]
            target['identity_note'] = note
            target['annotation_status'] = 'MANUAL_BBOX_DRAFT_PENDING_EVENT_QA'
            target['bbox_policy'] = 'visible extent; motorcycle includes rider; no interpolation across zoom/replay'
            picture = Image.open(image_path).convert('RGB')
            painter = ImageDraw.Draw(picture)
            left, top, width, height = bbox
            painter.rectangle((left, top, left + width, top + height), outline='lime', width=3)
            painter.text((8, 8), f"{target['target_id']} | source {source_frame} | DRAFT", fill='yellow')
            path = OUTPUT / case['case_id'] / f"{target['target_id']}_bbox.png"
            picture.save(path)
            target['qa_bbox_image'] = path.relative_to(ROOT).as_posix()
            column, row = target_index % 3, target_index // 3
            overview.paste(picture.resize((320, 180)), (column * 320, row * 203 + 23))
            overview_draw.text((column * 320 + 3, row * 203 + 4), target['target_id'], fill='white')
            target_index += 1
    overview.save(OUTPUT / 'target_overview.jpg')
    write_json(annotations_path, document)
    review_path = OUTPUT / 'human_event_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    review['annotation_sha256'] = digest(annotations_path)
    write_json(review_path, review)
    report_path = OUTPUT / 'qa_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report.update(manual_bbox_seed_count=target_index, confirmed_bbox_count=0,
                  pending_geometry_count=target_index, annotation_sha256=digest(annotations_path))
    write_json(report_path, report)
    provenance_path = OUTPUT / 'provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    provenance['manual_seed_script_sha256'] = digest(Path(__file__))
    provenance['assessment_policy'] = 'News title/captions are not event ground truth; no automatic positive labels'
    write_json(provenance_path, provenance)
    print(f'{target_index} manual bbox seeds; event GT remains REVIEW_REQUIRED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local draft annotations, never auto-approved GT')
    parser.add_argument('command', choices=['build', 'seed-targets'])
    arguments = parser.parse_args()
    if arguments.command == 'build':
        build()
    else:
        seed_targets()
