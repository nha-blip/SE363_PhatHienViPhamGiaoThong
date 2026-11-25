import copy
import hashlib
import json
from pathlib import Path

if __package__:
    from .review_light_dataset import correction_focus
else:
    from review_light_dataset import correction_focus


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'data/traffic_light_external_audit_20261022/revision_001_20261026/correction_review.json'
OUTPUT = PREVIOUS.parent.parent / 'revision_002_bbox_edit_20261026'
TASKS = (
    ('S2TLD_1080x1920_0093f71333d0b11125d4', 2, 'geometry', 'BBox #2 quá lớn. Bấm Vẽ lại bbox, kéo khung ôm đúng cả cụm đèn.'),
    ('S2TLD_1080x1920_00f03a0b805c0e156b23', 1, 'geometry', 'BBox #1 quá nhỏ. Bấm Vẽ lại bbox, kéo khung bao đủ cả cụm đèn.'),
    ('S2TLD_1080x1920_be4e195042af1e050f0a', 2, 'color', 'Đã đổi bbox #2 từ off sang red theo ghi chú của bạn. Chỉ cần kiểm tra và xác nhận.'),
)


def main():
    content = PREVIOUS.read_bytes()
    previous = json.loads(content)
    lookup = {row['id']: row for row in previous['reviews']}
    rows = []
    for identity, number, task, hint in TASKS:
        old = lookup[identity]
        objects = copy.deepcopy(old['objects'])
        before = copy.deepcopy(objects[number - 1])
        if task == 'color':
            if before['source_class'] != 'off':
                raise ValueError('Source class changed; reconcile before continuing')
            objects[number - 1]['source_class'] = 'red'
            objects[number - 1]['proposed_class'] = 'red'
        rows.append({'id': identity, 'source_image': old['source_image'],
                     'qa_image': str(OUTPUT / 'initial_previews' / (identity + '.jpg')),
                     'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED', 'class_mapping_confirmed': False,
                     'notes': '', 'previous_review_notes': old['notes'], 'task_hint': hint,
                     'objects': objects, 'geometry_required': task == 'geometry',
                     'edit_object_numbers': [number] if task == 'geometry' else [], 'geometry_edits': [],
                     'correction': {'action': {'action': 'DRAW_BBOX_REQUIRED' if task == 'geometry' else 'CHANGE_CLASS', 'object_number': number},
                                    'before': before, 'after': copy.deepcopy(objects[number - 1])}})
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'initial_previews').mkdir()
    (OUTPUT / 'previous_review.snapshot.json').write_bytes(content)
    for row in rows:
        correction_focus(row).save(row['qa_image'], quality=98)
    data = {'status': 'PENDING_HUMAN_QA', 'is_ground_truth': False, 'allow_bbox_edit': True,
            'previous_review_sha256': hashlib.sha256(content).hexdigest(),
            'coordinate_system': 'native_source_pixels', 'current_dataset_modified': False, 'reviews': rows}
    (OUTPUT / 'bbox_edit_review.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    if PREVIOUS.read_bytes() != content:
        raise RuntimeError('Previous review changed during preparation')
    print(f'Prepared {len(rows)} cases; 2 need manual geometry, 1 color corrected; previous review unchanged')


if __name__ == '__main__':
    main()
