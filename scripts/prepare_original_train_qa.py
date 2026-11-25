import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/project363/dataset_improvement_20261020/human_label_review.json'
OUTPUT = ROOT / 'runs/project363/original_train_label_qa_20261026'
CLASS_NAMES = {'Vehicle363-v1': ['car', 'motorcycle', 'bus', 'truck'],
               'TrafficLight363-v1': ['red', 'yellow', 'green', 'off']}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    content = SOURCE.read_bytes()
    source = json.loads(content)
    if any(row['decision'] != 'PENDING' for row in source['reviews']):
        raise ValueError('Legacy review has decisions; reconcile them before creating a new UI copy')
    batches = {'vehicle': [], 'light': []}
    checksums = {}
    for old in source['reviews']:
        if old['split'] != 'train':
            raise ValueError('Only train QA samples are allowed')
        image = Path(old['source_image'])
        label = Path(old['source_label'])
        comparison = ROOT / old['comparison_image']
        checksum = digest(label)
        if checksum != old['source_label_sha256']:
            raise ValueError('Source annotation changed since QA sample: ' + old['sample_id'])
        if not comparison.is_file():
            raise FileNotFoundError(comparison)
        with Image.open(image) as picture:
            width, height = picture.size
        lines = [line for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
        values = lines[old['focus_label_row_index_zero_based']].split()
        identity = int(values[0])
        names = CLASS_NAMES[old['dataset']]
        if names[identity] != old['gt_class']:
            raise ValueError('Focus class mismatch')
        center_x, center_y, box_width, box_height = map(float, values[1:])
        box = [(center_x - box_width / 2) * width, (center_y - box_height / 2) * height,
               (center_x + box_width / 2) * width, (center_y + box_height / 2) * height]
        if max(abs(first - second) for first, second in zip(box, old['bbox_xyxy_native'])) > 1.1:
            raise ValueError('Focus coordinate mismatch: ' + old['sample_id'])
        kind = 'vehicle' if old['dataset'] == 'Vehicle363-v1' else 'light'
        obj = {'source_class': old['gt_class'], 'proposed_class': old['gt_class'],
               'bbox_xyxy': box, 'valid': 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height}
        hint = 'Chỉ kiểm tra đối tượng #1 được đánh dấu; không phải tất cả xe/đèn trong ảnh.'
        if kind == 'vehicle':
            hint += ' Quy ước MIO: pickup_truck/work_van thuộc truck; chưa chắc thì chọn Không chắc.'
        batches[kind].append({
            'id': old['sample_id'], 'annotation_kind': kind,
            'qa_image': str(comparison), 'source_image': str(image), 'source_image_sha256': digest(image),
            'source_label': str(label), 'source_label_sha256': checksum,
            'source_label_row_index_zero_based': old['focus_label_row_index_zero_based'],
            'source_group': old['source_group'], 'source': old['source'],
            'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED', 'class_mapping_confirmed': False,
            'notes': old.get('notes', ''), 'task_hint': hint, 'allowed_classes': names,
            'objects': [copy.deepcopy(obj)], 'original_focus_object': copy.deepcopy(obj),
            'edit_object_numbers': [1], 'geometry_edits': [], 'class_edits': [], 'remove_focus_object': False,
            'review_scope': 'FOCUS_OBJECT_ONLY',
            'correction': {'action': {'object_number': 1, 'action': 'REVIEW_FOCUS'},
                           'before': copy.deepcopy(obj), 'after': copy.deepcopy(obj)}})
        checksums[str(label)] = checksum
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'legacy_review.snapshot.json').write_bytes(content)
    for kind, rows in batches.items():
        data = {'status': 'PENDING_HUMAN_QA', 'is_ground_truth': False, 'annotation_kind': kind,
                'allow_bbox_edit': True, 'allow_class_edit': True, 'allow_focus_removal': True,
                'automatic_application': False, 'legacy_review_sha256': hashlib.sha256(content).hexdigest(), 'reviews': rows}
        (OUTPUT / (kind + '_review.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'status': 'READY_FOR_LOCAL_QA', 'source_label_sha256': checksums,
              'sample_counts': {key: len(value) for key, value in batches.items()},
              'class_counts': {key: dict(Counter(row['objects'][0]['source_class'] for row in value)) for key, value in batches.items()},
              'source_data_modified': False, 'active_ab_inputs_modified': False}
    (OUTPUT / 'preparation_report.json').write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')
    if SOURCE.read_bytes() != content or any(digest(Path(name)) != value for name, value in checksums.items()):
        raise RuntimeError('Source changed during preparation')
    print(json.dumps({key: value for key, value in report.items() if key != 'source_label_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
