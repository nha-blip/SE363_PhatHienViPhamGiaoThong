import hashlib
import json
import math
import sqlite3
from collections import Counter

import cv2
import numpy as np

if __package__:
    from .filter_detector_datasets_720p import ROOT, write_json
else:
    from filter_detector_datasets_720p import ROOT, write_json


OUTPUT = ROOT / 'data/readable_crop_pilot_20261027'
QA = ROOT / 'data/bbox_readability_qa_20261027/human_review.json'
AUDIT = ROOT / 'runs/project363/quality_filter_balanced_20261027/audit.sqlite'


def intersects(first, second):
    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(first[1], second[1])


def contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def safe_crop(focus, boxes, excluded):
    margin = max(48, max(focus[2] - focus[0], focus[3] - focus[1]))
    crop = [max(0, math.floor(focus[0] - margin)), max(0, math.floor(focus[1] - margin)),
            min(1280, math.ceil(focus[2] + margin)), min(720, math.ceil(focus[3] + margin))]
    for iteration in range(len(boxes) * 4 + 1):
        conflict = next((box for index, box in enumerate(boxes) if intersects(crop, box)
                         and (index in excluded or not contains(crop, box))), None)
        if conflict is None:
            return crop
        candidates = ([crop[0], crop[1], min(crop[2], math.floor(conflict[0])), crop[3]],
                      [max(crop[0], math.ceil(conflict[2])), crop[1], crop[2], crop[3]],
                      [crop[0], crop[1], crop[2], min(crop[3], math.floor(conflict[1]))],
                      [crop[0], max(crop[1], math.ceil(conflict[3])), crop[2], crop[3]])
        valid = [candidate for candidate in candidates if contains(candidate, focus)
                 and candidate[2] - candidate[0] >= 16 and candidate[3] - candidate[1] >= 16]
        if not valid:
            return None
        crop = max(valid, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
    return None


def main():
    reviews = json.loads(QA.read_text(encoding='utf-8'))
    if any(row['decision'] == 'PENDING' for row in reviews['reviews']):
        raise ValueError('QA incomplete')
    decisions = {(row['audit_identity'], row['object_index']): row['decision'] for row in reviews['reviews']}
    OUTPUT.mkdir(exist_ok=False)
    connection = sqlite3.connect(f'file:{AUDIT.as_posix()}?mode=ro', uri=True)
    counts = Counter()
    records = []
    cv2.setNumThreads(1)
    for sample in reviews['reviews']:
        if sample['decision'] != 'CLEAR':
            continue
        row = json.loads(connection.execute('SELECT record FROM images WHERE identity=?', (sample['audit_identity'],)).fetchone()[0])
        if row['split'] != 'train':
            raise ValueError('Pilot must only use train')
        if any(reason.startswith('HUMAN_') or reason.startswith('INVALID_') for reason in row['reasons']):
            counts['HELD_SOURCE_QA'] += 1
            continue
        boxes = [obj['target_box'] for obj in row['objects']]
        excluded = set()
        for index, obj in enumerate(row['objects']):
            decision = decisions.get((row['key'], index))
            short = min(boxes[index][2] - boxes[index][0], boxes[index][3] - boxes[index][1]) / 2
            threshold = 12 if row['kind'] == 'vehicle' else 4
            if decision in {'UNREADABLE', 'UNCERTAIN'} or (decision != 'CLEAR' and
                    (short < threshold or obj.get('crop_laplacian_variance', 0) < (50 if row['kind'] == 'vehicle' else 30))):
                excluded.add(index)
        focus_index = sample['object_index']
        crop = safe_crop(boxes[focus_index], boxes, excluded)
        record = {'sample_id': sample['id'], 'kind': row['kind'], 'audit_identity': row['key'],
                  'focus_index': focus_index, 'excluded_indices': sorted(excluded), 'crop_xyxy_720p': crop}
        if crop is None:
            record['status'] = 'SKIPPED_NO_SAFE_CROP'
            counts[record['status']] += 1
            records.append(record)
            continue
        if row['decision'] == 'KEEP':
            picture = cv2.imread(row['output_image'])
        else:
            source = cv2.imread(row['image'])
            if source is None:
                raise ValueError('Cannot decode source')
            ratio = min(1, 1280 / source.shape[1], 720 / source.shape[0])
            width, height = round(source.shape[1] * ratio), round(source.shape[0] * ratio)
            resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
            picture = np.full((720, 1280, 3), 114, dtype=np.uint8)
            left, top = (1280 - width) // 2, (720 - height) // 2
            picture[top:top + height, left:left + width] = resized
        if picture is None or picture.shape[:2] != (720, 1280):
            raise ValueError('Invalid 720p frame')
        left, top, right, bottom = crop
        included = [index for index, box in enumerate(boxes) if intersects(crop, box)]
        if focus_index not in included or any(index in excluded or not contains(crop, boxes[index]) for index in included):
            raise ValueError('Unsafe crop')
        folder = OUTPUT / row['kind']
        for category in ('images', 'labels', 'previews'):
            (folder / category).mkdir(parents=True, exist_ok=True)
        image_path = folder / 'images' / (sample['id'] + '.png')
        label_path = folder / 'labels' / (sample['id'] + '.txt')
        target = picture[top:bottom, left:right].copy()
        lines = []
        marked = target.copy()
        for index in included:
            box = boxes[index]
            local = [box[0] - left, box[1] - top, box[2] - left, box[3] - top]
            identity = row['objects'][index]['class']
            lines.append(f"{identity} {(local[0]+local[2])/2/(right-left):.10f} {(local[1]+local[3])/2/(bottom-top):.10f} {(local[2]-local[0])/(right-left):.10f} {(local[3]-local[1])/(bottom-top):.10f}")
            cv2.rectangle(marked, (round(local[0]), round(local[1])), (round(local[2]), round(local[3])), (0, 255, 255), 1)
            cv2.putText(marked, str(identity), (round(local[0]), max(10, round(local[1])-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if not cv2.imwrite(str(image_path), target):
            raise OSError('Failed crop write')
        label_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        factor = min(3, 1000 / marked.shape[1], 700 / marked.shape[0])
        preview_path = folder / 'previews' / (sample['id'] + '.png')
        if not cv2.imwrite(str(preview_path), cv2.resize(marked, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)):
            raise OSError('Failed preview write')
        record.update(status='DRAFT_SAFE_CROP', image=str(image_path), label=str(label_path),
                      preview=str(preview_path), included_indices=included, focus_human_confirmed=True,
                      other_objects_human_confirmed=False)
        counts[row['kind'] + '_crops'] += 1
        counts[row['kind'] + '_bbox'] += len(included)
        records.append(record)
    connection.close()
    write_json(OUTPUT / 'manifest.json', {'status': 'DRAFT_PILOT_REQUIRES_VISUAL_QA', 'counts': dict(counts),
               'qa_sha256': hashlib.sha256(QA.read_bytes()).hexdigest(), 'source_modified': False,
               'training_started': False, 'records': records})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    main()
