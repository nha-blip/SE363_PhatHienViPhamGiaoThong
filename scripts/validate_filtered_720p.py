import argparse
import heapq
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import cv2
import yaml

if __package__:
    from .filter_detector_datasets_720p import EXPANDED, NAMES, ORIGINAL, OUTPUT, REPORT, write_json
else:
    from filter_detector_datasets_720p import EXPANDED, NAMES, ORIGINAL, OUTPUT, REPORT, write_json


def preview(row, destination):
    kept = row['decision'] == 'KEEP'
    path = row['output_image'] if kept else row['image']
    picture = cv2.imread(path)
    if picture is None:
        return False
    objects = row.get('objects', [])
    if objects:
        focus = min(objects, key=lambda obj: obj.get('crop_laplacian_variance', float('inf')))
        box = focus['target_box'] if kept else focus['box']
        margin = max(12, math.ceil(max(box[2] - box[0], box[3] - box[1]) * 0.25))
        crop = picture[max(0, math.floor(box[1]) - margin):min(picture.shape[0], math.ceil(box[3]) + margin),
                       max(0, math.floor(box[0]) - margin):min(picture.shape[1], math.ceil(box[2]) + margin)]
        label = NAMES[row['kind']][focus['class']]
        crop_left, crop_top = max(0, math.floor(box[0]) - margin), max(0, math.floor(box[1]) - margin)
    else:
        crop = picture
        label = 'background'
    if crop.size == 0:
        return False
    import numpy as np

    overview_scale = min(960 / picture.shape[1], 540 / picture.shape[0])
    overview = cv2.resize(picture, (max(1, round(picture.shape[1] * overview_scale)), max(1, round(picture.shape[0] * overview_scale))))
    canvas = np.full((900, 1000, 3), 25, dtype=np.uint8)
    canvas[65:65 + overview.shape[0], 10:10 + overview.shape[1]] = overview
    scale = min(470 / crop.shape[1], 245 / crop.shape[0])
    enlarged = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    canvas[645:645 + enlarged.shape[0], 10:10 + enlarged.shape[1]] = enlarged
    marked = enlarged.copy()
    if objects:
        corners = [round((value - (crop_left if index % 2 == 0 else crop_top)) * scale) for index, value in enumerate(box)]
        cv2.rectangle(marked, tuple(corners[:2]), tuple(corners[2:]), (0, 0, 0), 6)
        cv2.rectangle(marked, tuple(corners[:2]), tuple(corners[2:]), (0, 255, 255), 2)
    canvas[645:645 + marked.shape[0], 510:510 + marked.shape[1]] = marked
    title = f"{row['kind']} {row['decision']} | native {row.get('width')}x{row.get('height')} | {row['split']}"
    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(canvas, ','.join(row['reasons'])[:110] or 'Passed automatic rules; inspect visually', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, f'Object crop: {label} (nearest-neighbor display only)', (10, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])


def main():
    global OUTPUT, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('original', 'balanced'), default='original')
    args = parser.parse_args()
    if args.profile == 'balanced':
        root = Path(__file__).resolve().parents[1]
        OUTPUT = root / 'data/YOLO363_balanced_720p_20261027'
        REPORT = root / 'runs/project363/quality_filter_balanced_20261027'
    status = json.loads((REPORT / 'status.json').read_text())
    if status['status'] != 'COMPLETE_HEURISTIC_FILTER_REQUIRES_REVIEW':
        raise RuntimeError('Filter is not complete')
    if status['processed'] != 164615:
        raise ValueError('Unexpected input count; compare working dataset inventory before using output')
    connection = sqlite3.connect(f'file:{(REPORT / "audit.sqlite").as_posix()}?mode=ro', uri=True)
    selected = {}
    heaps = {kind: [] for kind in NAMES}
    reject_limits = Counter()
    verified_files = Counter()
    bbox_counts = Counter()
    counter = 0
    for identity, record in connection.execute('SELECT identity,record FROM images ORDER BY identity'):
        row = json.loads(record)
        counter += 1
        if row['decision'] == 'KEEP':
            if not Path(row['output_image']).is_file() or not Path(row['output_label']).is_file():
                raise ValueError('Missing exported image/label')
            key = row['kind'] + '/' + row['split']
            verified_files[key] += 1
            bbox_counts[key] += len(row['objects'])
            if verified_files[key] <= 20:
                picture = cv2.imread(row['output_image'])
                if picture is None or picture.shape[:2] != (720, 1280):
                    raise ValueError('Exported frame dimensions invalid')
                lines = Path(row['output_label']).read_text().splitlines()
                if len(lines) != len(row['objects']):
                    raise ValueError('Object count changed in export')
                for line, obj in zip(lines, row['objects']):
                    values = line.split()
                    if int(values[0]) != obj['class']:
                        raise ValueError('Class mismatch')
                    center_x, center_y, width, height = map(float, values[1:])
                    decoded = [(center_x - width / 2) * 1280, (center_y - height / 2) * 720,
                               (center_x + width / 2) * 1280, (center_y + height / 2) * 720]
                    if max(abs(first - second) for first, second in zip(decoded, obj['target_box'])) > 0.001:
                        raise ValueError('Transformed label mismatch')
            score = min((obj.get('crop_laplacian_variance', 1e12) for obj in row['objects']), default=1e12)
            heap = heaps[row['kind']]
            heapq.heappush(heap, (-score, identity, row))
            if len(heap) > 6:
                heapq.heappop(heap)
            for obj in row['objects']:
                sample_key = (row['kind'], 'class', obj['class'])
                selected.setdefault(sample_key, row)
        elif row.get('reasons'):
            for reason in row['reasons']:
                key = (row['kind'], reason)
                if reject_limits[key] < 2:
                    selected[(row['kind'], reason, reject_limits[key])] = row
                    reject_limits[key] += 1
    connection.close()
    if counter != status['processed']:
        raise ValueError('Database coverage mismatch')
    for kind in NAMES:
        for index, (score, identity, row) in enumerate(heaps[kind]):
            selected[(kind, 'borderline_keep', index)] = row
        for split in ('train', 'val', 'test'):
            key = kind + '/' + split
            if verified_files[key] != status['image_counts'][key].get('KEEP', 0):
                raise ValueError('Summary count mismatch')
            image_count = sum(1 for path in (OUTPUT / kind / 'images' / split).glob('*.jpg'))
            label_count = sum(1 for path in (OUTPUT / kind / 'labels' / split).glob('*.txt'))
            if image_count != verified_files[key] or label_count != image_count:
                raise ValueError('Untracked file or missing label')
        settings = {'path': (OUTPUT / kind).as_posix(), 'train': 'images/train', 'val': 'images/val',
                    'test': ((ORIGINAL / 'Vehicle363-v1/images/test').as_posix() if kind == 'vehicle'
                             else (EXPANDED / 'test_original.txt').as_posix()), 'names': NAMES[kind]}
        (OUTPUT / kind / 'data_original_test.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    folder = OUTPUT / 'quality_review'
    folder.mkdir(exist_ok=False)
    index = []
    seen = set()
    for row in selected.values():
        if row['key'] in seen:
            continue
        seen.add(row['key'])
        destination = folder / f"{row['kind']}_{row['decision']}_{row['key'][:12]}.jpg"
        if preview(row, destination):
            index.append({'preview': str(destination), 'audit_identity': row['key'], 'source_image': row['image'],
                          'decision': row['decision'], 'reasons': row['reasons'], 'metrics': row['metrics']})
    write_json(folder / 'index.json', index)
    result = {'status': 'EXPORTED_FILES_VALIDATED_REQUIRES_VISUAL_QA', 'audited_images': counter,
              'kept_image_label_pairs': dict(verified_files), 'kept_bbox_counts': dict(bbox_counts),
              'dimension_and_coordinate_samples': 'First 20 hash-sorted accepted frames per dataset/split',
              'all_kept_file_presence_checked': True, 'review_previews': len(index),
              'source_dataset_modified': False, 'training_started': False}
    write_json(OUTPUT / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
