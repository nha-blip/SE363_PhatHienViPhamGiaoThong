import copy
import heapq
import json
import sqlite3
from collections import Counter

import cv2

if __package__:
    from .filter_detector_datasets_720p import REPORT, write_json
    from .refine_filtered_720p import OUTPUT
    from .validate_filtered_720p import preview
else:
    from filter_detector_datasets_720p import REPORT, write_json
    from refine_filtered_720p import OUTPUT
    from validate_filtered_720p import preview


def main():
    from pathlib import Path

    summary = json.loads((OUTPUT / 'summary.json').read_text())
    if summary['status'] != 'COMPLETE_STRICT_HEURISTIC_FILTER':
        raise ValueError('Strict export not complete')
    connection = sqlite3.connect(f'file:{(OUTPUT / "selection.sqlite").as_posix()}?mode=ro', uri=True)
    connection.execute('ATTACH DATABASE ? AS baseline', (f'file:{(REPORT / "audit.sqlite").as_posix()}?mode=ro',))
    samples = {}
    worst = {'vehicle': [], 'light': []}
    rejects = Counter()
    counts = Counter()
    seen = 0
    for identity, decision, encoded_reasons, image, label, record in connection.execute(
            'SELECT s.identity,s.decision,s.reasons,s.image,s.label,b.record FROM selection s JOIN baseline.images b ON s.identity=b.identity'):
        row = json.loads(record)
        source_decision = row['decision']
        row['decision'] = decision
        row['reasons'] = json.loads(encoded_reasons)
        seen += 1
        if decision == 'KEEP':
            if not Path(image).is_file() or not Path(label).is_file():
                raise ValueError('Missing strict image/label')
            row.update(output_image=image, output_label=label)
            key = row['kind'] + '/' + row['split']
            counts[key] += 1
            if counts[key] <= 10:
                picture = cv2.imread(image)
                if picture is None or picture.shape[:2] != (720, 1280):
                    raise ValueError('Not a native target-size exported frame')
                lines = Path(label).read_text().splitlines()
                if len(lines) != len(row['objects']):
                    raise ValueError('Dropped object annotation')
                for line, obj in zip(lines, row['objects']):
                    identity_value, center_x, center_y, width, height = map(float, line.split())
                    box = [(center_x - width / 2) * 1280, (center_y - height / 2) * 720,
                           (center_x + width / 2) * 1280, (center_y + height / 2) * 720]
                    if int(identity_value) != obj['class'] or max(abs(first - second) for first, second in zip(box, obj['target_box'])) > 0.001:
                        raise ValueError('Label geometry/class mismatch')
            score = min((obj['crop_laplacian_variance'] for obj in row['objects']), default=1e12)
            heapq.heappush(worst[row['kind']], (-score, identity, row))
            if len(worst[row['kind']]) > 6:
                heapq.heappop(worst[row['kind']])
            for obj in row['objects']:
                samples.setdefault((row['kind'], 'class', obj['class']), row)
        else:
            if source_decision == 'KEEP':
                row = copy.deepcopy(row)
                row['image'] = row['output_image']
                for obj in row['objects']:
                    obj['box'] = obj['target_box']
            for reason in row['reasons']:
                key = (row['kind'], reason)
                if rejects[key] < 2:
                    samples[(row['kind'], reason, rejects[key])] = row
                    rejects[key] += 1
    connection.close()
    if seen != summary['processed']:
        raise ValueError('Full audit coverage mismatch')
    for kind, heap in worst.items():
        for index, item in enumerate(heap):
            samples[(kind, 'borderline_keep', index)] = item[2]
    for key, count in counts.items():
        kind, split = key.split('/')
        if count != summary['image_counts'][key]['KEEP']:
            raise ValueError('Count mismatch')
        if sum(1 for path in (OUTPUT / kind / 'images' / split).glob('*.jpg')) != count or sum(1 for path in (OUTPUT / kind / 'labels' / split).glob('*.txt')) != count:
            raise ValueError('Untracked or missing dataset files')
    folder = OUTPUT / 'quality_review'
    folder.mkdir(exist_ok=False)
    unique = set()
    previews = []
    for row in samples.values():
        if row['key'] in unique:
            continue
        unique.add(row['key'])
        path = folder / f"{row['kind']}_{row['decision']}_{row['key'][:12]}.jpg"
        if preview(row, path):
            previews.append({'preview': str(path), 'audit_identity': row['key'], 'decision': row['decision'], 'reasons': row['reasons']})
    write_json(folder / 'index.json', previews)
    result = {'status': 'FILES_VALIDATED_AUTOMATIC_FILTER_NOT_MANUAL_GT', 'input_images_audited': seen,
              'output_image_label_pairs': dict(counts), 'all_kept_file_presence_checked': True,
              'all_copy_checksums_verified_by_exporter': summary['image_and_label_copy_checksums_verified'],
              'decoded_geometry_samples': '10 accepted hash-order entries per kind/split',
              'review_previews': len(previews), 'source_dataset_modified': False}
    write_json(OUTPUT / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
