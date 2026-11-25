import argparse
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def summarize_matrix(matrix, names):
    count = len(names)
    if len(matrix) != count + 1 or any(len(row) != count + 1 for row in matrix):
        raise ValueError('Expected prediction rows, GT columns, and background row/column')
    if any(value < 0 or int(value) != value for row in matrix for value in row):
        raise ValueError('Confusion counts must be nonnegative integers')
    rows = []
    confusions = []
    for identity, name in enumerate(names):
        total = int(sum(row[identity] for row in matrix))
        correct = int(matrix[identity][identity])
        missed = int(matrix[count][identity])
        wrong_class = int(sum(matrix[predicted][identity] for predicted in range(count) if predicted != identity))
        predicted_total = int(sum(matrix[identity]))
        rows.append({'name': name, 'gt': total, 'correct': correct, 'unmatched_gt': missed,
                     'wrong_class_gt': wrong_class, 'unmatched_predictions': int(matrix[identity][count]),
                     'precision_at_025': correct / predicted_total if predicted_total else None,
                     'recall_at_025': correct / total if total else None})
        for predicted in range(count):
            if predicted != identity and matrix[predicted][identity]:
                confusions.append({'gt': name, 'predicted': names[predicted],
                                   'count': int(matrix[predicted][identity])})
    totals = {key: sum(row[key] for row in rows) for key in
              ('gt', 'correct', 'unmatched_gt', 'wrong_class_gt', 'unmatched_predictions')}
    assert totals['gt'] == totals['correct'] + totals['unmatched_gt'] + totals['wrong_class_gt']
    return {'totals': totals, 'per_class': rows,
            'confusions': sorted(confusions, key=lambda entry: entry['count'], reverse=True)}


def label_counts(configuration):
    settings = yaml.safe_load(configuration.read_text(encoding='utf-8'))
    names = settings['names']
    result = {}
    for split in ('train', 'val', 'test'):
        counts = [0] * len(names)
        files = 0
        empty_files = 0
        for folder, _, filenames in os.walk(Path(settings['path']) / 'labels' / split):
            for filename in filenames:
                if not filename.endswith('.txt'):
                    continue
                lines = (Path(folder) / filename).read_text(encoding='utf-8').splitlines()
                nonempty = [line for line in lines if line.strip()]
                files += 1
                empty_files += not nonempty
                for line in nonempty:
                    values = line.split()
                    identity = int(values[0])
                    if len(values) != 5 or identity not in names:
                        raise ValueError(f'Invalid class or box: {folder}/{filename}')
                    counts[identity] += 1
        if not files:
            raise ValueError(f'No {split} labels found')
        result[split] = {'label_files': files, 'empty_labels': empty_files,
                         'boxes': {names[identity]: value for identity, value in enumerate(counts)}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count-labels', action='store_true')
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    jobs = {}
    lines = ['# Bbox error audit', '',
             'Full held-out test results; confidence 0.25, IoU 0.50. This is not the maximum-F1 P/R operating point.',
             'Counts are boxes, not frames. Wrong-class matches are counted once per GT, not added again as unmatched predictions.',
             'Unmatched GT may be undetected, below threshold, or poorly localized. Unmatched predictions may be background, duplicate boxes, localization errors or annotation omissions. Human QA is required before editing GT.',
             'Aggregated confusion matrices do not contain bbox coordinates or error-by-size attribution; this audit cannot establish that small size, occlusion or GT errors caused a specific miss.', '',
             '| Model | GT | Correct | Unmatched GT | Wrong-class GT | Unmatched predictions |',
             '|---|---:|---:|---:|---:|---:|']
    for path in sorted(arguments.evaluation.glob('*/report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        if report['evaluation_scope'] != 'FULL_TEST_SPLIT' or report['status'] != 'COMPLETE':
            raise ValueError('Only completed full test reports are supported')
        names = [row['name'] for row in sorted(report['per_class'], key=lambda row: row['class_id'])]
        result = summarize_matrix(report['confusion_matrix'], names)
        for row, original in zip(result['per_class'], sorted(report['per_class'], key=lambda row: row['class_id'])):
            if row['gt'] != original['ground_truth_instances']:
                raise ValueError('Confusion matrix and GT counts disagree')
        result['weights_sha256'] = report['weights_sha256']
        result['test_image_list_sha256'] = report['test_image_list_sha256']
        result['test_annotation_content_sha256'] = report['test_annotation_content_sha256']
        jobs[report['job']] = result
        totals = result['totals']
        lines.append('| ' + report['job'] + ' | ' + ' | '.join(str(totals[key]) for key in
                     ('gt', 'correct', 'unmatched_gt', 'wrong_class_gt', 'unmatched_predictions')) + ' |')
    datasets = {}
    for dataset in ('vehicle', 'traffic_light') if arguments.count_labels else ():
        print(f'Counting {dataset} labels...', flush=True)
        datasets[dataset] = label_counts(ROOT / f'config/{dataset}_dataset.yaml')
    for job, result in jobs.items():
        lines += ['', '## ' + job, '',
                  '| GT class | GT | Correct | Unmatched GT | Wrong-class GT | Unmatched predictions of class | Recall @0.25 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for row in result['per_class']:
            recall = 'N/A' if row['recall_at_025'] is None else f"{row['recall_at_025']:.2%}"
            lines.append('| ' + row['name'] + ' | ' + ' | '.join(str(row[key]) for key in
                         ('gt', 'correct', 'unmatched_gt', 'wrong_class_gt', 'unmatched_predictions')) + ' | ' + recall + ' |')
        lines += ['', 'Largest class confusions (GT -> prediction):']
        lines += [f"- {entry['gt']} -> {entry['predicted']}: {entry['count']}" for entry in result['confusions'][:8]]
    lines += ['', '## Dataset label counts', '',
              'Raw label rows, before any runtime duplicate-label filtering.' if datasets else
              'Not recomputed. Use --count-labels for a separate, potentially slow full train/val/test label inventory. No claim that training data is too small follows from this audit.']
    for dataset, splits in datasets.items():
        lines += ['', '### ' + dataset]
        for split, value in splits.items():
            lines.append(f"- {split}: {value['label_files']} images/label files, {value['empty_labels']} empty; " +
                         ', '.join(f'{name}={count}' for name, count in value['boxes'].items()))
    lines += ['', '## Improvement priorities', '',
              '1. Vehicle: prioritize motorcycle misses and car/motorcycle unmatched predictions. Review duplicates, low-IoU boxes and dense traffic before changing annotations. Calibrate per-class confidence/NMS on validation, not test.',
              '2. Vehicle: inspect car/truck confusion and box tightness. Review train/validation errors by source, object size and occlusion before adding domain-matched data. AP50/AP50-95 gaps alone do not prove bad GT.',
              '3. Lights: yellow and off are weak. Audit class-specific train support, label consistency, dark lamps and yellow->red confusion. Add genuinely diverse, human-reviewed training sequences rather than adjacent near-duplicate frames.',
              '4. Lights: test resolution/ROI improvements using validation and remeasure latency. Current aggregates cannot distinguish tiny-object failure from localization, class confusion or confidence filtering.',
              '5. Use validation to choose thresholds, augmentation and new-data mixtures. Keep held-out test untouched; repeated test-driven tuning risks optimistic estimates. Consider a new held-out camera/source for final evaluation.',
              '6. Class balance is descriptive, not proof of causality; dataset composition and camera/source domain may differ between train and test. No GT labels or models were modified.']
    (arguments.output / 'report.json').write_text(json.dumps({'jobs': jobs, 'dataset_counts': datasets}, indent=2) + '\n', encoding='utf-8')
    (arguments.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({job: value['totals'] for job, value in jobs.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
