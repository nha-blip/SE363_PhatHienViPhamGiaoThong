import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / 'data/TrafficLight363_expansion_pool_20261026'
ORIGINAL = ROOT / 'data/YOLO363_training_working/TrafficLight363-v1'
OUTPUT = ROOT / 'data/TrafficLight363_expanded_candidate_20261026'
CONFIG = ROOT / 'config/light363_expansion_20261026'
NAMES = ['red', 'yellow', 'green', 'off']


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding='utf-8')


def choose_holdout(rows, target=0.2):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sequence_group']].append(row)
    keys = sorted(groups)
    if len(keys) > 16:
        raise ValueError('Too many groups for exhaustive split selection')
    sources = {row['source'] for row in rows}
    totals = Counter()
    for row in rows:
        totals.update(row['class_counts'])
    candidates = []
    for length in range(1, len(keys)):
        for combination in itertools.combinations(keys, length):
            selected = set(combination)
            val = [row for row in rows if row['sequence_group'] in selected]
            train = [row for row in rows if row['sequence_group'] not in selected]
            if {row['source'] for row in val} != sources or {row['source'] for row in train} != sources:
                continue
            val_counts = Counter()
            for row in val:
                val_counts.update(row['class_counts'])
            if any(not 0 < val_counts[name] < count for name, count in totals.items() if count):
                continue
            score = abs(len(val) / len(rows) - target)
            score += sum(abs(val_counts[name] / count - target) for name, count in totals.items() if count) / len(totals)
            candidates.append((score, tuple(combination)))
    if not candidates:
        raise ValueError('Cannot cover sources/classes in both partitions without splitting groups')
    score, holdout = min(candidates)
    return set(holdout), score


def class_counts(label):
    counts = Counter()
    for line in label.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError('Malformed label: ' + str(label))
        identity = float(values[0])
        if identity != int(identity) or not 0 <= int(identity) < len(NAMES):
            raise ValueError('Unsupported class in ' + str(label))
        center_x, center_y, width, height = map(float, values[1:])
        if not (width > 0 and height > 0 and center_x - width / 2 >= -1e-6
                and center_y - height / 2 >= -1e-6 and center_x + width / 2 <= 1 + 1e-6
                and center_y + height / 2 <= 1 + 1e-6):
            raise ValueError('Invalid normalized bbox: ' + str(label))
        counts[NAMES[int(identity)]] += 1
    return dict(counts)


def original_rows(split):
    result = []
    for image in sorted((ORIGINAL / 'images' / split).iterdir()):
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        label = ORIGINAL / 'labels' / split / (image.stem + '.txt')
        if not label.is_file():
            raise ValueError('Missing original label: ' + str(label))
        result.append({'id': 'original/' + split + '/' + image.stem, 'image': str(image), 'label': str(label),
                       'source': 'Light363_original', 'split': split, 'class_counts': class_counts(label)})
    return result


def repeat_count(counts, rare_repeat=3):
    return rare_repeat if counts.get('yellow', 0) or counts.get('off', 0) else 1


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts.update(row['class_counts'])
    return {'images': len(rows), 'bbox_counts': dict(counts)}


def write_list(path, rows):
    path.write_text(''.join(Path(row['image']).as_posix() + '\n' for row in rows), encoding='utf-8')


def main():
    if json.loads((POOL / 'status.json').read_text())['status'] != 'COMPLETE_FILTERED_POOL_NOT_MERGED':
        raise ValueError('Filtered pool is not complete')
    pool_rows = json.loads((POOL / 'pool_manifest.json').read_text(encoding='utf-8'))
    holdout, score = choose_holdout(pool_rows)
    new_train = [row for row in pool_rows if row['sequence_group'] not in holdout]
    new_val = [row for row in pool_rows if row['sequence_group'] in holdout]
    originals = {split: original_rows(split) for split in ('train', 'val', 'test')}
    for row in pool_rows:
        if class_counts(Path(row['label'])) != row['class_counts']:
            raise ValueError('Pool label counts changed: ' + row['id'])
    train = originals['train'] + new_train
    val = originals['val'] + new_val
    test = originals['test']
    paths = {name: {row['image'] for row in rows} for name, rows in (('train', train), ('val', val), ('test', test))}
    if paths['train'] & paths['val'] or paths['train'] & paths['test'] or paths['val'] & paths['test']:
        raise ValueError('Image path leakage')
    if {row['sequence_group'] for row in new_train} & {row['sequence_group'] for row in new_val}:
        raise ValueError('Sequence leakage')
    balanced = [row for row in train for occurrence in range(repeat_count(row['class_counts']))]
    OUTPUT.mkdir(exist_ok=False)
    CONFIG.mkdir(parents=True, exist_ok=False)
    for filename, rows in (('train_unique.txt', train), ('train_balanced.txt', balanced),
                           ('train_new_only.txt', new_train), ('val_combined.txt', val),
                           ('val_original.txt', originals['val']), ('val_new_source.txt', new_val),
                           ('test_original.txt', test)):
        write_list(OUTPUT / filename, rows)
    configurations = {
        'data.yaml': ('train_balanced.txt', 'val_combined.txt'),
        'data_unique.yaml': ('train_unique.txt', 'val_combined.txt'),
        'data_original_val.yaml': ('train_balanced.txt', 'val_original.txt'),
        'data_new_holdout.yaml': ('train_balanced.txt', 'val_new_source.txt'),
    }
    for filename, (train_file, val_file) in configurations.items():
        settings = {'path': OUTPUT.as_posix(), 'train': train_file, 'val': val_file,
                    'test': 'test_original.txt', 'names': NAMES}
        (CONFIG / filename).write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    write_json(OUTPUT / 'new_source_split.json', {
        'policy': 'Atomic sequence/day/camera groups; deterministic exhaustive allocation; no model predictions used',
        'target_val_fraction': 0.2, 'actual_val_fraction': len(new_val) / len(pool_rows),
        'allocation_score': score, 'train_groups': sorted({row['sequence_group'] for row in new_train}),
        'val_groups': sorted(holdout), 'new_train_ids': [row['id'] for row in new_train],
        'new_val_ids': [row['id'] for row in new_val]})
    write_json(OUTPUT / 'sampling_manifest.json', [{
        'id': row['id'], 'image': row['image'], 'label': row['label'], 'class_counts': row['class_counts'],
        'repeat_count': repeat_count(row['class_counts']), 'source': row['source']} for row in train])
    report = {'status': 'TRAINING_CANDIDATE_CONFIGURED_NOT_TRAINED',
              'original_train': summarize(originals['train']), 'new_train': summarize(new_train),
              'train_unique': summarize(train), 'train_effective_with_repeats': summarize(balanced),
              'original_val': summarize(originals['val']), 'new_source_val': summarize(new_val),
              'combined_val': summarize(val), 'original_test': summarize(test),
              'sampling': 'Each TRAIN image with yellow/off occurs 3 times total; others once; no duplicate image files',
              'validation_sampling': 'No repetition; no new holdout image in any train list',
              'original_dataset_modified': False, 'training_started': False,
              'default_config': str(CONFIG / 'data.yaml'), 'is_ground_truth_for_entire_pool': False}
    write_json(OUTPUT / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
