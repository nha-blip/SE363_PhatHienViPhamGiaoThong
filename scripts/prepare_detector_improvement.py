import argparse
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_external_detectors import LIGHT_NAMES, ROOT, VEHICLE_NAMES, read_labels, sha256, write_json
from scripts.render_external_detector_qa import panel


def capped_weights(counts, cap=3):
    largest = max(counts.values())
    return {name: min(cap, math.sqrt(largest / count)) if count else None for name, count in counts.items()}


def train_only(records):
    return [record for record in records if record['final_split'] == 'train']


def current_path(original, dataset):
    parts = Path(original.replace('\\', '/')).parts
    category = 'images' if 'images' in parts else 'labels'
    offset = parts.index(category)
    return ROOT / 'data/YOLO363_training_working' / dataset / Path(*parts[offset:])


def spread_groups(records, count, group_key):
    groups = defaultdict(list)
    for record in records:
        groups[record[group_key]].append(record)
    generator = random.Random(36320261020)
    keys = sorted(groups)
    generator.shuffle(keys)
    for key in keys:
        groups[key].sort(key=lambda item: item['final_image'])
        generator.shuffle(groups[key])
    result = []
    while len(result) < count:
        changed = False
        for key in keys:
            if groups[key] and len(result) < count:
                result.append(groups[key].pop())
                changed = True
        if not changed:
            break
    return result


def light_record(record):
    path = current_path(record['final_label'], 'TrafficLight363-v1')
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not np.isfinite(values).all() or values[0] != int(values[0]) or int(values[0]) not in LIGHT_NAMES:
            raise ValueError(f'Invalid light annotation: {path}')
        rows.append(int(values[0]))
    return {**record, 'current_label': str(path),
            'class_counts': dict(Counter(LIGHT_NAMES[identity] for identity in rows))}


def focus_preview(image, box, sample_id, target_name):
    left, top, right, bottom = box
    canvas = image.copy()
    cv2.rectangle(canvas, (round(left), round(top)), (round(right), round(bottom)), (0, 220, 255), 1)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half_width, half_height = max(35, right - left), max(35, bottom - top)
    crop_bounds = (max(0, int(center_x - half_width)), max(0, int(center_y - half_height)),
                   min(image.shape[1], int(center_x + half_width)), min(image.shape[0], int(center_y + half_height)))
    crop_left, crop_top, crop_right, crop_bottom = crop_bounds
    crop = image[crop_top:crop_bottom, crop_left:crop_right]
    crop_panel = panel(crop, f'ONE FOCUS BOX: {target_name}', 480, 640)
    scale = min(480 / crop.shape[1], 640 / crop.shape[0])
    offset_x = (480 - round(crop.shape[1] * scale)) // 2
    offset_y = 42 + (640 - round(crop.shape[0] * scale)) // 2
    cv2.rectangle(crop_panel,
                  (offset_x + round((left - crop_left) * scale), offset_y + round((top - crop_top) * scale)),
                  (offset_x + round((right - crop_left) * scale), offset_y + round((bottom - crop_top) * scale)),
                  (0, 220, 255), 1)
    return np.hstack((panel(canvas, f'{sample_id} | TRAIN GT: {target_name}', 960, 640), crop_panel))


def export_focus(record, dataset, names, target_name, directory, sample_id):
    image_path = current_path(record['final_image'], dataset)
    label_path = (Path(record['current_label']) if 'current_label' in record else
                  current_path(record['final_image'].replace('images/', 'labels/'), dataset).with_suffix('.txt'))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(image_path)
    ids, boxes = read_labels(label_path, image.shape[1], image.shape[0])
    candidates = [index for index, identity in enumerate(ids) if names[int(identity)] == target_name]
    if not candidates:
        raise ValueError(f'Metadata class missing in working label: {label_path}: {target_name}')
    focus = min(candidates, key=lambda index: (boxes[index, 2] - boxes[index, 0]) * (boxes[index, 3] - boxes[index, 1]))
    board = focus_preview(image, boxes[focus], sample_id, target_name)
    filename = f'{sample_id}.jpg'
    if not cv2.imwrite(str(directory / filename), board, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(filename)
    result = {'sample_id': sample_id, 'dataset': dataset, 'split': 'train',
              'source': record['source_dataset'],
              'source_group': record.get('source_group_id', record.get('source_group')),
              'comparison_image': str(directory / filename), 'source_image': str(image_path),
              'source_label': str(label_path), 'source_label_sha256': sha256(label_path),
              'focus_label_row_index_zero_based': focus, 'gt_class': target_name,
              'bbox_xyxy_native': boxes[focus].tolist(), 'decision': 'PENDING',
              'final_class': None, 'replacement_bbox_xyxy_native': None, 'notes': ''}
    write_json(directory / f'{sample_id}.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'status.json', {'status': 'PREPARING'})
    vehicle_metadata = ROOT / 'data/Vehicle363-v1/RELEASE_MANIFEST.json'
    light_metadata = ROOT / 'data/TrafficLight363-v1/final_qa_report.json'
    vehicle = json.loads(vehicle_metadata.read_text(encoding='utf-8'))
    light = json.loads(light_metadata.read_text(encoding='utf-8'))
    print('Loading preserved source manifests; not rescanning all vehicle labels...', flush=True)
    vehicle_records = train_only(json.loads((ROOT / 'data/Vehicle363-v1/metadata/source_manifest.json').read_text(encoding='utf-8')))
    light_manifest = json.loads((ROOT / 'data/TrafficLight363-v1/metadata/source_manifest.json').read_text(encoding='utf-8'))
    print('Checking current light labels across train/val/test...', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        light_records = list(pool.map(light_record, light_manifest['images']))
    current_lights = {}
    for split in ('train', 'val', 'test'):
        counts = Counter({name: 0 for name in LIGHT_NAMES.values()})
        per_source = defaultdict(Counter)
        records = [record for record in light_records if record['final_split'] == split]
        for record in records:
            counts.update(record['class_counts'])
            per_source[record['source_dataset']].update(record['class_counts'])
        expected = light['split_analysis']['statistics'][split]
        if dict(counts) != expected['classes'] or len(records) != expected['images']:
            raise ValueError(f'Working light labels differ from release metadata: {split}')
        current_lights[split] = {'images': len(records), 'class_counts': dict(counts),
                                 'per_source_class_counts': {source: dict(value) for source, value in per_source.items()}}
    if len(vehicle_records) != vehicle['split_counts']['train']['images']:
        raise ValueError('Vehicle manifest image count mismatch')
    inventory = {'vehicle': vehicle['split_counts'], 'traffic_light': current_lights,
                 'provenance': {'vehicle_metadata': str(vehicle_metadata), 'vehicle_metadata_sha256': sha256(vehicle_metadata),
                                'light_metadata': str(light_metadata), 'light_metadata_sha256': sha256(light_metadata)},
                 'verification_scope': {'vehicle': 'Preserved release counts; only selected train QA labels and images reopened, NOT a full recount.',
                                        'traffic_light': 'All current labels listed in the source manifest reopened and matched to release split counts; unlisted extra files not audited.'}}
    write_json(args.output / 'dataset_inventory.json', inventory)
    reviews = []
    specifications = [('vehicle', 'Vehicle363-v1', VEHICLE_NAMES, vehicle_records, 'source_group'),
                      ('traffic_light', 'TrafficLight363-v1', LIGHT_NAMES, train_only(light_records), 'source_group_id')]
    for task, dataset, names, records, group_key in specifications:
        directory = args.output / 'train_label_qa' / task
        directory.mkdir(parents=True)
        for source in sorted({record['source_dataset'] for record in records}):
            for name in names.values():
                candidates = [record for record in records if record['source_dataset'] == source and
                              record.get('target_class_counts', record.get('class_counts', {})).get(name, 0)]
                quota = 16 if task == 'traffic_light' and name in ('yellow', 'off') else 8
                chosen = spread_groups(candidates, quota, group_key)
                for record in chosen:
                    sample_id = f'{task}_{len(reviews) + 1:04d}'
                    reviews.append(export_focus(record, dataset, names, name, directory, sample_id))
                print(f'QA {task}/{source}/{name}: {len(chosen)}', flush=True)
        write_json(args.output / 'status.json', {'status': 'PREPARING', 'qa_samples': len(reviews), 'task': task})
    write_json(args.output / 'human_label_review.json', {
        'status': 'PENDING_HUMAN_QA_NOT_GROUND_TRUTH', 'is_ground_truth': False,
        'allowed_decisions': ['PENDING', 'KEEP_GT', 'CHANGE_CLASS', 'CHANGE_BBOX', 'REMOVE_BOX', 'UNCERTAIN'],
        'automatic_application': False, 'reviews': reviews})
    proposal = {'status': 'DRAFT_BLOCKED_ON_HUMAN_QA', 'training_started': False,
                'dataset_modified': False, 'validation_and_test_unchanged': True,
                'vehicle': {'strategy': 'No blanket rebalance; review car/truck semantics and motorcycle hard cases first.',
                            'mapping_warning': 'Preserved MIO mapping intentionally merges pickup_truck and work_van into truck; do not relabel these to car merely because they resemble passenger vehicles.'},
                'traffic_light': {'strategy': 'Candidate image-level oversampling after QA; retain mixed-class labels and negative images.',
                                  'candidate_class_weights': capped_weights(current_lights['train']['class_counts']),
                                  'weight_rule': 'min(3, sqrt(max_train_class_boxes / train_class_boxes)); proposal only, NOT a trainer parameter.',
                                  'constraints': ['Never copy validation/test into training.',
                                                  'Cap repeated frames within source sequence; preserve source diversity.',
                                                  'Maintain negative/background images and majority-class exposure.',
                                                  'Compare baseline vs balanced-only vs balanced plus validated new data on unchanged validation.',
                                                  'Re-evaluate held-out test only after selecting the configuration.']}}
    write_json(args.output / 'balancing_proposal.json', proposal)
    lines = ['# Dataset inventory and QA preparation', '',
             'Vehicle counts are preserved release metadata, not a new full label recount. All listed working light labels were read and verified against release split counts.', '',
             '| Train dataset | Class | Boxes | Share |', '|---|---|---:|---:|']
    for dataset, counts in [('vehicle', vehicle['split_counts']['train']['class_counts']),
                            ('traffic_light', current_lights['train']['class_counts'])]:
        for name, count in counts.items():
            lines.append(f'| {dataset} | {name} | {count} | {count / sum(counts.values()):.2%} |')
    lines += ['', f'Prepared {len(reviews)} training-only focus-box QA images. No model predictions are shown, to reduce prediction anchoring.',
              'Vehicle strata: source x class, 8 examples each. Light strata: source x class, 16 yellow/off and 8 red/green; groups spread where possible. Missing source/class combinations are not fabricated.',
              'MIO pickup_truck/work_van -> truck is the existing broad-class policy. Decide the policy before changing individual labels.',
              'No GT edits, sampling changes or training were performed. Balancing proposal remains gated on human QA.',
              'Repeated inspection of prior test errors is diagnostic; those samples must not become training hard examples.']
    (args.output / 'dataset_inventory.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(args.output / 'status.json', {'status': 'READY_FOR_HUMAN_QA', 'qa_samples': len(reviews)})
    print(f'READY_FOR_HUMAN_QA: {len(reviews)} samples', flush=True)


if __name__ == '__main__':
    main()
