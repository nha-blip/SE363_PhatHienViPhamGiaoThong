import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/traffic_light_external_raw'
OUTPUT = ROOT / 'data/traffic_light_external_audit_20261022'
CURRENT = ROOT / 'data/YOLO363_training_working/TrafficLight363-v1'
CLASSES = {'red': 0, 'yellow': 1, 'green': 2, 'off': 3}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding='utf-8')


def mapped_class(name):
    if name in CLASSES:
        return name
    if name == 'wait_on' or name.endswith('red_yellow'):
        return None
    for color in ('red', 'yellow', 'green'):
        if name.endswith('_' + color):
            return color
    return None


def valid_box(box, width, height):
    return (len(box) == 4 and all(math.isfinite(value) for value in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def fingerprint(path):
    content = path.read_bytes()
    picture = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if picture is None:
        return {'path': str(path), 'error': 'IMAGE_DECODE_FAILED'}
    reduced = cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), (9, 8))
    dhash = np.packbits(reduced[:, 1:] > reduced[:, :-1]).tobytes().hex()
    return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
            'pixel_sha256': hashlib.sha256(str(picture.shape).encode() + picture.tobytes()).hexdigest(),
            'dhash': dhash, 'width': picture.shape[1], 'height': picture.shape[0]}


def extract_records():
    records = []
    for archive_path in sorted(RAW.rglob('*.zip')):
        source = archive_path.stem
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.namelist()
            images = [name for name in members if Path(name).suffix.lower() in {'.jpg', '.png', '.jpeg'}]
            if source == 'ATLAS_sample':
                names = {int(key): value for key, value in yaml.load(archive.read('ATLAS/ATLAS_classes.yaml'), Loader=yaml.BaseLoader)['names'].items()}
            else:
                names = None
            for image_member in images:
                parts = image_member.split('/')
                category = 'images' if names is not None else 'JPEGImages'
                if category not in parts:
                    raise ValueError(f'Unexpected image layout: {image_member}')
                annotation_parts = list(parts)
                annotation_parts[parts.index(category)] = 'labels' if names is not None else 'Annotations'
                annotation_member = str(Path('/'.join(annotation_parts)).with_suffix('.txt' if names is not None else '.xml')).replace('\\', '/')
                identity = source + '_' + hashlib.sha256(image_member.encode()).hexdigest()[:20]
                image_path = OUTPUT / 'images' / (identity + Path(image_member).suffix.lower())
                image_path.write_bytes(archive.read(image_member))
                record = {'id': identity, 'source': source, 'archive': str(archive_path),
                          'image_member': image_member, 'annotation_member': annotation_member,
                          'image': str(image_path), 'objects': [], 'issues': []}
                picture = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if picture is None:
                    record['issues'].append('IMAGE_DECODE_FAILED')
                    records.append(record)
                    continue
                height, width = picture.shape[:2]
                record.update(width=width, height=height)
                if annotation_member not in members:
                    record['issues'].append('MISSING_ANNOTATION')
                else:
                    annotation = archive.read(annotation_member)
                    (OUTPUT / 'original_annotations' / (identity + Path(annotation_member).suffix)).write_bytes(annotation)
                    if names is None:
                        tree = ET.fromstring(annotation)
                        if int(tree.findtext('size/width')) != width or int(tree.findtext('size/height')) != height:
                            record['issues'].append('XML_IMAGE_SIZE_MISMATCH')
                        objects = [(obj.findtext('name'), [float(obj.findtext('bndbox/' + key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]) for obj in tree.findall('object')]
                        record['coordinate_policy'] = 'XML coordinates preserved; pixel convention requires QA before conversion'
                    else:
                        objects = []
                        for line in annotation.decode('utf-8').splitlines():
                            values = line.split()
                            if not values:
                                continue
                            if len(values) != 5:
                                record['issues'].append('INVALID_YOLO_ROW')
                                continue
                            class_id = int(values[0])
                            center_x, center_y, box_width, box_height = map(float, values[1:])
                            box = [(center_x - box_width / 2) * width, (center_y - box_height / 2) * height,
                                   (center_x + box_width / 2) * width, (center_y + box_height / 2) * height]
                            objects.append((names[class_id], box))
                        record['coordinate_policy'] = 'Source YOLO normalized coordinates decoded to pixels'
                    for name, box in objects:
                        valid = valid_box(box, width, height)
                        if not valid:
                            record['issues'].append('INVALID_BBOX')
                        mapped = mapped_class(name)
                        record['objects'].append({'source_class': name, 'proposed_class': mapped,
                                                  'bbox_xyxy': box, 'valid': valid})
                    if any(obj['proposed_class'] is None for obj in record['objects']):
                        record['issues'].append('UNMAPPED_CLASS_REQUIRES_REVIEW')
                record['eligible_after_human_qa'] = not record['issues']
                records.append(record)
        print(f'{source}: extracted {len(images)} images', flush=True)
    return records


def duplicates(records):
    new_paths = [Path(record['image']) for record in records]
    current_paths = sorted(path for path in (CURRENT / 'images').rglob('*') if path.suffix.lower() in {'.jpg', '.png', '.jpeg'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        fingerprints = list(pool.map(fingerprint, new_paths + current_paths))
    write_json(OUTPUT / 'fingerprints.json', fingerprints)
    groups = {}
    new_path_set = set(map(str, new_paths))
    for key in ('sha256', 'pixel_sha256'):
        buckets = defaultdict(list)
        for item in fingerprints:
            if key in item:
                buckets[item[key]].append(item['path'])
        groups[key] = [paths for paths in buckets.values() if len(paths) > 1 and any(path in new_path_set for path in paths)]
    valid = [item for item in fingerprints if 'dhash' in item]
    matrix = np.array([list(bytes.fromhex(item['dhash'])) for item in valid], dtype=np.uint8)
    lookup = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
    indexes = {item['path']: index for index, item in enumerate(valid)}
    candidates = []
    for record in records:
        index = indexes.get(record['image'])
        if index is None:
            continue
        distances = lookup[np.bitwise_xor(matrix, matrix[index])].sum(axis=1)
        matches = [int(match) for match in np.where(distances <= 4)[0] if match != index]
        if matches:
            ordered = sorted(matches, key=lambda match: (int(distances[match]), valid[match]['path']))
            candidates.append({'image': record['image'], 'total_candidates': len(matches),
                               'matches_first_10': [{'path': valid[match]['path'], 'distance': int(distances[match])} for match in ordered[:10]]})
        record['near_duplicate_candidates'] = len(matches)
        if (index + 1) % 1000 == 0:
            print(f'Duplicate scan: {index + 1}/{len(records)}', flush=True)
    result = {'existing_images_scanned': len(current_paths), 'new_images_scanned': len(records),
              'exact_groups': groups, 'near_candidates': candidates,
              'near_policy': '64-bit dHash distance <=4; candidates ONLY, not proof of duplicates; no deletion'}
    write_json(OUTPUT / 'duplicate_audit.json', result)
    return result


def render_review(records):
    buckets = defaultdict(list)
    for record in records:
        for name in sorted({obj['source_class'] for obj in record['objects']}):
            buckets[(record['source'], name)].append(record)
    selected = {}
    for key, rows in sorted(buckets.items()):
        for record in sorted(rows, key=lambda row: row['id'])[:6]:
            selected[record['id']] = record
    for record in records:
        if 'INVALID_BBOX' in record['issues']:
            selected[record['id']] = record
    previous = {}
    review_path = OUTPUT / 'human_review.json'
    if review_path.exists():
        previous = {row['id']: row for row in json.loads(review_path.read_text(encoding='utf-8'))['reviews']}
    reviews = []
    for record in selected.values():
        picture = cv2.imread(record['image'])
        if picture is None:
            continue
        overview_scale = 1440 / picture.shape[1]
        overview = cv2.resize(picture, (1440, round(picture.shape[0] * overview_scale)))
        for identity, obj in enumerate(record['objects'], 1):
            left, top, right, bottom = [round(value * overview_scale) for value in obj['bbox_xyxy']]
            cv2.rectangle(overview, (left, top), (right, bottom), (0, 0, 0), 7)
            cv2.rectangle(overview, (left, top), (right, bottom), (0, 255, 255), 3)
            label = f'#{identity}'
            label_top = max(0, top - 27)
            cv2.rectangle(overview, (max(0, left), label_top), (max(0, left) + 58, label_top + 25), (0, 0, 0), -1)
            cv2.putText(overview, label, (max(0, left) + 3, label_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        canvas = np.full((overview.shape[0] + 390, 1440, 3), 30, dtype=np.uint8)
        canvas[:overview.shape[0]] = overview
        cv2.putText(canvas, record['id'], (10, overview.shape[0] + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        focus = sorted(record['objects'], key=lambda obj: (obj['source_class'] not in {'off', 'yellow', 'wait_on'}, obj['source_class']))[:3]
        for position, obj in enumerate(focus):
            left, top, right, bottom = obj['bbox_xyxy']
            margin = max(12, int(max(right - left, bottom - top) * 0.5))
            crop_left = max(0, int(left) - margin)
            crop_top = max(0, int(top) - margin)
            crop = picture[crop_top:min(picture.shape[0], math.ceil(bottom) + margin),
                           crop_left:min(picture.shape[1], math.ceil(right) + margin)]
            if crop.size:
                scale = min(225 / crop.shape[1], 270 / crop.shape[0])
                crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
                marked = crop.copy()
                crop_box = [round((left - crop_left) * scale), round((top - crop_top) * scale),
                            round((right - crop_left) * scale), round((bottom - crop_top) * scale)]
                cv2.rectangle(marked, tuple(crop_box[:2]), tuple(crop_box[2:]), (0, 0, 0), 6)
                cv2.rectangle(marked, tuple(crop_box[:2]), tuple(crop_box[2:]), (0, 255, 255), 2)
                offset = position * 480
                for shift, panel in ((0, crop), (240, marked)):
                    canvas[overview.shape[0] + 100:overview.shape[0] + 100 + panel.shape[0], offset + shift:offset + shift + panel.shape[1]] = panel
                identity = record['objects'].index(obj) + 1
                cv2.putText(canvas, f"#{identity} {obj['source_class']}", (offset + 5, overview.shape[0] + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                for shift, label in ((0, 'RAW'), (240, 'BBOX')):
                    cv2.putText(canvas, label, (offset + shift + 5, overview.shape[0] + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        destination = OUTPUT / 'qa_images' / (record['id'] + '.jpg')
        cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 98])
        reviews.append({'id': record['id'], 'qa_image': str(destination), 'source_image': record['image'],
                        'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED', 'class_mapping_confirmed': False,
                        'notes': '', 'objects': record['objects']})
        if record['id'] in previous:
            for field in ('decision', 'bbox_scope', 'class_mapping_confirmed', 'notes'):
                reviews[-1][field] = previous[record['id']][field]
    write_json(OUTPUT / 'human_review.json', {'is_ground_truth': False, 'status': 'PENDING_HUMAN_QA', 'reviews': reviews})
    return len(reviews)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        records = json.loads((OUTPUT / 'records.json').read_text(encoding='utf-8'))
        count = render_review(records)
        summary = json.loads((OUTPUT / 'summary.json').read_text(encoding='utf-8'))
        summary['qa_images'] = count
        write_json(OUTPUT / 'summary.json', summary)
        print(f'Rendered {count} QA images; previous human decisions preserved', flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for folder in ('images', 'original_annotations', 'qa_images'):
        (OUTPUT / folder).mkdir(exist_ok=True)
    records = extract_records()
    duplicate_report = duplicates(records)
    count = render_review(records)
    write_json(OUTPUT / 'records.json', records)
    summary = {'status': 'AUDITED_NOT_MERGED', 'images': len(records), 'qa_images': count,
               'existing_images_scanned': duplicate_report['existing_images_scanned'],
               'source_classes': {source: dict(Counter(obj['source_class'] for row in records if row['source'] == source for obj in row['objects'])) for source in sorted({row['source'] for row in records})},
               'issues': dict(Counter(issue for row in records for issue in set(row['issues']))),
               'exact_duplicate_groups': {key: len(value) for key, value in duplicate_report['exact_groups'].items()},
               'images_with_near_candidates': len(duplicate_report['near_candidates']),
               'mapping': {name: mapped_class(name) for row in records for name in [obj['source_class'] for obj in row['objects']]},
               'train_labels_written': False, 'current_dataset_modified': False}
    write_json(OUTPUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
