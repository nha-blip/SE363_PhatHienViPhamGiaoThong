import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import psutil
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/quality_filter_720p_20261026'
OUTPUT = ROOT / 'data/YOLO363_filtered_720p_20261026'
ORIGINAL = ROOT / 'data/YOLO363_training_working'
EXPANDED = ROOT / 'data/TrafficLight363_expanded_candidate_20261026'
QA = ROOT / 'runs/project363/original_train_label_qa_20261026'
NAMES = {'vehicle': ['car', 'motorcycle', 'bus', 'truck'], 'light': ['red', 'yellow', 'green', 'off']}
POLICY = {'target_width': 1280, 'target_height': 720, 'allow_upscale': False,
          'min_native_width': 1280, 'min_native_height': 720,
          'vehicle_min_short_side': 12, 'vehicle_min_long_side': 20,
          'light_min_short_side': 4, 'light_min_long_side': 8,
          'min_global_laplacian_variance': 25,
          'vehicle_min_crop_laplacian_variance': 15,
          'light_min_crop_laplacian_variance': 10,
          'laplacian': 'cv2 grayscale CV_32F ksize=1 at letterboxed 1280x720',
          'mixed_image_policy': 'Reject whole image if ANY annotated object fails; never silently remove a true-object label',
          'test_policy': 'Inherit splits; filtered test is a separate subset; original benchmark remains unchanged',
          'blur_caveat': 'Heuristic, not semantic certainty; low texture/noise/compression can cause mistakes'}
DISCARD = {'vehicle_0007', 'vehicle_0011', 'vehicle_0015', 'vehicle_0025',
           'vehicle_0034', 'vehicle_0039', 'vehicle_0051', 'vehicle_0061'}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding='utf-8')
    temporary.replace(path)


def label_path(image):
    parts = list(Path(image).parts)
    index = len(parts) - 1 - list(reversed(parts)).index('images')
    parts[index] = 'labels'
    return Path(*parts).with_suffix('.txt')


def inputs():
    for split in ('train', 'val', 'test'):
        folder = ORIGINAL / 'Vehicle363-v1/images' / split
        with os.scandir(folder) as entries:
            for entry in entries:
                if Path(entry.name).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    yield {'kind': 'vehicle', 'split': split, 'image': entry.path, 'source': entry.name.split('__')[0]}
    for split, filename in (('train', 'train_unique.txt'), ('val', 'val_combined.txt'), ('test', 'test_original.txt')):
        for line in (EXPANDED / filename).read_text(encoding='utf-8').splitlines():
            image = Path(line)
            source = image.name.split('__')[0]
            for prefix in ('S2TLD_1080x1920', 'S2TLD_720x1280', 'ATLAS_sample'):
                if image.name.startswith(prefix):
                    source = prefix
            yield {'kind': 'light', 'split': split, 'image': str(image), 'source': source}


def qa_overrides():
    result = {}
    for kind in NAMES:
        data = json.loads((QA / (kind + '_review.json')).read_text(encoding='utf-8'))
        for row in data['reviews']:
            image = str(Path(row['source_image']).resolve()).casefold()
            entry = result.setdefault(image, {'reject': [], 'edits': [], 'label_hash': row['source_label_sha256']})
            if entry['label_hash'] != row['source_label_sha256']:
                raise ValueError('Conflicting QA label hashes')
            if row['id'] in DISCARD:
                entry['reject'].append('HUMAN_REQUEST_DISCARD_IMAGE')
            elif row['decision'] != 'KEEP' or not row['class_mapping_confirmed'] or row['bbox_scope'] not in {'WHOLE_OBJECT', 'SIGNAL_HEAD'} or row.get('notes', '').strip():
                entry['reject'].append('HUMAN_QA_UNRESOLVED')
            elif row.get('class_edits') or row.get('geometry_edits'):
                entry['edits'].append({'row': row['source_label_row_index_zero_based'],
                                       'object': row['objects'][0], 'sample_id': row['id']})
    return result


def parse_labels(content, names, width, height):
    objects = []
    for line in content.decode('utf-8-sig').splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise ValueError('Malformed YOLO row')
        identity, center_x, center_y, box_width, box_height = values
        if identity != int(identity) or not 0 <= int(identity) < len(names):
            raise ValueError('Invalid class')
        box = [(center_x - box_width / 2) * width, (center_y - box_height / 2) * height,
               (center_x + box_width / 2) * width, (center_y + box_height / 2) * height]
        if not (box[0] >= -0.05 and box[1] >= -0.05 and box[2] <= width + 0.05 and box[3] <= height + 0.05 and box[0] < box[2] and box[1] < box[3]):
            raise ValueError('Invalid native bbox')
        box = [max(0, min(value, width if index % 2 == 0 else height)) for index, value in enumerate(box)]
        objects.append({'class': int(identity), 'box': box})
    return objects


def transform_box(box, scale_x, scale_y, pad_x, pad_y):
    return [box[0] * scale_x + pad_x, box[1] * scale_y + pad_y,
            box[2] * scale_x + pad_x, box[3] * scale_y + pad_y]


def analyze(item, overrides, export=False):
    started = time.monotonic()
    image_path = Path(item['image'])
    key = hashlib.sha256((item['kind'] + '/' + item['split'] + '/' + str(image_path.resolve())).encode()).hexdigest()
    result = {**item, 'key': key, 'decision': 'REJECT', 'reasons': [], 'objects': [], 'metrics': {}, 'qa_edits': []}
    label = label_path(image_path)
    try:
        with Image.open(image_path) as picture:
            width, height = picture.size
        result.update(width=width, height=height, label=str(label))
        content = label.read_bytes()
        result['label_sha256'] = hashlib.sha256(content).hexdigest()
        objects = parse_labels(content, NAMES[item['kind']], width, height)
        result['original_class_counts'] = dict(Counter(obj['class'] for obj in objects))
        override = overrides.get(str(image_path.resolve()).casefold())
        if override:
            if result['label_sha256'] != override['label_hash']:
                raise ValueError('QA source label changed')
            result['reasons'].extend(override['reject'])
            for edit in override['edits']:
                objects[edit['row']] = {'class': NAMES[item['kind']].index(edit['object']['proposed_class']),
                                        'box': edit['object']['bbox_xyxy']}
                result['qa_edits'].append(edit['sample_id'])
        result['objects'] = objects
    except (OSError, ValueError, IndexError, KeyError) as error:
        result['reasons'].append('INVALID_IMAGE_OR_ANNOTATION')
        result['error'] = str(error)
        return result
    if width < POLICY['min_native_width'] or height < POLICY['min_native_height']:
        result['reasons'].append('NATIVE_BELOW_720P')
    if result['reasons']:
        return result
    ratio = min(1280 / width, 720 / height)
    if not POLICY['allow_upscale']:
        ratio = min(1, ratio)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    scale_x, scale_y = resized_width / width, resized_height / height
    pad_x, pad_y = (1280 - resized_width) // 2, (720 - resized_height) // 2
    target_objects = []
    for obj in objects:
        box = transform_box(obj['box'], scale_x, scale_y, pad_x, pad_y)
        box_width, box_height = box[2] - box[0], box[3] - box[1]
        small = (min(box_width, box_height) < POLICY[item['kind'] + '_min_short_side']
                 or max(box_width, box_height) < POLICY[item['kind'] + '_min_long_side'])
        target_objects.append({**obj, 'target_box': box, 'too_small': small})
    result['objects'] = target_objects
    useful_object_mode = POLICY.get('require_one_useful_object', False)
    if (not any(not obj['too_small'] for obj in target_objects) if useful_object_mode
            else any(obj['too_small'] for obj in target_objects)):
        result['reasons'].append('CONTAINS_TOO_SMALL_OBJECT_AT_720P')
        return result
    content = image_path.read_bytes()
    picture = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if picture is None or picture.shape[:2] != (height, width):
        result['reasons'].append('IMAGE_DECODE_ERROR')
        return result
    result['source_image_sha256'] = hashlib.sha256(content).hexdigest()
    resized = cv2.resize(picture, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    global_score = float(laplacian.var())
    result['metrics']['global_laplacian_variance'] = global_score
    if global_score < POLICY['min_global_laplacian_variance']:
        result['reasons'].append('GLOBAL_BLUR_HEURISTIC')
    for obj in target_objects:
        left, top, right, bottom = obj['target_box']
        region = laplacian[max(0, math.floor(top) - pad_y):min(resized_height, math.ceil(bottom) - pad_y),
                           max(0, math.floor(left) - pad_x):min(resized_width, math.ceil(right) - pad_x)]
        score = float(region.var()) if region.size else 0
        obj['crop_laplacian_variance'] = score
    if (not any(not obj['too_small'] and obj['crop_laplacian_variance'] >= POLICY[item['kind'] + '_min_crop_laplacian_variance'] for obj in target_objects) if useful_object_mode
            else any(obj['crop_laplacian_variance'] < POLICY[item['kind'] + '_min_crop_laplacian_variance'] for obj in target_objects)):
        result['reasons'].append('OBJECT_BLUR_HEURISTIC')
    if result['reasons']:
        return result
    result['decision'] = 'KEEP'
    if export:
        destination = OUTPUT / item['kind'] / 'images' / item['split'] / (image_path.stem + '__' + key[:12] + '.jpg')
        target_label = label_path(destination)
        frame = cv2.copyMakeBorder(resized, pad_y, 720 - resized_height - pad_y,
                                   pad_x, 1280 - resized_width - pad_x, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        temporary = destination.with_suffix('.partial.jpg')
        if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError('Failed writing filtered image')
        os.replace(temporary, destination)
        lines = []
        for obj in target_objects:
            left, top, right, bottom = obj['target_box']
            lines.append(f"{obj['class']} {(left + right) / 2560:.10f} {(top + bottom) / 1440:.10f} {(right - left) / 1280:.10f} {(bottom - top) / 720:.10f}")
        target_label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        result.update(output_image=str(destination), output_label=str(target_label))
    result['seconds'] = time.monotonic() - started
    return result


def bounded_results(items, overrides, export, workers):
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for index in range(workers * 3):
            item = next(iterator, None)
            if item is None:
                break
            pending.add(pool.submit(analyze, item, overrides, export))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                item = next(iterator, None)
                if item is not None:
                    pending.add(pool.submit(analyze, item, overrides, export))


def main():
    global OUTPUT, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--profile', choices=('original', 'balanced'), default='original')
    args = parser.parse_args()
    if args.profile == 'balanced':
        OUTPUT = ROOT / 'data/YOLO363_balanced_720p_20261027'
        REPORT = ROOT / 'runs/project363/quality_filter_balanced_20261027'
        POLICY.update(min_native_width=1, min_native_height=1,
                      vehicle_min_short_side=12, vehicle_min_long_side=20,
                      light_min_short_side=6, light_min_long_side=12,
                      vehicle_min_crop_laplacian_variance=50,
                      light_min_crop_laplacian_variance=30,
                      min_global_laplacian_variance=25,
                      require_one_useful_object=True,
                      mixed_image_policy='Keep frame with at least one sufficiently large sharp object; preserve ALL valid annotations including smaller objects',
                      resolution_policy='No minimum native resolution; downscale only, pad to 1280x720 without upscaling')
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (AttributeError, psutil.AccessDenied):
        pass
    REPORT.mkdir(parents=True, exist_ok=True)
    overrides = qa_overrides()
    if args.pilot:
        generator = random.Random(363)
        groups = defaultdict(list)
        seen = Counter()
        for item in inputs():
            group = (item['kind'], item['source'])
            seen[group] += 1
            reservoir = groups[group]
            if len(reservoir) < 120:
                reservoir.append(item)
            else:
                index = generator.randrange(seen[group])
                if index < 120:
                    reservoir[index] = item
        items = [item for rows in groups.values() for item in rows]
        results = list(bounded_results(items, overrides, False, args.workers))
        counts = defaultdict(Counter)
        for row in results:
            counts[row['kind'] + '/' + row['source']].update([row['decision']])
        write_json(REPORT / 'pilot.json', {'policy': POLICY, 'sample_counts': {key: dict(value) for key, value in counts.items()}, 'records': results})
        print(json.dumps({key: dict(value) for key, value in counts.items()}, indent=2), flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for kind in NAMES:
        for split in ('train', 'val', 'test'):
            for folder in ('images', 'labels'):
                (OUTPUT / kind / folder / split).mkdir(parents=True)
    write_json(OUTPUT / 'policy.json', POLICY)
    connection = sqlite3.connect(REPORT / 'audit.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS images (identity TEXT PRIMARY KEY, kind TEXT, split TEXT, source TEXT, decision TEXT, record TEXT)')
    if connection.execute('SELECT count(*) FROM images').fetchone()[0]:
        raise RuntimeError('Audit already contains a run; do not overwrite')
    counts = defaultdict(Counter)
    sources = defaultdict(Counter)
    class_counts = defaultdict(Counter)
    source_class_counts = defaultdict(Counter)
    reasons = defaultdict(Counter)
    processed = 0
    started = time.monotonic()
    try:
        for row in bounded_results(inputs(), overrides, True, args.workers):
            connection.execute('INSERT INTO images VALUES (?,?,?,?,?,?)', (row['key'], row['kind'], row['split'], row['source'], row['decision'], json.dumps(row)))
            processed += 1
            group = row['kind'] + '/' + row['split']
            counts[group][row['decision']] += 1
            sources[row['kind'] + '/' + row['source']][row['decision']] += 1
            reasons[group].update(row['reasons'])
            source_class_counts[group].update({NAMES[row['kind']][int(identity)]: count for identity, count in row.get('original_class_counts', {}).items()})
            if row['decision'] == 'KEEP':
                class_counts[group].update(NAMES[row['kind']][obj['class']] for obj in row['objects'])
            if processed % 100 == 0:
                connection.commit()
            if processed % 1000 == 0:
                state = {'status': 'RUNNING', 'processed': processed, 'seconds': round(time.monotonic() - started),
                         'image_counts': {key: dict(value) for key, value in counts.items()}}
                write_json(REPORT / 'status.json', state)
                print(json.dumps(state), flush=True)
        connection.commit()
        report = {'status': 'COMPLETE_HEURISTIC_FILTER_REQUIRES_REVIEW', 'processed': processed, 'policy': POLICY,
                  'image_counts': {key: dict(value) for key, value in counts.items()},
                  'source_counts': {key: dict(value) for key, value in sources.items()},
                  'source_bbox_counts': {key: dict(value) for key, value in source_class_counts.items()},
                  'kept_bbox_counts': {key: dict(value) for key, value in class_counts.items()},
                  'reject_reason_counts': {key: dict(value) for key, value in reasons.items()},
                  'elapsed_seconds': time.monotonic() - started, 'source_dataset_modified': False,
                  'active_ab_inputs_modified': False, 'training_started': False}
        for kind in NAMES:
            available = all(counts[kind + '/' + split]['KEEP'] > 0 for split in ('train', 'val', 'test'))
            if available:
                settings = {'path': (OUTPUT / kind).as_posix(), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': NAMES[kind]}
                (OUTPUT / kind / 'data.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
        write_json(OUTPUT / 'summary.json', report)
        write_json(REPORT / 'status.json', report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        connection.commit()
        write_json(REPORT / 'status.json', {'status': 'FAILED', 'processed': processed, 'error': str(error)})
        raise
    finally:
        connection.close()


if __name__ == '__main__':
    main()
