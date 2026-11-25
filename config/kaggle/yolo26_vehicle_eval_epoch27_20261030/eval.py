import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


WORK = Path('/kaggle/working/vehicle_eval')
STAGING = Path('/tmp/vehicle_val')
ARCHIVE_SHA256 = 'd4fe27fbbf8d0b670c71a2b7500c8a277f00688d7eded24878373440ee1420fe'
CHECKPOINT_SHA256 = '7a80f6bb144e123c238b585ac4253d98dfb0346579337a0c01f0f7d7f88344ac'
NAMES = ['car', 'motorcycle', 'bus', 'truck']
VAL_IMAGES = 21460
VAL_BBOX = 78772
TRAINING_MAP5095 = 0.6472
IMGSZ = 640
CONF = 0.25
MATCH_IOU = 0.5
BUCKETS = ((8, '<8px'), (16, '8-16px'), (32, '16-32px'), (float('inf'), '>=32px'))


def emit(event, **values):
    print('VEHICLE_EVAL ' + json.dumps({'event': event, **values}), flush=True)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def input_file(pattern):
    matches = list(Path('/kaggle/input').glob('**/' + pattern))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one input for {pattern}, got {len(matches)}')
    return matches[0]


def val_members(archive):
    members = []
    images, labels = set(), set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or path.parts[:1] != ('vehicle',) or path.parts[2:3] != ('val',):
            continue
        if path.is_absolute() or '..' in path.parts or path.parts[1] not in ('images', 'labels'):
            raise ValueError('Unsafe archive member')
        (images if path.parts[1] == 'images' else labels).add(path.stem)
        members.append(member)
    if len(images) != VAL_IMAGES or images != labels:
        raise ValueError('Unexpected validation image/label pairs')
    return members


def stage_validation(source, destination):
    if digest_file(source) != ARCHIVE_SHA256:
        raise ValueError('Archive SHA256 mismatch')
    emit('ARCHIVE_VERIFIED')
    with zipfile.ZipFile(source) as archive:
        members = val_members(archive)
        destination.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        for index, member in enumerate(members, 1):
            archive.extract(member, destination)
            if index % 10000 == 0:
                emit('EXTRACT', completed=index, total=len(members), elapsed_seconds=round(time.monotonic() - started, 1))
    root = destination / 'vehicle'
    emit('VALIDATION_READY', root=str(root))
    return root


def box_iou(first, second):
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    area = (first[2]-first[0]) * (first[3]-first[1]) + (second[2]-second[0]) * (second[3]-second[1]) - overlap
    return overlap / area if area > 0 else 0.0


def match_detail(truths, predictions, iou_threshold=MATCH_IOU):
    """Class-agnostic greedy matching by confidence; returns predicted class per truth and unmatched prediction classes."""
    matched = [None] * len(truths)
    unmatched = []
    for box, score, identity in sorted(predictions, key=lambda row: -row[1]):
        best, best_iou = None, iou_threshold
        for index, (truth_box, _) in enumerate(truths):
            if matched[index] is not None:
                continue
            overlap = box_iou(box, truth_box)
            if overlap > best_iou:
                best, best_iou = index, overlap
        if best is None:
            unmatched.append(identity)
        else:
            matched[best] = identity
    return matched, unmatched


def match_frame(truths, predictions, iou_threshold=MATCH_IOU):
    """Return one OK, WRONG or MISSED outcome per ground-truth box."""
    matched, _ = match_detail(truths, predictions, iou_threshold)
    return ['MISSED' if identity is None else 'OK' if identity == truth[1] else 'WRONG'
            for truth, identity in zip(truths, matched)]


def size_bucket(box, scale):
    short = min(box[2]-box[0], box[3]-box[1]) * scale
    return next(name for limit, name in BUCKETS if short < limit)


def read_labels(path, width, height):
    rows = []
    for line in path.read_text().splitlines():
        identity, x, y, w, h = line.split()[:5]
        x, y, w, h = float(x)*width, float(y)*height, float(w)*width, float(h)*height
        rows.append(((x-w/2, y-h/2, x+w/2, y+h/2), int(identity)))
    return rows


def summarize_matrix(matrix, names):
    """Ultralytics confusion matrix: rows are predicted, columns are true, last index is background."""
    background = len(names)
    summary = {'columns_are_true_rows_are_predicted': True}
    for true_index, name in enumerate(names):
        total = sum(matrix[row][true_index] for row in range(background + 1))
        summary[name] = {'total': total, 'correct': matrix[true_index][true_index],
                         'missed_as_background': matrix[background][true_index],
                         'predicted_as': {other: matrix[row][true_index] for row, other in enumerate(names)
                                          if row != true_index},
                         'false_positive_from_background': matrix[true_index][background]}
    return summary


def per_size(model, images, labels):
    counts = {}
    confusion = {name: Counter() for name in NAMES}
    false_positives = Counter()
    frames = 0
    for result in model.predict(source=str(images), imgsz=IMGSZ, conf=CONF, batch=32, device=0, stream=True, verbose=False):
        height, width = result.orig_shape
        path = Path(result.path)
        truths = read_labels(labels / (path.stem + '.txt'), width, height)
        predictions = [(tuple(row[:4]), row[4], int(row[5])) for row in result.boxes.data.cpu().tolist()]
        scale = IMGSZ / max(width, height)
        matched, unmatched = match_detail(truths, predictions)
        for (box, identity), predicted in zip(truths, matched):
            outcome = 'MISSED' if predicted is None else 'OK' if predicted == identity else 'WRONG'
            counts.setdefault(NAMES[identity], {}).setdefault(size_bucket(box, scale), Counter())[outcome] += 1
            confusion[NAMES[identity]]['MISSED' if predicted is None else NAMES[predicted]] += 1
        false_positives.update(NAMES[identity] for identity in unmatched)
        frames += 1
        if frames % 5000 == 0:
            emit('SIZE_PASS', completed=frames, total=VAL_IMAGES)
    report = {}
    for name, buckets in counts.items():
        report[name] = {}
        for bucket, outcome in buckets.items():
            total = sum(outcome.values())
            report[name][bucket] = {**outcome, 'total': total, 'correct_class_recall': outcome['OK'] / total,
                                    'localized_recall': (outcome['OK'] + outcome['WRONG']) / total}
    return report, frames, {'true_to_predicted': {name: dict(row) for name, row in confusion.items()},
                            'unmatched_predictions': dict(false_positives)}


def main():
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.178'], check=True)
    import torch
    import yaml
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    WORK.mkdir(parents=True, exist_ok=True)
    checkpoint = input_file('vehicle_epoch27_best_stripped.pt')
    if digest_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint SHA256 mismatch')
    emit('CHECKPOINT_VERIFIED', path=str(checkpoint))
    root = stage_validation(input_file('vehicle_trainval.zip.bin'), STAGING)
    configuration = WORK / 'data.yaml'
    configuration.write_text(yaml.safe_dump({'path': str(root), 'train': 'images/val', 'val': 'images/val',
                                             'names': NAMES}, sort_keys=False))
    model = YOLO(str(checkpoint))
    if [model.names[index] for index in sorted(model.names)] != NAMES:
        raise ValueError('Checkpoint class names differ')
    started = time.monotonic()
    metrics = model.val(data=str(configuration), split='val', imgsz=IMGSZ, batch=32, device=0, workers=2,
                        plots=True, project=str(WORK), name='val', exist_ok=True, verbose=True)
    validation_seconds = time.monotonic() - started
    box = metrics.box
    per_class = {}
    for position, index in enumerate(box.ap_class_index):
        per_class[NAMES[int(index)]] = {'precision': float(box.p[position]), 'recall': float(box.r[position]),
                                        'ap50': float(box.ap50[position]), 'ap50_95': float(box.ap[position]),
                                        'bbox': int(metrics.nt_per_class[int(index)])}
    matrix = [[int(value) for value in row] for row in metrics.confusion_matrix.matrix.tolist()]
    write_json(WORK / 'report.json', {'status': 'VALIDATION_DONE', 'overall': {'precision': float(box.mp),
               'recall': float(box.mr), 'map50': float(box.map50), 'map50_95': float(box.map)}, 'per_class': per_class})
    emit('VALIDATION_DONE', map50_95=float(box.map), seconds=round(validation_seconds, 1))
    started = time.monotonic()
    sizes, frames, confusion = per_size(model, root / 'images/val', root / 'labels/val')
    bbox_total = sum(row['total'] for buckets in sizes.values() for row in buckets.values())
    delta = float(box.map) - TRAINING_MAP5095
    report = {'status': 'COMPLETE', 'split': 'val', 'test_used': False, 'checkpoint_sha256': CHECKPOINT_SHA256,
              'archive_sha256': ARCHIVE_SHA256, 'imgsz': IMGSZ, 'ultralytics': '8.4.178',
              'gpu': torch.cuda.get_device_name(0), 'images': frames, 'bbox_read': bbox_total, 'bbox_expected': VAL_BBOX,
              'overall': {'precision': float(box.mp), 'recall': float(box.mr), 'map50': float(box.map50),
                          'map50_95': float(box.map)},
              'training_map50_95': TRAINING_MAP5095, 'delta_from_training': delta,
              'reproduced_within_0_005': abs(delta) <= 0.005, 'per_class': per_class,
              'ultralytics_matrix_conf_0_001_do_not_read_as_operating_point': matrix,
              'ultralytics_matrix_summary': summarize_matrix(matrix, NAMES),
              'ultralytics_matrix_policy': 'Validator passes conf=0.001 and IoU 0.45; rows predicted, columns true',
              'operating_point_policy': f'conf>={CONF}, greedy class-agnostic IoU>{MATCH_IOU}; short side scaled to {IMGSZ}',
              'confusion_at_operating_point': confusion, 'per_size': sizes, 'speed_ms': {key: float(value) for key, value in metrics.speed.items()},
              'validation_seconds': validation_seconds, 'size_pass_seconds': time.monotonic() - started,
              'not_a_batch1_end_to_end_realtime_benchmark': True}
    write_json(WORK / 'report.json', report)
    emit('COMPLETE', **{key: report[key] for key in ('images', 'bbox_read', 'reproduced_within_0_005', 'delta_from_training')})
    shutil.rmtree(STAGING)


if __name__ == '__main__':
    main()
