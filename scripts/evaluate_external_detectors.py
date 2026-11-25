import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
VEHICLE_NAMES = {0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck'}
LIGHT_NAMES = {0: 'red', 1: 'yellow', 2: 'green', 3: 'off'}
JOBS = (
    ('oiv7_vehicle', 'yolov8s-oiv7.pt', 'vehicle', 'vehicle'),
    ('oiv7_light_localization', 'yolov8s-oiv7.pt', 'traffic_light', 'localization'),
    ('kastel_light_state', 'kastel_traffic_lights_yolov8x.pt', 'traffic_light', 'state'),
)
TRAINED_JOBS = (
    ('nano_vehicle', str(ROOT / 'weights/trained_yolo_20261018/nano/vehicle/best.pt'), 'vehicle', 'vehicle'),
    ('small_vehicle', str(ROOT / 'runs/local_training/vehicle_small_resume_epoch60_100/weights/best.pt'), 'vehicle', 'vehicle'),
    ('nano_light_state', str(ROOT / 'weights/trained_yolo_20261018/nano/traffic_light/best.pt'), 'traffic_light', 'state'),
    ('small_light_state', str(ROOT / 'weights/trained_yolo_20261018/small/traffic_light/best.pt'), 'traffic_light', 'state'),
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def class_mapping(names, mode):
    result = {}
    excluded = {}
    exact_vehicle = {name: identity for identity, name in VEHICLE_NAMES.items()}
    for identity, name in names.items():
        normalized = name.lower()
        target = None
        if mode == 'vehicle':
            target = exact_vehicle.get(normalized)
        elif mode == 'localization' and normalized == 'traffic light':
            target = 0
        elif mode == 'state':
            if normalized == 'off':
                target = 3
            elif normalized.endswith('_red_yellow'):
                excluded[int(identity)] = name
            else:
                for color, target_id in (('red', 0), ('yellow', 1), ('green', 2)):
                    if normalized == color or normalized.endswith('_' + color):
                        target = target_id
                        break
        if target is not None:
            result[int(identity)] = target
    if not result:
        raise ValueError(f'No compatible classes for {mode}: {names}')
    return result, excluded


def read_labels(path, width, height, collapse=False):
    classes = []
    boxes = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(f'Invalid annotation: {path}')
        identity, center_x, center_y, box_width, box_height = values
        if identity != int(identity) or not 0 <= identity < 4:
            raise ValueError(f'Unexpected class: {path}')
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and
                0 < box_width <= 1 and 0 < box_height <= 1):
            raise ValueError(f'Invalid normalized bbox: {path}')
        classes.append(0 if collapse else int(identity))
        boxes.append([(center_x - box_width / 2) * width,
                      (center_y - box_height / 2) * height,
                      (center_x + box_width / 2) * width,
                      (center_y + box_height / 2) * height])
    return np.asarray(classes, dtype=np.float32), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def test_images(dataset_path, limit):
    dataset = yaml.safe_load(dataset_path.read_text(encoding='utf-8'))
    images_root = Path(dataset['path']) / dataset['test']
    labels_root = Path(dataset['path']) / 'labels/test'
    images = sorted(path for path in images_root.rglob('*')
                    if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
    if not images:
        raise ValueError(f'No test images: {images_root}')
    if limit and len(images) > limit:
        indices = np.linspace(0, len(images) - 1, limit, dtype=int)
        images = [images[index] for index in indices]
    pairs = [(image, labels_root / image.relative_to(images_root).with_suffix('.txt')) for image in images]
    if any(not label.is_file() for _, label in pairs):
        raise FileNotFoundError('Missing test labels')
    return pairs


def mapped_predictions(result, mapping):
    boxes = result.boxes.data.detach().cpu().numpy()
    keep = np.asarray([int(identity) in mapping for identity in boxes[:, 5]], dtype=bool)
    selected = boxes[keep].copy()
    for row in selected:
        row[5] = mapping[int(row[5])]
    return selected[np.argsort(-selected[:, 4], kind='stable')]


def latency(model, pairs, mapping, arguments):
    import cv2
    import torch

    count = min(arguments.benchmark_images, len(pairs))
    indices = np.linspace(0, len(pairs) - 1, count, dtype=int)
    samples = [pairs[index][0] for index in indices]
    first = cv2.imread(str(samples[0]))
    if arguments.device != 'cpu':
        torch.cuda.reset_peak_memory_stats()
    options = dict(imgsz=640, device=arguments.device, quantize=32 if arguments.device == 'cpu' else 16,
                   conf=0.25, iou=0.7, max_det=300, classes=list(mapping),
                   verbose=False, save=False)
    for _ in range(10):
        model.predict(first, **options)
    timings = []
    decode_timings = []
    for path in samples:
        started = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode: {path}')
        decode_timings.append((time.perf_counter() - started) * 1000)
        if arguments.device != 'cpu':
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = model.predict(image, **options)[0]
        mapped_predictions(result, mapping)
        if arguments.device != 'cpu':
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    mean = float(np.mean(timings))
    return {'images': count, 'warmup_images': 10, 'batch': 1, 'imgsz': 640,
            'confidence': 0.25, 'mean_ms': mean, 'p50_ms': float(np.percentile(timings, 50)),
            'p95_ms': float(np.percentile(timings, 95)), 'fps_model_call': 1000 / mean,
            'mean_decode_ms': float(np.mean(decode_timings)),
            'precision': 'FP32' if arguments.device == 'cpu' else 'FP16',
            'peak_torch_allocated_mb': None if arguments.device == 'cpu' else
                torch.cuda.max_memory_allocated() / (1024 * 1024),
            'fps_decode_and_model': 1000 / float(np.mean(np.add(timings, decode_timings))),
            'excluded': ['camera capture', 'tracking', 'violation logic', 'rendering', 'encoding', 'VLM'],
            'realtime_30fps_model_call_only': bool(np.percentile(timings, 95) <= 1000 / 30),
            'not_a_deployment_realtime_certification': True}


def evaluate(job, arguments):
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou

    identity, filename, dataset_name, mode = job
    output = arguments.output / identity
    output.mkdir(parents=True, exist_ok=False)
    weights = ROOT / 'weights/external_eval_20261019' / filename
    pairs = test_images(ROOT / f'config/{dataset_name}_dataset.yaml', arguments.limit)
    factory = getattr(arguments, 'model_factory', None)
    model = factory(str(weights), mode) if factory else YOLO(str(weights))
    if getattr(arguments, 'profile', 'external') == 'trained':
        expected = VEHICLE_NAMES if mode == 'vehicle' else LIGHT_NAMES
        if model.names != expected:
            raise ValueError(f'Trained checkpoint namespace mismatch: {model.names}')
    mapping_function = getattr(arguments, 'mapping_function', class_mapping)
    mapping, excluded = mapping_function(model.names, mode)
    names = VEHICLE_NAMES if mode == 'vehicle' else {0: 'traffic_light'} if mode == 'localization' else LIGHT_NAMES
    policy = {'source_names': model.names, 'mapped_source_ids': mapping,
              'excluded_red_yellow': excluded, 'target_names': names,
              'vehicle_mapping': 'Only exact Car/Motorcycle/Bus/Truck. No guessed Van/Taxi/Ambulance mapping.',
              'light_mapping': 'Directional labels pooled by color; red_yellow unsupported, not forced into red or yellow.',
              'direction_accuracy': 'NOT_EVALUABLE: current GT has no arrow-direction labels',
              'oiv7_color_accuracy': 'NOT_SUPPORTED: only generic Traffic light class'}
    if getattr(arguments, 'profile', 'external') == 'trained':
        policy = {'source_names': model.names, 'mapped_source_ids': mapping,
                  'excluded_red_yellow': excluded, 'target_names': names,
                  'mapping_policy': 'Exact four-class project namespace; identity mapping.',
                  'direction_accuracy': 'NOT_EVALUABLE: current GT has no arrow-direction labels'}
    policy.update(getattr(arguments, 'policy_overrides', {}))
    write_json(output / 'mapping.json', policy)
    metrics = DetMetrics(names=names)
    confusion = ConfusionMatrix(names=names)
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    unsupported_count = 0
    label_digest = hashlib.sha256()
    image_list_digest = hashlib.sha256()
    started = time.perf_counter()
    for index, (image_path, label_path) in enumerate(pairs, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Cannot decode: {image_path}')
        height, width = image.shape[:2]
        target_ids, target_boxes = read_labels(label_path, width, height, mode == 'localization')
        label_digest.update(label_path.read_bytes())
        image_list_digest.update((image_path.as_posix() + '\n').encode())
        result = model.predict(image, imgsz=640, device=arguments.device,
                               quantize=32 if arguments.device == 'cpu' else 16, conf=0.001, iou=0.7,
                               max_det=300, classes=list(mapping) + list(excluded),
                               verbose=False, save=False)[0]
        native_ids = result.boxes.cls.detach().cpu().numpy()
        unsupported_count += sum(int(value) in excluded for value in native_ids)
        predictions = mapped_predictions(result, mapping)
        prediction_writer = getattr(arguments, 'prediction_writer', None)
        if prediction_writer:
            prediction_writer(image_path, label_path, predictions, result)
        predicted_ids = torch.from_numpy(predictions[:, 5])
        target_tensor = torch.from_numpy(target_ids)
        iou = box_iou(torch.from_numpy(target_boxes), torch.from_numpy(predictions[:, :4]))
        true_positives = BaseValidator.match_predictions(matcher, predicted_ids, target_tensor, iou).numpy()
        metrics.update_stats({'tp': true_positives, 'conf': predictions[:, 4],
                              'pred_cls': predictions[:, 5], 'target_cls': target_ids,
                              'target_img': np.unique(target_ids), 'im_name': image_path.name})
        confusion.process_batch(
            {'bboxes': torch.from_numpy(predictions[:, :4]),
             'conf': torch.from_numpy(predictions[:, 4]), 'cls': predicted_ids},
            {'bboxes': torch.from_numpy(target_boxes), 'cls': target_tensor},
            conf=0.25, iou_thres=0.5)
        if index == 1 or index % 100 == 0 or index == len(pairs):
            write_json(output / 'status.json', {'status': 'EVALUATING', 'processed': index,
                                               'total': len(pairs), 'seconds': time.perf_counter() - started})
            print(f'{identity}: {index}/{len(pairs)}', flush=True)
    metrics.process(save_dir=output, plot=False)
    per_class = []
    for class_id, name in names.items():
        count = int(metrics.nt_per_class[class_id])
        row = {'class_id': class_id, 'name': name, 'ground_truth_instances': count,
               'precision': None, 'recall': None, 'ap50': None, 'ap50_95': None}
        positions = np.flatnonzero(metrics.box.ap_class_index == class_id)
        if count and len(positions):
            position = int(positions[0])
            precision, recall, ap50, ap = metrics.class_result(position)
            row.update(precision=float(precision), recall=float(recall), ap50=float(ap50), ap50_95=float(ap))
        per_class.append(row)
    accuracy_seconds = time.perf_counter() - started
    benchmark = latency(model, pairs, mapping, arguments)
    report = {'status': 'COMPLETE', 'job': identity,
              'evaluation_scope': 'SMOKE_SUBSET_NOT_FINAL' if arguments.limit else 'FULL_TEST_SPLIT',
              'images': len(pairs), 'weights': str(weights), 'weights_sha256': sha256(weights),
              'test_image_list_sha256': image_list_digest.hexdigest(),
              'test_annotation_content_sha256': label_digest.hexdigest(),
              'metrics': metrics.results_dict, 'per_class': per_class,
              'precision_recall_policy': 'Ultralytics maximum-mean-F1 confidence, not fixed 0.25',
              'confusion_matrix': confusion.matrix.tolist(),
              'confusion_policy': 'rows=prediction, columns=GT, last row/column=background; conf=.25 IoU=.5',
              'unsupported_red_yellow_predictions_at_conf_0001': int(unsupported_count),
              'mapping': policy, 'latency': benchmark,
              'accuracy_seconds': accuracy_seconds,
              'gpu_training_contention': arguments.contention,
              'direction_and_vehicle_subtype_accuracy': 'NOT_EVALUABLE_FROM_CURRENT_GT'}
    write_json(output / 'report.json', report)
    write_json(output / 'status.json', {'status': 'COMPLETE', 'processed': len(pairs)})
    del model
    if arguments.device != 'cpu':
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--benchmark-images', type=int, default=200)
    parser.add_argument('--wait-for-pid', type=int)
    parser.add_argument('--contention', action='store_true')
    parser.add_argument('--profile', choices=('external', 'trained'), default='external')
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    import psutil
    import torch
    import ultralytics

    torch.set_num_threads(1)
    environment = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                   'device': arguments.device, 'limit': arguments.limit, 'profile': arguments.profile,
                   'created_at_utc': datetime.now(timezone.utc).isoformat()}
    if arguments.wait_for_pid:
        try:
            training = psutil.Process(arguments.wait_for_pid)
            creation = training.create_time()
            while training.is_running() and training.create_time() == creation:
                write_json(arguments.output / 'status.json', {
                    'status': 'WAITING_FOR_TRAINING_TO_EXIT', 'pid': arguments.wait_for_pid,
                    'last_checked_at_utc': datetime.now(timezone.utc).isoformat()})
                time.sleep(60)
        except psutil.NoSuchProcess:
            pass
    if arguments.device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        idle_samples = 0
        while idle_samples < 3:
            utilization = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
                '--id', arguments.device,
            ], check=True, capture_output=True, text=True).stdout.strip()
            idle_samples = idle_samples + 1 if int(utilization) <= 10 else 0
            write_json(arguments.output / 'status.json', {
                'status': 'WAITING_FOR_GPU_IDLE', 'utilization': int(utilization),
                'consecutive_idle_samples': idle_samples})
            time.sleep(10)
        environment['gpu'] = torch.cuda.get_device_name(0)
    write_json(arguments.output / 'environment.json', environment)
    reports = []
    try:
        for job in TRAINED_JOBS if arguments.profile == 'trained' else JOBS:
            reports.append(evaluate(job, arguments))
            write_json(arguments.output / 'report.json', {'environment': environment, 'jobs': reports})
    except Exception as error:
        write_json(arguments.output / 'status.json', {'status': 'FAILED', 'error': str(error)})
        raise
    lines = ['# Detector evaluation: ' + arguments.profile, '',
             '| Model/task | Images | mAP50 | mAP50-95 | P95 ms | FPS model call |',
             '|---|---:|---:|---:|---:|---:|']
    for report in reports:
        metric = report['metrics']
        speed = report['latency']
        lines.append(f"| {report['job']} | {report['images']} | {metric['metrics/mAP50(B)']:.4f} | "
                     f"{metric['metrics/mAP50-95(B)']:.4f} | {speed['p95_ms']:.2f} | {speed['fps_model_call']:.2f} |")
    lines += ['', 'Scope: ' + ('SMOKE SUBSET, NOT FINAL' if arguments.limit else 'FULL TEST SPLIT'),
              'Frame-call latency includes preprocess, inference, NMS, CPU transfer and class mapping.',
              'Camera, tracking, VLM, rendering and encoding excluded. No deployment realtime claim.']
    if arguments.profile == 'external':
        lines += [
              'OIV7 light scores are localization-only, not color classification.',
              'KASTEL direction labels pooled by color; red-yellow excluded and reported separately.',
              'Only exact Car/Motorcycle/Bus/Truck names mapped. No guessed vehicle subtype labels.',
              'No vehicle labels exist in KASTEL; vehicle detection accuracy is not applicable.',
              'No arrow-direction GT exists in current light dataset; direction accuracy not evaluated.']
    (arguments.output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(arguments.output / 'status.json', {'status': 'COMPLETE', 'scope': reports[0]['evaluation_scope']})


if __name__ == '__main__':
    main()
