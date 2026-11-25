import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from scripts.evaluate_external_detectors import read_labels, sha256, write_json
from scripts.export_detector_errors import choose_errors, match_errors, render


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_light_diagnostic_20261029'
DATA = ROOT / 'data/Light363_fullscene_20261028'
WEIGHTS = ROOT / 'weights/yolo26_autorestart_20261028/cycle_00'
NAMES = {0: 'red', 1: 'yellow', 2: 'green', 3: 'off'}


def size_bucket(short_side):
    return '<4px' if short_side < 4 else '4-8px' if short_side < 8 else '8-16px' if short_side < 16 else '>=16px'


def predictions(path, width, height):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            identity, center_x, center_y, box_width, box_height, confidence = map(float, line.split())
            if confidence >= 0.25:
                rows.append([(center_x-box_width/2)*width, (center_y-box_height/2)*height,
                             (center_x+box_width/2)*width, (center_y+box_height/2)*height,
                             confidence, identity])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def summarize_errors(directory):
    buckets = defaultdict(Counter)
    classes = defaultdict(Counter)
    confusion = np.zeros((4, 4), dtype=int)
    records = []
    gt_overlap = Counter()
    for image_path in sorted((DATA / 'images/val').iterdir()):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        label = DATA / 'labels/val' / (image_path.stem+'.txt')
        identities, boxes = read_labels(label, width, height)
        predicted = predictions(directory / 'labels' / (image_path.stem+'.txt'), width, height)
        gt_status, pred_status, matches = match_errors(identities, boxes, predicted)
        for index, status in enumerate(gt_status):
            box = boxes[index]
            short = min(box[2]-box[0], box[3]-box[1])*min(640/width, 640/height)
            buckets[size_bucket(short)][status] += 1
            classes[NAMES[int(identities[index])]][status] += 1
        for match in matches:
            confusion[int(identities[match['gt_index']]), int(predicted[match['prediction_index'], 5])] += 1
        for index, box in enumerate(boxes):
            for other in range(index):
                intersection = np.maximum(0, np.minimum(box[2:], boxes[other, 2:])-np.maximum(box[:2], boxes[other, :2])).prod()
                union = (box[2]-box[0])*(box[3]-box[1])+(boxes[other, 2]-boxes[other, 0])*(boxes[other, 3]-boxes[other, 1])-intersection
                if union > 0 and intersection/union > 0.8:
                    gt_overlap['same_class' if identities[index] == identities[other] else 'different_class'] += 1
        if any(status != 'OK' for status in gt_status+pred_status):
            records.append({'source_image': str(image_path), 'names': {str(key): value for key, value in NAMES.items()},
                            'error_types': sorted(set(gt_status+pred_status)-{'OK'}),
                            'ground_truth': [{'class_id': int(identity), 'bbox_xyxy': box.tolist(), 'status': status}
                                             for identity, box, status in zip(identities, boxes, gt_status)],
                            'predictions': [{'class_id': int(row[5]), 'bbox_xyxy': row[:4].tolist(),
                                             'confidence': float(row[4]), 'status': status}
                                            for row, status in zip(predicted, pred_status)], 'matches': matches})
    def rates(counts):
        total = sum(counts.values())
        return {**dict(counts), 'total': total,
                'localized_recall_iou50': (counts['OK']+counts['WRONG'])/total if total else None,
                'correct_class_recall_iou50': counts['OK']/total if total else None}
    errors = directory / 'error_images'
    errors.mkdir(exist_ok=True)
    for index, record in enumerate(choose_errors(records, 24), 1):
        cv2.imwrite(str(errors / f'{index:03d}.jpg'), render(record), [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_json(errors / f'{index:03d}.json', record)
    result = {'policy': 'Fixed conf >=.25, greedy class-agnostic IoU >.5 matching; size buckets always at640 for paired comparison',
              'bbox_size_buckets': {key: rates(value) for key, value in buckets.items()},
              'per_class_errors': {key: rates(value) for key, value in classes.items()},
              'localized_confusion_gt_rows_pred_columns': confusion.tolist(),
              'gt_high_overlap_pairs': dict(gt_overlap), 'images_with_errors': len(records),
              'note': 'Mismatch and overlapping-GT flags are not confirmed annotation mistakes'}
    write_json(directory / 'error_analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[640, 1280])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for planned paired evaluation')
    REPORT.mkdir(parents=True, exist_ok=True)
    configuration = REPORT / 'data.yaml'
    configuration.write_text(yaml.safe_dump({'path': str(DATA), 'train': 'train_balanced.txt',
                                             'val': 'images/val', 'names': list(NAMES.values())}, sort_keys=False))
    report = {'status': 'RUNNING', 'validation_only': True, 'test_used': False, 'experiments': {}}
    weight = WEIGHTS / 'light/checkpoints/best.pt'
    report['light_weights_sha256'] = sha256(weight)
    for size in args.sizes:
        directory = REPORT / f'light_{size}'
        write_json(REPORT / 'status.json', {'status': 'VALIDATING', 'imgsz': size})
        model = YOLO(str(weight))
        if model.names != NAMES:
            raise ValueError('Light class mapping mismatch')
        metrics = model.val(data=str(configuration), split='val', imgsz=size, batch=8 if size == 640 else 2,
                            device=0, workers=0, quantize=16, conf=0.001, iou=0.7, max_det=300,
                            save_txt=True, save_conf=True, plots=True, project=str(REPORT),
                            name=f'light_{size}', exist_ok=False)
        class_metrics = []
        for index, class_id in enumerate(metrics.box.ap_class_index):
            precision, recall, ap50, ap = metrics.class_result(index)
            class_metrics.append({'class': NAMES[int(class_id)], 'precision': precision, 'recall': recall,
                                  'ap50': ap50, 'ap50_95': ap})
        result = {'metrics': metrics.results_dict, 'per_class': class_metrics,
                  'validation_speed_ms': metrics.speed, 'imgsz': size,
                  'not_a_batch1_end_to_end_realtime_benchmark': True}
        write_json(directory / 'metrics.json', result)
        result['errors'] = summarize_errors(directory)
        report['experiments'][str(size)] = result
        write_json(REPORT / 'report.json', report)
        del model
        torch.cuda.empty_cache()
    vehicle = YOLO(str(WEIGHTS / 'vehicle/checkpoints/best.pt'))
    if list(vehicle.names.values()) != ['car', 'motorcycle', 'bus', 'truck']:
        raise ValueError('Vehicle class mapping mismatch')
    images = sorted((ROOT / 'data/YOLO363_training_working/Vehicle363-v1/images/val').iterdir())
    selected = [images[index] for index in np.linspace(0, len(images)-1, 32, dtype=int)]
    counts = Counter()
    for result in vehicle.predict([str(path) for path in selected], imgsz=640, batch=4,
                                  device=0, quantize=16, conf=0.25, verbose=False, stream=True):
        counts.update(vehicle.names[int(value)] for value in result.boxes.cls.cpu().numpy())
    report['vehicle_smoke'] = {'images': len(selected), 'predicted_counts': dict(counts),
                               'status': 'MODEL_LOAD_AND_INFERENCE_OK', 'not_a_full_accuracy_evaluation': True}
    report['status'] = 'COMPLETE'
    write_json(REPORT / 'report.json', report)
    write_json(REPORT / 'status.json', {'status': 'COMPLETE'})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
