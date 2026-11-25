import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_external_detectors import (
    LIGHT_NAMES, ROOT, TRAINED_JOBS, VEHICLE_NAMES, read_labels, sha256, write_json,
)
from scripts.render_external_detector_qa import collect_qa_pairs, panel, select_pairs, zoom_panel


COLORS = {'OK': (70, 200, 70), 'WRONG': (255, 0, 255),
          'MISSED': (0, 220, 255), 'EXTRA': (0, 0, 255)}


def match_errors(gt_ids, gt_boxes, predictions):
    predicted_boxes = predictions[:, :4]
    intersection_size = np.maximum(0, np.minimum(gt_boxes[:, None, 2:], predicted_boxes[None, :, 2:]) -
                                   np.maximum(gt_boxes[:, None, :2], predicted_boxes[None, :, :2]))
    intersection = intersection_size.prod(axis=2)
    gt_area = np.maximum(0, gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(axis=1)
    pred_area = np.maximum(0, predicted_boxes[:, 2:] - predicted_boxes[:, :2]).prod(axis=1)
    overlap = intersection / (gt_area[:, None] + pred_area[None, :] - intersection + 1e-7)
    gt_index, pred_index = np.where(overlap > 0.5)
    matches = np.column_stack((gt_index, pred_index, overlap[gt_index, pred_index]))
    if len(matches) > 1:
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    gt_status = ['MISSED'] * len(gt_ids)
    pred_status = ['EXTRA'] * len(predictions)
    details = []
    for gt_identity, pred_identity, overlap_value in matches:
        gt_identity, pred_identity = int(gt_identity), int(pred_identity)
        status = 'OK' if gt_ids[gt_identity] == predictions[pred_identity, 5] else 'WRONG'
        gt_status[gt_identity] = status
        pred_status[pred_identity] = status
        details.append({'gt_index': gt_identity, 'prediction_index': pred_identity,
                        'iou': float(overlap_value), 'status': status})
    return gt_status, pred_status, details


def choose_errors(records, count):
    selected = []
    seen = set()
    for status in ('WRONG', 'MISSED', 'EXTRA'):
        candidates = [index for index, record in enumerate(records) if status in record['error_types']]
        quota = max(1, count // 3)
        for position in np.linspace(0, len(candidates) - 1, min(quota, len(candidates)), dtype=int):
            index = candidates[position]
            if index not in seen:
                selected.append(records[index])
                seen.add(index)
            if len(selected) >= count:
                return selected
    for index, record in enumerate(records):
        if len(selected) >= count:
            break
        if index not in seen:
            selected.append(record)
            seen.add(index)
    return selected


def annotations(record, side):
    names = record['names']
    result = []
    for box in record[side]:
        label = f"{box['status']} {names[str(box['class_id'])]}"
        if 'confidence' in box:
            label += f" {box['confidence']:.2f}"
        result.append({**box, 'label': label, 'scored': True})
    return result


def draw(image, boxes):
    canvas = image.copy()
    for box in boxes:
        left, top, right, bottom = [int(round(value)) for value in box['bbox_xyxy']]
        color = COLORS[box['status']]
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.putText(canvas, box['label'], (max(0, left), max(15, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def render(record):
    image = cv2.imread(record['source_image'])
    if image is None:
        raise ValueError(record['source_image'])
    ground_truth = annotations(record, 'ground_truth')
    predicted = annotations(record, 'predictions')
    board = np.hstack((panel(draw(image, ground_truth), 'LEFT: GT | MISSED=yellow, WRONG=magenta'),
                       panel(draw(image, predicted), 'RIGHT: predictions | EXTRA=red, WRONG=magenta')))
    header = np.full((64, board.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(header, record['source_image'].replace('\\', '/').split('/')[-1][:150],
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(header, 'Conf > 0.25, IoU > 0.50. Automatic mismatch flags, NOT confirmed GT label errors.',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    error_boxes = [box for box in ground_truth if box['status'] != 'OK']
    error_boxes += [box for box in predicted if box['status'] == 'EXTRA']
    priority = {'WRONG': 0, 'MISSED': 1, 'EXTRA': 2}
    error_boxes.sort(key=lambda box: (priority[box['status']],
                     (box['bbox_xyxy'][2] - box['bbox_xyxy'][0]) *
                     (box['bbox_xyxy'][3] - box['bbox_xyxy'][1])))
    crops = np.full((232, 1920, 3), 25, dtype=np.uint8)
    for index, box in enumerate(error_boxes[:3]):
        left, top, right, bottom = box['bbox_xyxy']
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        half_width, half_height = max(24, right - left), max(24, bottom - top)
        bounds = (max(0, int(center_x - half_width)), max(0, int(center_y - half_height)),
                  min(image.shape[1], int(center_x + half_width)), min(image.shape[0], int(center_y + half_height)))
        display_gt = [{**annotation, 'class_id': list(COLORS).index(annotation['status'])} for annotation in ground_truth]
        display_pred = [{**annotation, 'class_id': list(COLORS).index(annotation['status'])} for annotation in predicted]
        from scripts import render_external_detector_qa as rendering
        original_colors = rendering.COLORS
        rendering.COLORS = tuple(COLORS.values())
        try:
            gt_names = sorted({record['names'][str(annotation['class_id'])] for annotation in ground_truth
                               if annotation['bbox_xyxy'][2] > bounds[0] and annotation['bbox_xyxy'][0] < bounds[2]
                               and annotation['bbox_xyxy'][3] > bounds[1] and annotation['bbox_xyxy'][1] < bounds[3]})
            pred_names = sorted({record['names'][str(annotation['class_id'])] for annotation in predicted
                                 if annotation['bbox_xyxy'][2] > bounds[0] and annotation['bbox_xyxy'][0] < bounds[2]
                                 and annotation['bbox_xyxy'][3] > bounds[1] and annotation['bbox_xyxy'][1] < bounds[3]})
            crops[:, index * 640:(index + 1) * 640] = np.hstack((
                zoom_panel(image, display_gt, bounds, 'GT: ' + ','.join(gt_names)),
                zoom_panel(image, display_pred, bounds, 'PRED: ' + (','.join(pred_names) or 'NONE'))))
        finally:
            rendering.COLORS = original_colors
    return np.vstack((header, board, crops))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--vehicle-candidates', type=int, default=500)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    if args.count < 1 or args.vehicle_candidates < 1:
        raise ValueError('Counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(1)
    manifest = {'status': 'RUNNING', 'scope': 'SAMPLED_ERROR_QA_NOT_ALL_TEST_ERRORS', 'jobs': []}
    write_json(args.output / 'manifest.json', manifest)
    try:
        for identity, filename, dataset, mode in TRAINED_JOBS:
            names = VEHICLE_NAMES if mode == 'vehicle' else LIGHT_NAMES
            directory = args.output / identity
            directory.mkdir()
            report = json.loads((ROOT / 'runs/project363/trained_yolo_eval_20261020/full_gpu' /
                                 identity / 'report.json').read_text())
            if sha256(Path(filename)) != report['weights_sha256']:
                raise ValueError('Weights differ from tested checkpoint')
            model = YOLO(filename)
            if model.names != names:
                raise ValueError('Unexpected checkpoint namespace')
            if mode == 'vehicle':
                pairs = select_pairs(collect_qa_pairs(dataset, max(1, args.vehicle_candidates // 5)),
                                     args.vehicle_candidates)
            else:
                pairs = collect_qa_pairs(dataset, 0)
            errors = []
            for index, (image_path, label_path) in enumerate(pairs, 1):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(image_path)
                gt_ids, gt_boxes = read_labels(label_path, image.shape[1], image.shape[0])
                prediction = model.predict(image, imgsz=640, conf=0.001, iou=0.7, max_det=300,
                                           device=args.device, quantize=32 if args.device == 'cpu' else 16,
                                           verbose=False, save=False)[0].boxes.data.detach().cpu().numpy()
                prediction = prediction[prediction[:, 4] > 0.25]
                gt_status, pred_status, matches = match_errors(gt_ids, gt_boxes, prediction)
                types = sorted(set(gt_status + pred_status) - {'OK'})
                if types:
                    errors.append({'source_image': str(image_path), 'source_label': str(label_path),
                                   'source_label_sha256': sha256(label_path),
                                   'names': {str(key): value for key, value in names.items()},
                                   'error_types': types, 'matches': matches,
                                   'ground_truth': [{'class_id': int(class_id), 'bbox_xyxy': box.tolist(), 'status': status}
                                                    for class_id, box, status in zip(gt_ids, gt_boxes, gt_status)],
                                   'predictions': [{'class_id': int(row[5]), 'bbox_xyxy': row[:4].tolist(),
                                                    'confidence': float(row[4]), 'status': status}
                                                   for row, status in zip(prediction, pred_status)]})
                if index == 1 or index % 100 == 0 or index == len(pairs):
                    write_json(args.output / 'status.json', {'status': 'SCANNING', 'job': identity,
                               'processed': index, 'total': len(pairs), 'error_images': len(errors)})
                    print(f'{identity}: {index}/{len(pairs)}; error images={len(errors)}', flush=True)
            selected = choose_errors(errors, args.count)
            entries = []
            for index, record in enumerate(selected, 1):
                stem = f'{index:04d}_' + '_'.join(record['error_types'])
                if not cv2.imwrite(str(directory / f'{stem}.jpg'), render(record), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise OSError('Cannot save QA image')
                write_json(directory / f'{stem}.json', record)
                entries.append({'image': f'{stem}.jpg', 'source_image': record['source_image'],
                                'error_types': record['error_types']})
            job = {'job': identity, 'weights_sha256': report['weights_sha256'], 'scanned': len(pairs),
                   'candidate_images_with_errors': len(errors), 'exported': len(selected), 'images': entries}
            write_json(directory / 'index.json', job)
            manifest['jobs'].append({key: value for key, value in job.items() if key != 'images'})
            write_json(args.output / 'manifest.json', manifest)
            print(f'{identity}: exported {len(selected)} comparisons', flush=True)
            del model
            if args.device != 'cpu':
                torch.cuda.empty_cache()
    except Exception as error:
        write_json(args.output / 'status.json', {'status': 'FAILED', 'error': str(error)})
        raise
    manifest['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'status.json', {'status': 'COMPLETE'})


if __name__ == '__main__':
    main()
