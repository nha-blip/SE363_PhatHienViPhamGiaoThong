import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_external_detectors import LIGHT_NAMES, ROOT, VEHICLE_NAMES, read_labels, sha256, write_json
from scripts.export_detector_errors import match_errors
from scripts.render_external_detector_qa import collect_qa_pairs, panel, select_pairs


COLORS = {'OK': (70, 200, 70), 'WRONG': (255, 0, 255), 'MISSED': (0, 220, 255),
          'EXTRA': (0, 0, 255), 'UNSCORED': (180, 180, 180)}


def build_record(cached, report, image):
    localization = report['job'].endswith('_localization')
    vehicle = report['job'].endswith('_vehicle')
    names = VEHICLE_NAMES if vehicle else LIGHT_NAMES
    gt_ids, gt_boxes = read_labels(Path(cached['label']), image.shape[1], image.shape[0])
    scoring_ids = np.zeros_like(gt_ids) if localization else gt_ids
    predictions = np.asarray(cached['predictions'], dtype=np.float32).reshape(-1, 6)
    gt_status, pred_status, matches = match_errors(scoring_ids, gt_boxes, predictions)
    mapping = {int(key): value for key, value in report['mapping']['mapped_source_ids'].items()}
    native_names = {int(key): value for key, value in report['mapping']['source_names'].items()}
    native_lookup = {}
    unscored = []
    for row in cached['native_predictions']:
        native_id = int(row[5])
        if native_id in mapping:
            native_lookup[(tuple(np.asarray(row[:5], dtype=np.float32)), mapping[native_id])] = native_names[native_id]
        else:
            unscored.append({'bbox_xyxy': row[:4], 'confidence': row[4],
                             'label': native_names[native_id], 'status': 'UNSCORED'})
    ground_truth = [{'class_id': int(identity), 'label': names[int(identity)], 'bbox_xyxy': box.tolist(), 'status': status}
                    for identity, box, status in zip(gt_ids, gt_boxes, gt_status)]
    predicted = []
    for row, status in zip(predictions, pred_status):
        identity = int(row[5])
        label = native_lookup.get((tuple(row[:5]), identity))
        if label is None:
            raise ValueError('Mapped/native cached predictions disagree')
        predicted.append({'class_id': identity, 'label': label, 'confidence': float(row[4]),
                          'bbox_xyxy': row[:4].tolist(), 'status': status})
    return {'job': report['job'], 'localization_only': localization, 'source_image': cached['image'],
            'source_label': cached['label'], 'source_label_sha256': sha256(Path(cached['label'])),
            'native_size': [image.shape[1], image.shape[0]], 'ground_truth': ground_truth,
            'predictions': predicted, 'unscored_predictions': unscored, 'matches': matches}


def focus_pairs(record, count=3):
    matched = {entry['gt_index']: entry['prediction_index'] for entry in record['matches']}
    candidates = []
    for index, box in enumerate(record['ground_truth']):
        predicted_index = matched.get(index)
        prediction = record['predictions'][predicted_index] if predicted_index is not None else None
        candidates.append({'gt_index': index, 'prediction_index': predicted_index,
                           'gt': box, 'prediction': prediction, 'status': box['status']})
    for index, box in enumerate(record['predictions']):
        if box['status'] == 'EXTRA':
            candidates.append({'gt_index': None, 'prediction_index': index,
                               'gt': None, 'prediction': box, 'status': 'EXTRA'})
    priority = {'WRONG': 0, 'MISSED': 1, 'EXTRA': 2, 'OK': 3}

    def key(focus):
        box = (focus['gt'] or focus['prediction'])['bbox_xyxy']
        return priority[focus['status']], (box[2] - box[0]) * (box[3] - box[1])

    return sorted(candidates, key=key)[:count]


def box_on_panel(canvas, annotation, original_size, bounds=None, text=False):
    if annotation is None:
        return
    original_width, original_height = original_size
    bounds = bounds or (0, 0, original_width, original_height)
    crop_left, crop_top, crop_right, crop_bottom = bounds
    width, height = crop_right - crop_left, crop_bottom - crop_top
    scale = min(canvas.shape[1] / width, (canvas.shape[0] - 42) / height)
    offset_x = (canvas.shape[1] - round(width * scale)) // 2
    offset_y = 42 + (canvas.shape[0] - 42 - round(height * scale)) // 2
    left, top, right, bottom = annotation['bbox_xyxy']
    first = (offset_x + round((left - crop_left) * scale), offset_y + round((top - crop_top) * scale))
    second = (offset_x + round((right - crop_left) * scale), offset_y + round((bottom - crop_top) * scale))
    color = COLORS[annotation['status']]
    cv2.rectangle(canvas, first, second, color, 1)
    if text:
        label = annotation['label']
        if 'confidence' in annotation:
            label += f" {annotation['confidence']:.2f}"
        cv2.putText(canvas, label, (max(0, first[0]), max(55, first[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def focus_title(annotation, side):
    if annotation is None:
        return side + ': NO MATCH'
    title = side + ': ' + annotation['label']
    if side == 'PRED':
        title += f" {annotation['confidence']:.2f}"
    return title


def render(record, image):
    original_size = (image.shape[1], image.shape[0])
    left_panel = panel(image, 'LEFT: GT | yellow=MISSED; magenta=class disagreement')
    right_panel = panel(image, 'RIGHT: PRED | red=EXTRA; green=OK; gray=UNSCORED')
    for box in record['ground_truth']:
        box_on_panel(left_panel, box, original_size, text=True)
    for box in record['predictions'] + record['unscored_predictions']:
        box_on_panel(right_panel, box, original_size, text=True)
    header = np.full((64, 1920, 3), 25, dtype=np.uint8)
    cv2.putText(header, record['job'] + ' | ' + Path(record['source_image']).name[:130],
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    note = ('LOCALIZATION ONLY: GT colors are reference; predicted generic light does NOT identify color.'
            if record['localization_only'] else
            'Conf > .25, matching IoU > .50. Status is model/GT disagreement, NOT proof GT is wrong.')
    cv2.putText(header, note, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
    crop_row = np.full((232, 1920, 3), 25, dtype=np.uint8)
    focuses = focus_pairs(record)
    for index, focus in enumerate(focuses):
        left, top, right, bottom = (focus['gt'] or focus['prediction'])['bbox_xyxy']
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        half_width, half_height = max(24, right - left), max(24, bottom - top)
        bounds = (max(0, int(center_x - half_width)), max(0, int(center_y - half_height)),
                  min(image.shape[1], int(center_x + half_width)), min(image.shape[0], int(center_y + half_height)))
        crop = image[bounds[1]:bounds[3], bounds[0]:bounds[2]]
        gt_crop = panel(crop, focus_title(focus['gt'], 'GT'), 320, 190)
        pred_crop = panel(crop, focus_title(focus['prediction'], 'PRED'), 320, 190)
        box_on_panel(gt_crop, focus['gt'], original_size, bounds)
        box_on_panel(pred_crop, focus['prediction'], original_size, bounds)
        crop_row[:, index * 640:(index + 1) * 640] = np.hstack((gt_crop, pred_crop))
    return np.vstack((header, np.hstack((left_panel, right_panel)), crop_row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, default=ROOT / 'runs/project363/yolo_candidates_eval_20261021/full')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError('count must be positive')
    evaluation = json.loads((args.evaluation / 'report.json').read_text(encoding='utf-8'))
    if evaluation['status'] != 'COMPLETE':
        raise ValueError('Evaluation is not complete')
    args.output.mkdir(parents=True, exist_ok=False)
    samples = {dataset: select_pairs(collect_qa_pairs(dataset, args.count), args.count)
               for dataset in ('vehicle', 'traffic_light')}
    manifest = {'status': 'RUNNING', 'scope': 'SHARED_SAMPLED_IMAGES_NOT_ALL_TEST_IMAGES',
                'confidence': .25, 'matching_iou': .5, 'inference_rerun': False,
                'selection': 'Same class/background-stratified sample per dataset for all compatible model jobs; not error-only.',
                'jobs': []}
    write_json(args.output / 'manifest.json', manifest)
    for report in evaluation['jobs']:
        identity = report['job']
        dataset = 'vehicle' if identity.endswith('_vehicle') else 'traffic_light'
        selected = {str(path): position for position, (path, _) in enumerate(samples[dataset], 1)}
        cached = {}
        cache_path = args.evaluation / (identity + '.predictions_025.jsonl')
        line_count = 0
        seen = set()
        with cache_path.open(encoding='utf-8') as handle:
            for line in handle:
                entry = json.loads(line)
                line_count += 1
                if entry['image'] in seen:
                    raise ValueError('Duplicate image in prediction cache')
                seen.add(entry['image'])
                if entry['image'] in selected:
                    cached[entry['image']] = entry
        if line_count != report['images'] or set(cached) != set(selected):
            raise ValueError(f'Incomplete prediction cache: {identity}')
        directory = args.output / identity
        directory.mkdir()
        entries = []
        for image_path, label_path in samples[dataset]:
            entry = cached[str(image_path)]
            if Path(entry['label']) != label_path:
                raise ValueError('Cached label path mismatch')
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(image_path)
            record = build_record(entry, report, image)
            stem = f'{selected[str(image_path)]:04d}'
            destination = directory / (stem + '.jpg')
            if not cv2.imwrite(str(destination), render(record, image), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError(destination)
            write_json(directory / (stem + '.json'), record)
            entries.append({'comparison_image': destination.name, 'source_image': str(image_path),
                            'gt_boxes': len(record['ground_truth']), 'predicted_boxes': len(record['predictions'])})
        index = {'job': identity, 'images': entries, 'checkpoint_sha256': report['weights_sha256'],
                 'prediction_cache_sha256': sha256(cache_path)}
        write_json(directory / 'index.json', index)
        manifest['jobs'].append({'job': identity, 'count': len(entries)})
        write_json(args.output / 'manifest.json', manifest)
        write_json(args.output / 'status.json', {'status': 'RENDERING', 'completed_jobs': len(manifest['jobs']),
                   'total_jobs': len(evaluation['jobs'])})
        print(f'{identity}: {len(entries)} JPGs', flush=True)
    manifest['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'status.json', {'status': 'COMPLETE', 'images': sum(job['count'] for job in manifest['jobs'])})


if __name__ == '__main__':
    main()
