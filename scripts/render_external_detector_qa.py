import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from scripts.evaluate_external_detectors import (
    JOBS, LIGHT_NAMES, ROOT, VEHICLE_NAMES, class_mapping, read_labels, sha256,
    write_json,
)


COLORS = ((80, 190, 255), (0, 220, 220), (80, 220, 80), (230, 150, 100))


def select_pairs(pairs, count):
    if not count or count >= len(pairs):
        return pairs
    groups = {identity: [] for identity in range(4)}
    groups['background'] = []
    for index, (_, label) in enumerate(pairs):
        identities = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
        for identity in identities:
            groups[identity].append(index)
        if not identities:
            groups['background'].append(index)
    selected = set()
    quota = max(1, count // 5)
    for indices in groups.values():
        if indices:
            positions = np.linspace(0, len(indices) - 1, min(quota, len(indices)), dtype=int)
            selected.update(indices[position] for position in positions)
    for index in np.linspace(0, len(pairs) - 1, len(pairs), dtype=int):
        if len(selected) >= count:
            break
        selected.add(int(index))
    return [pairs[index] for index in sorted(selected)[:count]]


def collect_qa_pairs(dataset, count):
    import yaml

    settings = yaml.safe_load((ROOT / f'config/{dataset}_dataset.yaml').read_text())
    images_root = Path(settings['path']) / settings['test']
    labels_root = Path(settings['path']) / 'labels/test'
    images = []
    for folder, _, filenames in os.walk(images_root):
        images.extend(Path(folder) / filename for filename in filenames
                      if Path(filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
    images.sort()
    if not images:
        raise ValueError('No test images')
    if count:
        candidate_count = min(len(images), max(512, count * 5))
        indices = np.linspace(0, len(images) - 1, candidate_count, dtype=int)
        images = [images[index] for index in indices]
    return [(image, labels_root / image.relative_to(images_root).with_suffix('.txt')) for image in images]


def draw_boxes(image, annotations):
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thickness = max(1, round(max(width, height) / 800))
    font_scale = max(0.35, min(0.65, width / 1800))
    for annotation in annotations:
        box = annotation['bbox_xyxy']
        left = int(np.clip(round(box[0]), 0, width - 1))
        top = int(np.clip(round(box[1]), 0, height - 1))
        right = int(np.clip(round(box[2]), 0, width - 1))
        bottom = int(np.clip(round(box[3]), 0, height - 1))
        color = COLORS[annotation['class_id'] % len(COLORS)] if annotation['scored'] else (0, 140, 255)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
        label = annotation['label']
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_left = max(0, min(left, width - text_size[0] - 4))
        text_top = max(0, top - text_size[1] - baseline - 5)
        cv2.rectangle(canvas, (text_left, text_top),
                      (min(width - 1, text_left + text_size[0] + 4),
                       min(height - 1, text_top + text_size[1] + baseline + 4)), (15, 15, 15), -1)
        cv2.putText(canvas, label, (text_left + 2, text_top + text_size[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return canvas


def panel(image, title, width=960, height=640):
    canvas = np.full((height + 42, width, 3), 20, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left = (width - size[0]) // 2
    top = 42 + (height - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized
    cv2.putText(canvas, title[:100], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def zoom_panel(image, annotations, bounds, title):
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right]
    canvas = panel(crop, title, 320, 190)
    scale = min(320 / crop.shape[1], 190 / crop.shape[0])
    resized_width = max(1, round(crop.shape[1] * scale))
    resized_height = max(1, round(crop.shape[0] * scale))
    offset_x = (320 - resized_width) // 2
    offset_y = 42 + (190 - resized_height) // 2
    for annotation in annotations:
        box_left, box_top, box_right, box_bottom = annotation['bbox_xyxy']
        if box_right <= left or box_left >= right or box_bottom <= top or box_top >= bottom:
            continue
        point_left = offset_x + round((max(left, box_left) - left) * scale)
        point_top = offset_y + round((max(top, box_top) - top) * scale)
        point_right = offset_x + round((min(right, box_right) - left) * scale)
        point_bottom = offset_y + round((min(bottom, box_bottom) - top) * scale)
        color = COLORS[annotation['class_id'] % len(COLORS)] if annotation['scored'] else (0, 140, 255)
        cv2.rectangle(canvas, (point_left, point_top), (point_right, point_bottom), color, 2)
    return canvas


def compose(image, ground_truth, predictions, title, mode):
    gt_image = draw_boxes(image, ground_truth)
    predicted_image = draw_boxes(image, predictions)
    board = np.hstack((panel(gt_image, f'LEFT: GROUND TRUTH ({len(ground_truth)} boxes)'),
                       panel(predicted_image, f'RIGHT: PREDICTIONS >= 0.25 ({len(predictions)} boxes)')))
    header = np.full((64, board.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, title[:170], (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    note = 'Color labels scored; arrow directions not scored; orange text marks unsupported red-yellow.'
    if mode == 'localization':
        note = 'LOCALIZATION ONLY. Original GT colors shown for reference; OIV7 does not predict light color.'
    elif mode == 'vehicle':
        note = 'Scored classes: exact Car/Motorcycle/Bus/Truck only. No guessed subtype remapping.'
    cv2.putText(header, note, (12, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)
    parts = [header, board]
    if ground_truth:
        crop_row = np.full((232, board.shape[1], 3), 25, dtype=np.uint8)
        smallest = sorted(ground_truth, key=lambda item:
                          (item['bbox_xyxy'][2] - item['bbox_xyxy'][0]) *
                          (item['bbox_xyxy'][3] - item['bbox_xyxy'][1]))[:3]
        for index, annotation in enumerate(smallest):
            left, top, right, bottom = annotation['bbox_xyxy']
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            half_width, half_height = max(24, right - left), max(24, bottom - top)
            left, right = max(0, int(center_x - half_width)), min(image.shape[1], int(center_x + half_width))
            top, bottom = max(0, int(center_y - half_height)), min(image.shape[0], int(center_y + half_height))
            bounds = (left, top, right, bottom)
            gt_crop = zoom_panel(image, ground_truth, bounds, f'GT zoom {index + 1}')
            pred_crop = zoom_panel(image, predictions, bounds, f'PRED zoom {index + 1}')
            crop_row[:, index * 640:(index + 1) * 640] = np.hstack((gt_crop, pred_crop))
        parts.append(crop_row)
    return np.vstack(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--device', default='0')
    parser.add_argument('--redraw-only', action='store_true')
    args = parser.parse_args()
    if args.count < 0:
        raise ValueError('count must be nonnegative; zero means the full test set')
    if args.redraw_only:
        for identity, _, _, mode in JOBS:
            directory = args.output / identity
            for path in sorted(directory.glob('*.json')):
                if path.stem == 'index':
                    continue
                record = json.loads(path.read_text(encoding='utf-8'))
                image = cv2.imread(record['source_image'])
                if image is None:
                    raise ValueError(record['source_image'])
                title = f"{identity} | {path.stem} | {Path(record['source_image']).name}"
                board = compose(image, record['ground_truth'], record['predictions'], title, mode)
                if not cv2.imwrite(str(directory / record['comparison_image']), board,
                                   [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise OSError('Cannot redraw QA image')
            print(f'Redraw complete: {identity}', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(1)
    manifest = {'status': 'RUNNING', 'confidence': 0.25, 'imgsz': 640,
                'selection': 'GT-class/background strata from up to max(512,count*5) evenly spaced candidates; not a new accuracy benchmark',
                'layout': 'GT left; predictions right; paired crops below', 'jobs': []}
    for identity, filename, dataset, mode in JOBS:
        directory = args.output / identity
        directory.mkdir()
        weights = ROOT / 'weights/external_eval_20261019' / filename
        report_path = ROOT / 'runs/project363/external_detector_eval_20261019/full_gpu' / identity / 'report.json'
        report = json.loads(report_path.read_text())
        if sha256(weights) != report['weights_sha256']:
            raise ValueError('Checkpoint changed since evaluation')
        model = YOLO(str(weights))
        mapping, excluded = class_mapping(model.names, mode)
        pairs = select_pairs(collect_qa_pairs(dataset, args.count), args.count)
        names = VEHICLE_NAMES if mode == 'vehicle' else LIGHT_NAMES
        job = {'job': identity, 'model_sha256': report['weights_sha256'],
               'mapped_source_ids': mapping, 'excluded_red_yellow': excluded, 'images': []}
        for index, (image_path, label_path) in enumerate(pairs, 1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(image_path)
            height, width = image.shape[:2]
            gt_ids, gt_boxes = read_labels(label_path, width, height)
            ground_truth = [{'class_id': int(class_id), 'label': names[int(class_id)],
                             'bbox_xyxy': box.tolist(), 'scored': True}
                            for class_id, box in zip(gt_ids, gt_boxes)]
            prediction = model.predict(image, imgsz=640, conf=0.25, iou=0.7, max_det=300,
                                       classes=list(mapping) + list(excluded),
                                       device=args.device, quantize=32 if args.device == 'cpu' else 16,
                                       verbose=False, save=False)[0]
            annotations = []
            for row in prediction.boxes.data.detach().cpu().tolist():
                source_id = int(row[5])
                target_id = mapping.get(source_id)
                source_name = model.names[source_id]
                target_name = ('traffic_light' if mode == 'localization' else names[target_id]) if target_id is not None else 'UNSCORED'
                annotations.append({'source_id': source_id, 'source_name': source_name,
                                    'class_id': target_id if target_id is not None else source_id,
                                    'mapped_name': target_name, 'confidence': row[4],
                                    'bbox_xyxy': row[:4], 'scored': target_id is not None,
                                    'label': f'{source_name}->{target_name} {row[4]:.2f}'})
            filename = f'{index:04d}.jpg'
            title = f'{identity} | {index}/{len(pairs)} | {image_path.name}'
            board = compose(image, ground_truth, annotations, title, mode)
            if not cv2.imwrite(str(directory / filename), board, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError('Cannot save QA image')
            record = {'comparison_image': filename, 'source_image': str(image_path),
                      'source_label': str(label_path), 'source_label_sha256': sha256(label_path),
                      'native_size': [width, height], 'ground_truth': ground_truth,
                      'predictions': annotations}
            write_json(directory / f'{index:04d}.json', record)
            job['images'].append({'comparison_image': filename, 'source_image': str(image_path),
                                  'ground_truth_boxes': len(ground_truth), 'predicted_boxes': len(annotations)})
            if index == 1 or index % 20 == 0 or index == len(pairs):
                print(f'{identity}: {index}/{len(pairs)}', flush=True)
                write_json(args.output / 'status.json', {'status': 'RENDERING', 'job': identity,
                                                         'processed': index, 'total': len(pairs)})
        write_json(directory / 'index.json', job)
        manifest['jobs'].append({'job': identity, 'count': len(pairs)})
        del model
        if args.device != 'cpu':
            torch.cuda.empty_cache()
    manifest['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'status.json', {'status': 'COMPLETE'})
    text = '''# Anh QA: GT va du doan

- Mo cac JPG truc tiep bang Photos hoac image viewer, khong can web.
- Moi thu muc model co 100 anh lay mau theo lop GT/background; khong phai toan bo test set.
- Ben trai: GT goc. Ben phai: du doan confidence >= 0.25. Phia duoi: crop GT/PRED cung vung.
- OIV7 light chi danh gia dinh vi; mau GT ben trai chi de tham chieu.
- KASTEL giu ten nhan goc va mau da map. UNSCORED la red-yellow khong co lop GT tuong ung.
- OIV7 vehicle chi hien cac lop exact Car/Motorcycle/Bus/Truck theo protocol test.
- Khong thay confidence .25 bang confidence floor .001 dung de tinh AP; anh nay khong hien moi prediction cua duong PR.
- Kiem tra bbox om dung vat the, bbox den om ca cum hay chi bong den, doi mau sai, bo sot va bbox thua.
- JSON cung so ghi anh/label nguon, bbox native, nhan goc/mapped va confidence. index.json lien ket cac anh trong moi thu muc.
- Chi tao anh minh hoa, khong sua GT, dataset hay ket qua benchmark truoc do.
'''
    (args.output / 'README_QA.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
