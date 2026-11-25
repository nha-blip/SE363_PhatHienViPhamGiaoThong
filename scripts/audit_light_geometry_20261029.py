import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

from scripts.audit_yolo26_light_20261029 import DATA, NAMES, REPORT, ROOT, WEIGHTS
from scripts.evaluate_external_detectors import read_labels, write_json
from scripts.review_light_dataset import correction_focus


def read_predictions(path, width, height, confidence=0.001):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            identity, center_x, center_y, box_width, box_height, score = map(float, line.split())
            if score >= confidence:
                rows.append([(center_x-box_width/2)*width, (center_y-box_height/2)*height,
                             (center_x+box_width/2)*width, (center_y+box_height/2)*height, score, identity])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def geometry_candidates(boxes, identities, predicted):
    result = []
    if not len(boxes) or not len(predicted):
        return result
    overlaps = box_iou(torch.from_numpy(boxes), torch.from_numpy(predicted[:, :4])).numpy()
    for index, box in enumerate(boxes):
        for other in np.flatnonzero(predicted[:, 5] == identities[index]):
            candidate = predicted[other, :4]
            intersection = np.maximum(0, np.minimum(box[2:], candidate[2:])-np.maximum(box[:2], candidate[:2])).prod()
            gt_area = (box[2]-box[0])*(box[3]-box[1])
            area = (candidate[2]-candidate[0])*(candidate[3]-candidate[1])
            if (gt_area > 0 and intersection/gt_area >= 0.8 and area/gt_area >= 1.5
                    and 0.15 < overlaps[index, other] < 0.5):
                result.append({'gt_index': index, 'predicted_bbox': candidate.tolist(),
                               'iou': float(overlaps[index, other]), 'pred_area_to_gt': float(area/gt_area)})
                break
    return result


def source_name(path):
    name = Path(path).name.lower()
    return 'Bosch' if name.startswith('bosch') else 'S2TLD' if name.startswith('s2tld') else 'Kaggle_original'


def main():
    torch.set_num_threads(1)
    lookup = {}
    for path in (ROOT / 'data/TrafficLight363_expanded_candidate_20261026/val_combined.txt').read_text().splitlines():
        lookup[hashlib.sha256(Path(path).as_posix().encode()).hexdigest()[:24]] = path
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    metrics = {name: DetMetrics(names=NAMES) for name in ('raw', 'agnostic_nms_07', 'agnostic_nms_05')}
    per_source = {name: DetMetrics(names=NAMES) for name in ('Bosch', 'S2TLD', 'Kaggle_original')}
    source_counts = Counter()
    candidates = []
    for image_path in sorted((DATA / 'images/val').iterdir()):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        ids, boxes = read_labels(DATA / 'labels/val' / (image_path.stem+'.txt'), width, height)
        predicted = read_predictions(REPORT / 'light_640/labels' / (image_path.stem+'.txt'), width, height)
        source = source_name(lookup[image_path.stem])
        source_counts[source] += 1
        for mode, metric in [*metrics.items(), (source, per_source[source])]:
            selected = predicted
            if mode.startswith('agnostic') and len(predicted):
                threshold = 0.7 if mode.endswith('07') else 0.5
                keep = nms(torch.from_numpy(predicted[:, :4]), torch.from_numpy(predicted[:, 4]), threshold).numpy()
                selected = predicted[keep]
            iou = box_iou(torch.from_numpy(boxes), torch.from_numpy(selected[:, :4]))
            correct = BaseValidator.match_predictions(matcher, torch.from_numpy(selected[:, 5]),
                                                     torch.from_numpy(ids), iou).numpy()
            metric.update_stats({'tp': correct, 'conf': selected[:, 4], 'pred_cls': selected[:, 5],
                                 'target_cls': ids, 'target_img': np.unique(ids), 'im_name': image_path.name})
        focused = geometry_candidates(boxes, ids, predicted[predicted[:, 4] >= 0.25])
        for focus in focused:
            candidates.append({'split': 'val', 'source': source, 'source_image': str(image_path),
                               'source_label': str(DATA / 'labels/val' / (image_path.stem+'.txt')), **focus})
    metrics_result = {}
    for name, metric in {**metrics, **per_source}.items():
        metric.process(save_dir=REPORT, plot=False)
        metrics_result[name] = metric.results_dict
    write_json(REPORT / 'cached_postprocess_and_source.json', {'metrics': metrics_result, 'source_image_counts': dict(source_counts),
               'validation_geometry_candidates': candidates,
               'policy': 'Same cached conf>=.001 predictions, unchanged validation; NMS is an ablation, not a guaranteed fix'})
    manifest = json.loads((DATA / 'manifest.json').read_text())
    groups = defaultdict(list)
    for row in manifest:
        groups[source_name(row['source_image'])].append(row)
    chosen = []
    generator = random.Random(363)
    for rows in groups.values():
        chosen.extend(generator.sample(rows, min(200, len(rows))))
    model = YOLO(str(WEIGHTS / 'light/checkpoints/best.pt'))
    paths = [DATA / 'images/train' / (row['id']+'.jpg') for row in chosen]
    for index, prediction in enumerate(model.predict([str(path) for path in paths], imgsz=640, batch=4,
                                      device=0, quantize=16, conf=0.25, stream=True, verbose=False)):
        image_path = paths[index]
        image = prediction.orig_img
        label = DATA / 'labels/train' / (image_path.stem+'.txt')
        ids, boxes = read_labels(label, image.shape[1], image.shape[0])
        predicted = prediction.boxes.data.cpu().numpy()
        for focus in geometry_candidates(boxes, ids, predicted):
            candidates.append({'split': 'train', 'source': source_name(chosen[index]['source_image']),
                               'source_image': str(image_path), 'source_label': str(label), **focus})
    review = REPORT / 'geometry_review'
    review.mkdir(exist_ok=True)
    rows = []
    for split in ('train', 'val'):
        selected = [row for row in candidates if row['split'] == split]
        selected = selected[:16]
        for candidate in selected:
            image_path = Path(candidate['source_image'])
            image = cv2.imread(str(image_path))
            identities, boxes = read_labels(Path(candidate['source_label']), image.shape[1], image.shape[0])
            objects = [{'source_class': NAMES[int(identity)], 'proposed_class': NAMES[int(identity)],
                        'bbox_xyxy': box.tolist(), 'valid': True} for identity, box in zip(identities, boxes)]
            number = candidate['gt_index']+1
            before = copy.deepcopy(objects[number-1])
            identity = f'{split}_{image_path.stem}_{number}'
            row = {'id': identity, 'source_image': str(image_path), 'source_label': candidate['source_label'],
                   'source_label_sha256': hashlib.sha256(Path(candidate['source_label']).read_bytes()).hexdigest(),
                   'split': split, 'qa_image': str(review / (identity+'.jpg')),
                   'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED', 'class_mapping_confirmed': False,
                   'notes': '', 'objects': objects, 'allowed_classes': list(NAMES.values()),
                   'geometry_required': False, 'edit_object_numbers': [number], 'geometry_edits': [],
                   'task_hint': 'Kiem tra bbox chi om bong sang hay ca cum den. Du doan chi la goi y, khong phai GT. '
                                'Neu khong du bang chung, chon UNCERTAIN. Khong sua validation de lam dep diem.',
                   'model_geometry_suggestion_not_gt': candidate,
                   'correction': {'action': {'action': 'GEOMETRY_REVIEW', 'object_number': number},
                                  'before': before, 'after': copy.deepcopy(before)}}
            correction_focus(row).save(row['qa_image'], quality=98)
            rows.append(row)
    write_json(review / 'bbox_review.json', {'status': 'PENDING_HUMAN_QA', 'is_ground_truth': False,
               'allow_bbox_edit': True, 'coordinate_system': 'native_source_pixels',
               'current_dataset_modified': False, 'reviews': rows})
    write_json(REPORT / 'geometry_summary.json', {'train_images_sampled': len(paths),
               'candidate_counts': dict(Counter(row['split'] for row in candidates)),
               'candidate_source_counts': dict(Counter((row['split']+'_'+row['source']) for row in candidates)),
               'qa_rows': len(rows), 'labels_modified': False, 'confirmed_label_errors': False})
    print(json.dumps({'status': 'GEOMETRY_AND_POSTPROCESS_AUDIT_COMPLETE', 'qa_rows': len(rows),
                      'metrics': metrics_result}), flush=True)


if __name__ == '__main__':
    main()
