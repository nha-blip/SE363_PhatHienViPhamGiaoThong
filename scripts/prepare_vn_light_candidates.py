import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / 'data/iHanoi_capture'
OUTPUT = ROOT / 'data/VN_light_candidates_20261030'
WEIGHTS = ROOT / 'weights/yolo26_autorestart_20261028/cycle_00/light/checkpoints/best.pt'
NAMES = ['red', 'yellow', 'green', 'off']
EXTRA_CLIPS = {'C069.10-PTZ': CAPTURE / 'retry_pair_20261029_131120/C069_10_PTZ.mp4',
               'C057.08-PTZ': CAPTURE / 'retry_pair_20261029_131120/C057_08_PTZ.mp4'}
PRIORITY = {'THEO_DOI_DEN_VACH': 0, 'CAN_XEM_LAI': 1}
PROPOSAL_CONF = 0.2
CLUSTER_IOU = 0.3
FROZEN_DIFFERENCE = 0.5


def scene_band(brightness, threshold=12, gap=3, minimum=200):
    """Return the longest bright row run as (top, bottom), ignoring app bars and black padding."""
    best, start, last = None, None, None
    for row, value in enumerate(list(brightness) + [0] * (gap + 1)):
        if value > threshold:
            if start is None:
                start = row
            last = row
        elif start is not None and row - last > gap:
            if best is None or last + 1 - start > best[1] - best[0]:
                best = (start, last + 1)
            start = None
    return best if best and best[1] - best[0] >= minimum else None


def stable_band(bands, height, tolerance=8, maximum_share=0.75, minimum_frames=3):
    """Use the camera view shared by most frames; loading and map pages have another layout and reach the screen bottom."""
    usable = {frame: band for frame, band in bands.items()
              if band and band[1] - band[0] <= height * maximum_share and band[1] <= height - 20}
    if not usable:
        return None, []
    votes = Counter((band[0] // tolerance, band[1] // tolerance) for band in usable.values())
    (top, bottom), _ = votes.most_common(1)[0]
    matched = {frame: band for frame, band in usable.items()
               if abs(band[0] // tolerance - top) <= 1 and abs(band[1] // tolerance - bottom) <= 1}
    if len(matched) < minimum_frames:
        return None, []
    tops = sorted(band[0] for band in matched.values())
    bottoms = sorted(band[1] for band in matched.values())
    return (tops[len(tops) // 2], bottoms[len(bottoms) // 2]), sorted(matched)


def box_iou(first, second):
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    area = (first[2]-first[0]) * (first[3]-first[1]) + (second[2]-second[0]) * (second[3]-second[1]) - overlap
    return overlap / area if area > 0 else 0.0


def cluster_detections(detections, iou_threshold=CLUSTER_IOU):
    """Group (frame, box, confidence, class) rows of one fixed camera into distinct signal heads."""
    clusters = []
    for frame, box, confidence, identity in sorted(detections, key=lambda row: -row[2]):
        match = next((row for row in clusters if box_iou(row['box'], box) >= iou_threshold), None)
        if match is None:
            match = {'box': list(box), 'confidence': confidence, 'frame': frame, 'class': identity,
                     'frames': set(), 'classes': Counter()}
            clusters.append(match)
        match['frames'].add(frame)
        match['classes'][identity] += 1
    return clusters


def select_clusters(clusters, top=3):
    kept = [row for row in clusters if len(row['frames']) >= 3 or row['confidence'] >= 0.3]
    return sorted(kept, key=lambda row: (-len(row['frames']), -row['confidence']))[:top]


def review_row(identity, camera, image, qa_image, box, class_name, time_seconds, details, width=None, height=None):
    """Build one row of the existing local QA tool; a missing box asks the reviewer to draw a head."""
    placeholder = box is None
    if placeholder:
        box = [5.0, 5.0, float(width - 5), float(height - 5)]
    item = {'source_class': class_name or 'chưa có đề xuất', 'proposed_class': class_name,
            'bbox_xyxy': [float(value) for value in box], 'valid': True}
    hint = ('Vẽ khung cho đèn tín hiệu xe mà model bỏ sót, chọn màu đang sáng rồi chọn Đúng. '
            'Nếu không còn đèn nào: bấm Bỏ bbox thừa.' if placeholder else
            'Đề xuất của model, không phải GT. Bbox phải bao cả cụm đèn; không gồm bộ đếm giây. '
            'Sửa khung hoặc màu nếu sai. Không phải đèn tín hiệu xe: bấm Bỏ bbox thừa. Nhìn không rõ: Không chắc.')
    return {'id': identity, 'camera': camera, 'source_image': str(image), 'qa_image': str(qa_image), 'split': 'unassigned',
            'source_time_s': round(time_seconds, 3), 'timebase_verified': False, 'decision': 'PENDING',
            'bbox_scope': 'UNREVIEWED', 'class_mapping_confirmed': False, 'notes': '', 'objects': [item],
            'allowed_classes': NAMES, 'geometry_required': placeholder, 'edit_object_numbers': [1], 'geometry_edits': [],
            'task_hint': hint, 'model_suggestion_not_gt': details,
            'correction': {'action': {'action': 'GEOMETRY_REVIEW', 'object_number': 1},
                           'before': dict(item), 'after': dict(item)}}


def camera_clips(choices):
    """Keep reviewed cameras only: never bare recursive scans or skipped cameras. Black video is judged from content."""
    clips, skipped = [], []
    for camera, choice in choices.items():
        priority = PRIORITY.get(choice['choice'])
        if priority is None:
            continue
        clip = EXTRA_CLIPS.get(camera) or CAPTURE / 'city_survey_20261029' / camera.replace('.', '_', 1) / 'screen.mp4'
        if clip.is_file():
            clips.append((priority, camera, clip))
        else:
            skipped.append({'camera': camera, 'reason': 'NO_CLIP'})
    return sorted(clips, key=lambda row: (row[0], row[1])), skipped


def scan_clip(model, clip, stride_seconds):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * stride_seconds))
    frames = []
    index = 0
    while True:
        if not capture.grab():
            break
        if index % step == 0:
            decoded, frame = capture.retrieve()
            if decoded:
                frames.append((index, frame))
        index += 1
    capture.release()
    if not frames:
        return {'status': 'UNREADABLE'}, [], None, fps
    bands = {number: scene_band(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean(axis=1)) for number, frame in frames}
    band, retained = stable_band(bands, frames[0][1].shape[0])
    if band is None:
        return {'status': 'NO_SCENE', 'frames': total}, [], None, fps
    top, bottom = band
    scenes = [(number, frame[top:bottom]) for number, frame in frames if number in set(retained)]
    difference = max((float(np.abs(scenes[position][1].astype(int) - scenes[0][1].astype(int)).mean())
                      for position in range(1, len(scenes))), default=0.0)
    if difference < FROZEN_DIFFERENCE:
        return {'status': 'FROZEN', 'frames': total, 'difference': difference}, [], None, fps
    detections = []
    for number, scene in scenes:
        result = model.predict(scene, imgsz=640, conf=PROPOSAL_CONF, device='cpu', verbose=False)[0]
        for row in result.boxes.data.cpu().tolist():
            detections.append((number, tuple(row[:4]), row[4], int(row[5])))
    summary = {'status': 'OK', 'frames': total, 'fps_screen_recording_not_camera': fps, 'scene_rows': [top, bottom],
               'sampled_frames': len(scenes), 'dropped_other_layout_frames': len(frames) - len(scenes),
               'detections': len(detections), 'difference': difference}
    return summary, detections, scenes, fps


def write_outputs(camera, clusters, scenes, fps, width_height, folder):
    import cv2

    scene_lookup = dict(scenes)
    middle = scenes[len(scenes) // 2][0]
    rows = []
    safe = camera.replace('.', '_')
    selected = select_clusters(clusters)
    for number, cluster in enumerate(selected + [None], 1):
        frame = cluster['frame'] if cluster else middle
        scene = scene_lookup[frame]
        name = f'{safe}_{frame:06d}'
        image = folder / 'frames' / (name + '.jpg')
        if not image.exists():
            cv2.imwrite(str(image), scene, [cv2.IMWRITE_JPEG_QUALITY, 95])
        qa = folder / 'qa' / f'{safe}_{number}.jpg'
        overview = scene.copy()
        for other in selected:
            left, top, right, bottom = map(int, other['box'])
            color = (0, 255, 255) if other is cluster else (255, 255, 0)
            cv2.rectangle(overview, (left, top), (right, bottom), color, 2)
        cv2.imwrite(str(qa), overview, [cv2.IMWRITE_JPEG_QUALITY, 90])
        details = None
        if cluster:
            details = {'confidence': round(cluster['confidence'], 3), 'support_frames': len(cluster['frames']),
                       'sampled_frames': len(scenes), 'class_votes': {NAMES[key]: value for key, value in cluster['classes'].items()}}
        rows.append(review_row(f'{safe}_{number}', camera, image, qa, cluster['box'] if cluster else None,
                               NAMES[cluster['class']] if cluster else None, frame / fps, details,
                               width=scene.shape[1], height=scene.shape[0]))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Propose Vietnam traffic-light heads from reviewed cameras; never GT')
    parser.add_argument('--stride-seconds', type=float, default=1.0)
    args = parser.parse_args()
    from ultralytics import YOLO

    choices = json.loads((CAPTURE / 'city_survey_20261029/human_choices.json').read_text(encoding='utf-8'))
    clips, skipped = camera_clips(choices)
    OUTPUT.mkdir(exist_ok=False)
    for name in ('frames', 'qa', 'review'):
        (OUTPUT / name).mkdir()
    model = YOLO(str(WEIGHTS))
    reviews, manifest = [], []
    for priority, camera, clip in clips:
        summary, detections, scenes, fps = scan_clip(model, clip, args.stride_seconds)
        record = {'camera': camera, 'priority': priority, 'clip': str(clip),
                  'clip_sha256': hashlib.sha256(clip.read_bytes()).hexdigest(), **summary}
        if summary['status'] == 'OK':
            clusters = cluster_detections(detections)
            rows = write_outputs(camera, clusters, scenes, fps, None, OUTPUT)
            reviews.extend(rows)
            record.update(proposed_heads=len(rows) - 1, clusters_found=len(clusters),
                          class_votes=dict(Counter(NAMES[row[3]] for row in detections)))
        manifest.append(record)
        print(json.dumps({key: record.get(key) for key in ('camera', 'status', 'sampled_frames', 'detections', 'proposed_heads')}), flush=True)
    (OUTPUT / 'review/bbox_review.json').write_text(json.dumps(
        {'status': 'PENDING_HUMAN_QA', 'is_ground_truth': False, 'allow_bbox_edit': True, 'allow_class_edit': True,
         'allow_focus_removal': True, 'coordinate_system': 'native_source_pixels', 'current_dataset_modified': False,
         'reviews': reviews}, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / 'scan_manifest.json').write_text(json.dumps(
        {'status': 'PROPOSALS_NOT_GT', 'weights': str(WEIGHTS), 'stride_seconds': args.stride_seconds,
         'timebase_verified': False, 'cameras': manifest, 'skipped': skipped}, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT / 'MO_QA_DEN_VN.cmd').write_text(
        '@echo off\r\ncd /d "%~dp0..\\.."\r\n"C:\\Users\\NHA\\miniconda3\\envs\\my_env\\python.exe" -m scripts.review_light_dataset '
        '--review "%~dp0review\\bbox_review.json"\r\nif errorlevel 1 pause\r\n', encoding='ascii')
    print(json.dumps({'cameras': len(manifest), 'rows': len(reviews), 'skipped': len(skipped),
                      'status_counts': dict(Counter(row['status'] for row in manifest))}), flush=True)


if __name__ == '__main__':
    main()
