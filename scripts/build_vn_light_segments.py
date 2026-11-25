import argparse
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .prepare_vn_light_candidates import NAMES, box_iou, review_row, scene_band
else:
    from prepare_vn_light_candidates import NAMES, box_iou, review_row, scene_band


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/VN_light_candidates_20261030'
MATCH_SCORE = 0.55
MERGE_IOU = 0.3
SEARCH_MARGIN = 40
MAXIMUM_GAP_SAMPLES = 6
HUE_RANGES = {'red': ((0, 8), (170, 180)), 'yellow': ((9, 35),), 'green': ((40, 100),)}


def lit_state(crop, saturation=80, value=110, minimum_pixels=3, minimum_share=0.012, dominance=0.6):
    """Estimate the lit color of one signal head; returns 'unknown' instead of guessing from weak evidence."""
    import cv2
    import numpy as np

    if crop.size == 0:
        return 'unknown'
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 1] >= saturation) & (hsv[..., 2] >= value)
    counts = {}
    for name, ranges in HUE_RANGES.items():
        mask = np.zeros(bright.shape, bool)
        for low, high in ranges:
            mask |= (hsv[..., 0] >= low) & (hsv[..., 0] <= high)
        counts[name] = int((mask & bright).sum())
    best = max(counts, key=counts.get)
    total = sum(counts.values())
    if counts[best] < max(minimum_pixels, minimum_share * crop.shape[0] * crop.shape[1]) or counts[best] < dominance * total:
        return 'unknown'
    return best


def state_segments(observations, maximum_gap=MAXIMUM_GAP_SAMPLES):
    """Group ordered (frame, state) rows by color across short uncertain gaps; uncertain frames themselves stay unlabeled."""
    segments = []
    gap = 0
    for frame, state in observations:
        if state in ('unknown', 'lost'):
            gap += 1
            continue
        if segments and segments[-1]['state'] == state and gap <= maximum_gap:
            segments[-1]['frames'].append(frame)
        else:
            segments.append({'state': state, 'frames': [frame]})
        gap = 0
    return segments


def group_heads(tracks, minimum_iou=MERGE_IOU):
    """Merge reviewed boxes that follow the same physical head; a human-drawn box stays the representative."""
    ordered = sorted(tracks, key=lambda row: (not row['drawn'], row['id']))
    groups = []
    for track in ordered:
        for group in groups:
            shared = sorted(set(group['boxes']) & set(track['boxes']))
            if not shared:
                continue
            values = sorted(box_iou(group['boxes'][frame], track['boxes'][frame]) for frame in shared)
            if values[len(values) // 2] >= minimum_iou:
                group['merged_rows'].append(track['id'])
                break
        else:
            groups.append({**track, 'merged_rows': [track['id']]})
    return groups


def resolve_rows(reviews):
    """Read reviewer intent without editing the review: only KEEP boxes are heads; other rows stay recorded with their note."""
    kept, rejected = [], []
    for row in reviews:
        if row['decision'] == 'KEEP' and row['objects']:
            kept.append(row)
        else:
            rejected.append({'id': row['id'], 'camera': row['camera'], 'decision': row['decision'], 'notes': row.get('notes', '')})
    return kept, rejected


def read_scenes(clip, band, stride_seconds):
    import cv2

    capture = cv2.VideoCapture(str(clip))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * stride_seconds))
    scenes, index = {}, 0
    while capture.grab():
        if index % step == 0:
            decoded, frame = capture.retrieve()
            if decoded:
                found = scene_band(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean(axis=1))
                if found and abs(found[0] - band[0]) <= 16 and abs(found[1] - band[1]) <= 16:
                    scenes[index] = frame[band[0]:band[1]]
        index += 1
    capture.release()
    return scenes, fps


def track_box(scenes, reference, box):
    """Follow one head near its camera-shift position; a low score means occlusion or camera change, not a position."""
    import cv2
    import numpy as np

    left, top, right, bottom = box
    pad = max(12, round(max(right - left, bottom - top) * 0.5))
    height, width = reference.shape[:2]
    x0, y0 = max(0, int(left) - pad), max(0, int(top) - pad)
    x1, y1 = min(width, int(right) + pad + 1), min(height, int(bottom) + pad + 1)
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    template = gray[y0:y1, x0:x1]
    base = np.float32(gray)
    boxes = {}
    for frame, scene in scenes.items():
        current = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        if current.shape != gray.shape:
            continue
        (shift_x, shift_y), _ = cv2.phaseCorrelate(base, np.float32(current))
        wx0 = max(0, x0 + round(shift_x) - SEARCH_MARGIN)
        wy0 = max(0, y0 + round(shift_y) - SEARCH_MARGIN)
        wx1 = min(width, x1 + round(shift_x) + SEARCH_MARGIN)
        wy1 = min(height, y1 + round(shift_y) + SEARCH_MARGIN)
        if wx1 - wx0 < template.shape[1] or wy1 - wy0 < template.shape[0]:
            continue
        result = cv2.matchTemplate(current[wy0:wy1, wx0:wx1], template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score >= MATCH_SCORE:
            dx, dy = wx0 + location[0] - x0, wy0 + location[1] - y0
            boxes[frame] = (left + dx, top + dy, right + dx, bottom + dy)
    return boxes


def main():
    import cv2

    parser = argparse.ArgumentParser(description='Propagate reviewed heads and propose color segments; never GT')
    parser.add_argument('--stride-seconds', type=float, default=0.5)
    parser.add_argument('--revision', default='segments_revision_001')
    args = parser.parse_args()
    review = json.loads((SOURCE / 'review/bbox_review.json').read_text(encoding='utf-8'))
    manifest = {row['camera']: row for row in json.loads((SOURCE / 'scan_manifest.json').read_text(encoding='utf-8'))['cameras']}
    kept, rejected = resolve_rows(review['reviews'])
    output = SOURCE / args.revision
    output.mkdir(exist_ok=False)
    for name in ('frames', 'overview', 'review'):
        (output / name).mkdir()
    rows, report = [], []
    for camera in sorted({row['camera'] for row in kept}):
        record = manifest[camera]
        scenes, fps = read_scenes(record['clip'], record['scene_rows'], args.stride_seconds)
        tracks = []
        for row in (item for item in kept if item['camera'] == camera):
            reference = cv2.imread(row['source_image'])
            box = row['objects'][0]['bbox_xyxy']
            tracks.append({'id': row['id'], 'drawn': row['geometry_required'] or bool(row.get('geometry_edits')),
                           'box': box, 'label': row['objects'][0]['proposed_class'], 'reference': row['source_image'],
                           'boxes': track_box(scenes, reference, box)})
        heads = group_heads(tracks)
        safe = camera.replace('.', '_')
        overview = next(iter(scenes.values())).copy()
        for number, head in enumerate(heads, 1):
            reference = cv2.imread(head['reference'])
            left, top, right, bottom = map(int, head['box'])
            reference_state = lit_state(reference[top:bottom, left:right])
            observations = []
            for frame in sorted(scenes):
                box = head['boxes'].get(frame)
                if box is None:
                    observations.append((frame, 'lost'))
                    continue
                x0, y0, x1, y1 = map(int, box)
                observations.append((frame, lit_state(scenes[frame][max(0, y0):y1, max(0, x0):x1])))
            segments = state_segments(observations)
            counts = Counter(state for _, state in observations)
            report.append({'camera': camera, 'head': number, 'rows': head['merged_rows'], 'box': head['box'],
                           'human_label': head['label'], 'color_at_reviewed_frame': reference_state,
                           'observations': dict(counts), 'segments': [[row['state'], len(row['frames'])] for row in segments]})
            first = next((head['boxes'][frame] for frame in sorted(head['boxes'])), head['box'])
            cv2.rectangle(overview, tuple(map(int, first[:2])), tuple(map(int, first[2:])), (0, 255, 255), 2)
            cv2.putText(overview, str(number), (int(first[0]), max(14, int(first[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            for position, segment in enumerate(segments, 1):
                frame = segment['frames'][len(segment['frames']) // 2]
                image = output / 'frames' / f'{safe}_{frame:06d}.jpg'
                if not image.exists():
                    cv2.imwrite(str(image), scenes[frame], [cv2.IMWRITE_JPEG_QUALITY, 95])
                row = review_row(f'{safe}_den{number}_doan{position}', camera, image, image, list(head['boxes'][frame]),
                                 segment['state'], frame / fps,
                                 {'head': number, 'segment_frames': segment['frames'], 'sampled_stride_seconds': args.stride_seconds,
                                  'proposal': 'template tracking + HSV color, not GT'})
                row['task_hint'] = (f'Đèn số {number} của camera này, đoạn {position}. Màu đề xuất áp dụng cho {len(segment["frames"])} frame. '
                                    'Nếu đúng màu đang sáng và khung nằm đúng đèn: chọn Đúng + Cả cụm đèn + Xác nhận nhãn. '
                                    'Sai màu: chọn lại lớp. Khung lệch khỏi đèn hoặc đèn bị che: Cần sửa và ghi chú.')
                rows.append(row)
        cv2.imwrite(str(output / 'overview' / f'{safe}.jpg'), overview, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (output / 'review/bbox_review.json').write_text(json.dumps(
        {'status': 'PENDING_HUMAN_QA', 'is_ground_truth': False, 'allow_bbox_edit': True, 'allow_class_edit': True,
         'allow_focus_removal': False, 'coordinate_system': 'native_source_pixels', 'current_dataset_modified': False,
         'reviews': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(
        {'status': 'PROPOSALS_NOT_GT', 'stride_seconds': args.stride_seconds, 'heads': report,
         'non_kept_rows': rejected, 'names': NAMES}, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'MO_QA_MAU_DEN.cmd').write_text(
        '@echo off\r\ncd /d "%~dp0..\\..\\.."\r\n"C:\\Users\\NHA\\miniconda3\\envs\\my_env\\python.exe" -m scripts.review_light_dataset '
        '--review "%~dp0review\\bbox_review.json"\r\nif errorlevel 1 pause\r\n', encoding='ascii')
    print(json.dumps({'heads': len(report), 'segment_rows': len(rows)}), flush=True)
    for row in report:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
