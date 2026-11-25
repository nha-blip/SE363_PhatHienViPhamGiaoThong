import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from scripts.survey_ihanoi_cameras import CAPTURE, OUTPUT, ROOT


REPORT = OUTPUT / 'black_filter'


def camera_statistics(frame):
    height, width = frame.shape[:2]
    if (width, height) != (900, 1600):
        return None
    roi = frame[int(height*0.35):int(height*0.75), :, :][::8, ::8]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return {'mean': float(gray.mean()), 'dark_fraction': float(np.mean(gray <= 24)),
            'bright_fraction': float(np.mean(gray >= 100))}


def black_frame(frame):
    stats = camera_statistics(frame)
    return (stats is not None and stats['mean'] <= 5 and stats['dark_fraction'] >= 0.99
            and stats['bright_fraction'] < 0.0005)


def inspect(row):
    path = (ROOT / row['video']).resolve()
    if not path.is_relative_to(CAPTURE.resolve()) or not path.is_file():
        raise ValueError('Video missing or outside capture workspace')
    result = {'camera_id': row['camera_id'], 'video': row['video'], 'result': 'MANUAL_REVIEW',
              'initial_seconds_ignored': 5, 'roi_relative_xyxy': [0, 0.35, 1, 0.75]}
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or frames < 10:
            result['reason'] = 'Unreadable or insufficient frames'
            return result
        duration = frames/fps
        if duration < 7:
            result['reason'] = 'Too short for conservative black-screen check'
            return result
        result['duration_seconds'] = duration
        samples = []
        for second in np.linspace(5, duration-0.3, 9):
            capture.set(cv2.CAP_PROP_POS_MSEC, second*1000)
            valid, frame = capture.read()
            stats = camera_statistics(frame) if valid else None
            samples.append({'second': float(second), 'statistics': stats})
            if stats is None:
                result.update(reason='Unsupported layout or failed sampled decode', samples=samples)
                return result
            if not black_frame(frame):
                result.update(result='KEEP_NONBLACK_CONTENT', samples=samples)
                return result
        result['samples'] = samples
    finally:
        capture.release()
    capture = cv2.VideoCapture(str(path))
    checked = 0
    decoded = 0
    try:
        while True:
            valid, frame = capture.read()
            if not valid:
                break
            decoded += 1
            seconds = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if seconds < 5:
                continue
            checked += 1
            if not black_frame(frame):
                result.update(result='KEEP_NONBLACK_CONTENT', full_scan_checked_frames=checked)
                return result
    finally:
        capture.release()
    if decoded != int(frames) or checked < 10:
        result.update(reason='Incomplete decode or insufficient post-start frames', decoded_frames=decoded,
                      expected_frames=int(frames), full_scan_checked_frames=checked)
        return result
    result.update(result='BLACK_SCREEN_AFTER_STARTUP', decoded_frames=decoded,
                  full_scan_checked_frames=checked,
                  reason='Every decoded frame after the initial five seconds met conservative black ROI thresholds')
    return result


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    REPORT.mkdir(exist_ok=True)
    index = OUTPUT / 'survey_index.json'
    original = index.read_bytes()
    state = json.loads(original)
    if state['status'] == 'SURVEY_RUNNING':
        raise ValueError('Survey still active; refusing concurrent index changes')
    if state.get('black_filter_applied'):
        raise ValueError('Filter already applied; inspect the saved report')
    videos = [row for row in state['cameras'] if row.get('video')]
    report = {'status': 'ANALYSING', 'total_videos': len(videos), 'results': [],
              'permanent_deletion': False, 'thresholds': {'gray_dark_max': 24, 'mean_max': 5,
              'dark_fraction_min': 0.99, 'bright_gray_min': 100, 'bright_fraction_max': 0.0005}}
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(inspect, row): row for row in videos}
        for future in as_completed(pending):
            result = future.result()
            report['results'].append(result)
            report['summary'] = dict(Counter(row['result'] for row in report['results']))
            write(REPORT / 'report.json', report)
            print(json.dumps({'processed': len(report['results']), 'total': len(videos),
                              'camera': result['camera_id'], 'result': result['result']}), flush=True)
    report['status'] = 'ANALYSIS_COMPLETE'
    write(REPORT / 'report.json', report)
    if not args.apply:
        return
    if index.read_bytes() != original:
        raise ValueError('Survey index changed during analysis; refusing apply')
    backup = REPORT / 'survey_index.before_black_filter.json'
    if backup.exists():
        raise ValueError('Existing backup; inspect before retrying apply')
    backup.write_bytes(original)
    results = {row['camera_id']: row for row in report['results']}
    modified = copy.deepcopy(state)
    for row in modified['cameras']:
        result = results.get(row['camera_id'])
        if result is None:
            continue
        row['black_filter'] = result['result']
        if result['result'] != 'BLACK_SCREEN_AFTER_STARTUP':
            continue
        source = (ROOT / row['video']).resolve()
        if not source.is_relative_to(OUTPUT.resolve()) or source.suffix.lower() != '.mp4':
            row['black_filter'] = 'BLACK_EXTERNAL_REFERENCE_MANUAL_REVIEW'
            continue
        target = (OUTPUT / 'excluded_black' / source.parent.name / source.name).resolve()
        if not target.is_relative_to((OUTPUT / 'excluded_black').resolve()) or target.exists():
            raise ValueError('Unexpected quarantine target; refusing overwrite')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if row.get('sha256') and digest != row['sha256']:
            raise ValueError('Video hash changed before quarantine')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Quarantined video hash mismatch')
        row['status_before_black_filter'] = row['status']
        row['status'] = 'EXCLUDED_BLACK_SCREEN'
        row['video_before_black_filter'] = row['video']
        row['video'] = str(target.relative_to(ROOT))
        row['excluded_from_review_default'] = True
        result['quarantined_video'] = row['video']
    modified['black_filter_applied'] = True
    modified['black_filter_report'] = str((REPORT / 'report.json').relative_to(ROOT))
    write(index, modified)
    report['status'] = 'APPLIED_NONDESTRUCTIVE_QUARANTINE'
    write(REPORT / 'report.json', report)
    print(json.dumps({'status': report['status'], 'summary': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
