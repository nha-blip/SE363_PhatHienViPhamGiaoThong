import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.inventory_ihanoi_cameras import ADB, ROOT, camera_rows


CAPTURE = ROOT / 'data/iHanoi_capture'
OUTPUT = CAPTURE / 'city_survey_20261029'
SERIALS = ('emulator-5564', 'emulator-5584')
FFMPEG = ROOT / '.tools/ffmpeg_runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'


def center(bounds, minimum_height=60):
    values = [int(value) for value in re.findall(r'\d+', bounds or '')]
    if len(values) != 4:
        return None
    left, top, right, bottom = values
    if top < 284 or bottom > 1600 or bottom-top < minimum_height:
        return None
    return (left+right)//2, (top+bottom)//2


def next_section(raw, opened):
    for node in ET.fromstring(raw).iter('node'):
        name = node.get('content-desc', '')
        if (name.startswith(('Phường ', 'Xã ')) and '\n' not in name
                and name not in opened and node.get('clickable') == 'true'):
            point = center(node.get('bounds'))
            if point:
                return name, point
    return None


class Survey:
    def __init__(self, resume=False, serials=SERIALS):
        if (len(serials) != 2 or len(set(serials)) != 2
                or any(not re.fullmatch(r'emulator-\d+', serial) for serial in serials)):
            raise ValueError('Two distinct emulator serials required')
        self.serials = tuple(serials)
        OUTPUT.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self.records = {}
        self.workers = {}
        self.prior_workers = {}
        self.status = 'SURVEY_RUNNING'
        index = OUTPUT / 'survey_index.json'
        if index.exists():
            previous = json.loads(index.read_text(encoding='utf-8'))
            if not resume or previous['status'] == 'SURVEY_RUNNING':
                raise ValueError('Resume requires a safely ended prior survey; refusing duplicate writer')
            self.records = {row['camera_id']: row for row in previous['cameras']}
            if any(row['status'] == 'CLAIMED_ONCE' for row in self.records.values()):
                raise ValueError('Unfinished claim needs inspection before resume')
            if (OUTPUT / 'STOP').exists():
                raise ValueError('STOP still present; archive the safe-stop request before resuming')
            self.prior_workers = previous.get('workers', {})
            self.save()
            return
        if resume:
            raise ValueError('No existing survey to resume')
        source = json.loads((CAPTURE / 'city_inventory_20261029/cameras.json').read_text(encoding='utf-8'))
        for row in source['cameras']:
            self.records[row['camera_id']] = {**row, 'status': 'WAITING_SURVEY'}
        catalog = json.loads((CAPTURE / 'capture_catalog.json').read_text(encoding='utf-8'))
        for row in catalog['videos']:
            identity = row['camera_id']
            if identity not in self.records:
                continue
            if row.get('path') and (CAPTURE / row['path']).is_file():
                self.records[identity].update(status='EXISTING_REVIEW_VIDEO',
                    video=str((CAPTURE / row['path']).relative_to(ROOT)), existing_review=row)
            elif row.get('status') == 'REJECT':
                self.records[identity].update(status='PREVIOUSLY_REJECTED_BY_USER', existing_review=row)
        self.save()

    def save(self):
        with self.lock:
            snapshot = {'status': self.status, 'updated_at_local': datetime.now(
                        timezone(timedelta(hours=7))).isoformat(),
                        'concurrent_instance_limit': 2, 'seconds_per_attempt': 25,
                        'automatic_reopen': False, 'all_city_coverage_verified': False,
                        'is_ground_truth': False, 'selection': 'USER_WILL_REVIEW',
                        'workers': copy.deepcopy(self.workers),
                        'cameras': copy.deepcopy(list(self.records.values()))}
            temporary = OUTPUT / 'survey_index.tmp'
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(OUTPUT / 'survey_index.json')
            lines = ['# Camera khảo sát để người dùng chọn', '',
                     'Chỉ mục này cập nhật theo tiến độ; chưa phải danh sách video đạt chuẩn.', '',
                     '| Camera | Tên / vị trí | Trạng thái | Video |', '|---|---|---|---|']
            for row in snapshot['cameras']:
                path = row.get('video')
                link = ('[Mở video](../../../' + path + ')') if path else '—'
                lines.append(f"| {row['camera_id']} | {row.get('name', '').replace('|', '/')} | {row['status']} | {link} |")
            temporary = OUTPUT / 'REVIEW_CAMERAS.tmp'
            temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            temporary.replace(OUTPUT / 'REVIEW_CAMERAS.md')

    def adb(self, serial, *arguments):
        result = subprocess.run([str(ADB), '-s', serial, *arguments], capture_output=True, timeout=40)
        if result.returncode:
            raise RuntimeError('ADB operation failed')
        return result.stdout

    def dump(self, serial):
        for attempt in range(3):
            result = self.adb(serial, 'shell', 'uiautomator', 'dump', '/sdcard/ihanoi_survey.xml')
            if b'ERROR' not in result:
                raw = self.adb(serial, 'shell', 'cat', '/sdcard/ihanoi_survey.xml')
                if list(ET.fromstring(raw).iter('node')):
                    return raw
            time.sleep(1)
        raise ValueError('UI hierarchy unavailable after three read attempts')

    def listing(self, serial):
        for attempt in range(3):
            raw = self.dump(serial)
            try:
                return raw, camera_rows(raw)
            except ValueError:
                if attempt == 2:
                    raise
                time.sleep(1)

    def blocked(self):
        return ((OUTPUT / 'STOP').exists() or time.monotonic()-self.started > 7200
                or shutil.disk_usage(OUTPUT).free < 5*1024**3)

    def claim(self, rows):
        with self.lock:
            for identity, row in rows.items():
                if identity not in self.records:
                    self.records[identity] = {**row, 'status': 'WAITING_SURVEY'}
                point = center(row.get('ui_bounds'))
                if point and self.records[identity]['status'] == 'WAITING_SURVEY':
                    self.records[identity]['status'] = 'CLAIMED_ONCE'
                    return identity, point
        return None

    def record(self, serial, identity, point):
        folder = OUTPUT / re.sub(r'[^A-Za-z0-9_-]', '_', identity)
        folder.mkdir(exist_ok=False)
        remote = '/sdcard/ihanoi_survey_' + folder.name + '.mp4'
        path = folder / 'screen.mp4'
        with (folder / 'record.log').open('wb') as log:
            process = subprocess.Popen([str(ADB), '-s', serial, 'shell', 'screenrecord',
                       '--bit-rate', '3000000', '--time-limit', '25', remote], stdout=log,
                       stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                time.sleep(0.7)
                self.adb(serial, 'shell', 'input', 'tap', *map(str, point))
                time.sleep(2)
                raw = self.dump(serial)
                (folder / 'opened.xml').write_bytes(raw)
                verified = any(identity in node.get('content-desc', '')
                               for node in ET.fromstring(raw).iter('node'))
                if verified:
                    self.adb(serial, 'shell', 'input', 'tap', '850', '685')
                code = process.wait(timeout=45)
            finally:
                if process.poll() is None:
                    process.wait(timeout=45)
        self.adb(serial, 'pull', remote, str(path))
        self.adb(serial, 'shell', 'input', 'tap', '50', '100')
        import cv2

        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        previews = []
        for second in (10, 20):
            preview = folder / f'preview_{second}s.jpg'
            subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-ss', str(second),
                            '-i', str(path), '-frames:v', '1', '-q:v', '2', '-y', str(preview)],
                           capture_output=True, timeout=30, check=True)
            if preview.is_file():
                previews.append(str(preview.relative_to(ROOT)))
        with self.lock:
            row = self.records[identity]
            row.update(status='RECORDED_AWAITING_USER_REVIEW' if verified and code == 0 and count > 1
                       else 'FAILED_CAMERA_OR_RECORDING_CHECK', video=str(path.relative_to(ROOT)),
                       camera_identity_verified=verified, record_return_code=code,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                       frames=count, recording_fps=fps, duration_seconds=count/fps if fps else None,
                       previews=previews, no_automatic_quality_approval=True)
        self.save()

    def worker(self, serial):
        opened = set(self.prior_workers.get(serial, {}).get('sections_opened', ['Phường Ba Đình']))
        signatures = set()
        repeats = 0
        try:
            for step in range(400):
                if self.blocked():
                    break
                raw, rows = self.listing(serial)
                (OUTPUT / f'{serial}_current.xml').write_bytes(raw)
                target = self.claim(rows)
                if target:
                    identity, point = target
                    with self.lock:
                        self.workers[serial] = {'status': 'RECORDING', 'camera_id': identity, 'step': step,
                                                'sections_opened': sorted(opened)}
                    self.save()
                    print(json.dumps({'serial': serial, 'recording': identity}), flush=True)
                    self.record(serial, identity, point)
                    repeats = 0
                    continue
                section = next_section(raw, opened)
                if section:
                    name, point = section
                    opened.add(name)
                    self.adb(serial, 'shell', 'input', 'tap', *map(str, point))
                    time.sleep(0.7)
                    continue
                signature = tuple((node.get('content-desc'), node.get('bounds'))
                                  for node in ET.fromstring(raw).iter('node') if node.get('content-desc'))
                repeats = repeats+1 if signature in signatures else 0
                signatures.add(signature)
                if repeats >= 3:
                    break
                self.adb(serial, 'shell', 'input', 'swipe', '450', '1450', '450', '450', '450')
                time.sleep(0.7)
            with self.lock:
                self.workers[serial] = {'status': 'SCAN_ENDED_REVIEW_COVERAGE', 'sections_opened': sorted(opened)}
        except Exception as error:
            with self.lock:
                active = self.workers.get(serial, {}).get('camera_id')
                if active and self.records[active]['status'] == 'CLAIMED_ONCE':
                    self.records[active].update(status='ATTEMPT_FAILED_NO_RETRY', error=str(error))
                self.workers[serial] = {'status': 'STOPPED_SAFELY', 'reason': type(error).__name__ + ': ' + str(error),
                                        'sections_opened': sorted(opened)}
        self.save()

    def run(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(self.worker, self.serials))
        self.status = 'SURVEY_ENDED_REQUIRES_USER_SELECTION'
        self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--serials', nargs=2, default=SERIALS)
    arguments = parser.parse_args()
    Survey(resume=arguments.resume, serials=arguments.serials).run()
