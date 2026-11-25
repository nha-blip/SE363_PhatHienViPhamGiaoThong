import argparse
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from scripts.inventory_ihanoi_cameras import ADB, ROOT, camera_rows
from scripts.survey_ihanoi_cameras import center, next_section


CAPTURE = ROOT / 'data/iHanoi_capture'
SERIAL = 'emulator-5554'
SCRCPY = ROOT / '.tools/scrcpy/scrcpy-win64-v4.1/scrcpy.exe'


def phone_record_command(serial, destination, seconds):
    return [str(SCRCPY), '--serial', serial, '--no-window', '--no-playback', '--no-control',
            '--no-audio', '--video-bit-rate=6M', '--max-fps=30',
            f'--time-limit={seconds}', f'--record={destination}']


def remaining_camera_ids(choices, previous=None, retry_failed=False):
    wanted = {identity for identity, row in choices.items()
              if row['choice'] in ('THEO_DOI_DEN_VACH', 'CAN_XEM_LAI')}
    rows = (previous or {}).get('cameras', {})
    excluded = {identity for identity, row in rows.items()
                if not retry_failed or row.get('status') == 'PENDING_HUMAN_QA'}
    return wanted - excluded


def image_problem(frame):
    if frame is None or frame.size == 0:
        return 'UNREADABLE_FRAME'
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray < 24)) > 0.97:
        return 'BLACK_FRAME'
    channels = frame.astype(np.int16)
    neutral = channels.max(axis=2) - channels.min(axis=2) < 8
    gray_fill = neutral & (gray > 45) & (gray < 225)
    for row_index, row in enumerate(np.array_split(gray_fill, 4)):
        gray_row = np.array_split(gray, 4)[row_index]
        for column_index, tile in enumerate(np.array_split(row, 4, axis=1)):
            gray_tile = np.array_split(gray_row, 4, axis=1)[column_index]
            if tile.mean() > 0.98 and gray_tile.std() < 2.5:
                return 'GRAY_REGION'
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 12:
        return 'LOW_DETAIL_OR_BLURRED'
    return None


def scene_region(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active = np.mean(gray > 28, axis=1) > 0.15
    active[:int(height * 0.2)] = False
    active[int(height * 0.9):] = False
    runs = []
    start = None
    for position, value in enumerate(np.append(active, False)):
        if value and start is None:
            start = position
        elif not value and start is not None:
            runs.append((start, position))
            start = None
    if not runs:
        raise ValueError('No expanded camera scene')
    top, bottom = max(runs, key=lambda interval: interval[1] - interval[0])
    if bottom - top < height * 0.3:
        raise ValueError('Camera scene is not expanded enough')
    return (int(width * 0.03), top + 5, int(width * 0.97), bottom - 5)


class Guard:
    def __init__(self, seconds, phone=False):
        self.seconds = seconds
        self.phone = phone
        self.serial = 'WC4TNNVCW8TCBABA' if phone else SERIAL
        self.screen_size = (1080, 2412) if phone else (900, 1600)
        self.adb_prefix = ([str(ROOT / '.tools/android/platform-tools/adb.exe'), '-P', '5038']
                           if phone else [str(ADB)]) + ['-s', self.serial]
        self.started = time.monotonic()
        self.maximum_runtime_seconds = 7200
        self.external_stop_path = None
        self.folder = CAPTURE / ('guarded_' + datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S'))
        self.folder.mkdir(exist_ok=False)
        self.state = {'status': 'PREPARING', 'serial': self.serial, 'instances': 1,
                      'device_type': 'phone' if phone else 'emulator',
                      'seconds_per_clip': seconds, 'poll_interval_seconds': 0.5,
                      'freeze_limit_seconds': 2, 'is_ground_truth': False,
                      'ready_wait_seconds': 45, 'stable_seconds_before_recording': 5,
                      'automatic_checks_are_not_human_approval': True, 'cameras': {}}
        self.save()

    def adb(self, *arguments, timeout=15):
        arguments = list(arguments)
        if self.phone and arguments[:3] == ['shell', 'input', 'tap']:
            arguments[3] = str(round(int(arguments[3]) * self.screen_size[0] / 900))
            arguments[4] = str(round(int(arguments[4]) * self.screen_size[1] / 1600))
        if self.phone and arguments[:3] == ['shell', 'input', 'swipe']:
            for position in (3, 5):
                arguments[position] = str(round(int(arguments[position]) * self.screen_size[0] / 900))
            for position in (4, 6):
                arguments[position] = str(round(int(arguments[position]) * self.screen_size[1] / 1600))
        return subprocess.run(self.adb_prefix + arguments, check=True,
                              capture_output=True, timeout=timeout).stdout

    def require_ihanoi(self):
        activity = self.adb('shell', 'dumpsys', 'activity', 'activities', timeout=8).decode(errors='replace')
        resumed = [line for line in activity.splitlines()
                   if 'topResumedActivity=' in line or 'mResumedActivity:' in line]
        if not any('com.viettel.citizens.hanoi/' in line for line in resumed):
            raise ValueError('iHanoi is not foreground; refusing capture of other apps')

    def dump(self):
        self.require_ihanoi()
        for attempt in range(3):
            try:
                self.adb('shell', 'rm', '-f', '/sdcard/ihanoi_guard.xml')
                result = self.adb('shell', 'uiautomator', 'dump', '/sdcard/ihanoi_guard.xml')
                if b'ERROR' in result:
                    raise ValueError('Fresh UI hierarchy unavailable')
                raw = self.adb('shell', 'cat', '/sdcard/ihanoi_guard.xml')
                tree = ET.fromstring(raw)
                if not list(tree.iter('node')):
                    raise ValueError('Empty UI hierarchy')
                if self.phone:
                    for node in tree.iter('node'):
                        values = [int(value) for value in re.findall(r'\d+', node.get('bounds', ''))]
                        if len(values) == 4:
                            left, top, right, bottom = values
                            node.set('bounds', f'[{round(left * 900 / 1080)},{round(top * 1600 / 2412)}]'
                                     f'[{round(right * 900 / 1080)},{round(bottom * 1600 / 2412)}]')
                    raw = ET.tostring(tree)
                return raw
            except (subprocess.SubprocessError, ValueError):
                if attempt == 2 or self.stopped():
                    raise
                time.sleep(2)

    def screenshot(self):
        self.require_ihanoi()
        data = self.adb('exec-out', 'screencap', '-p', timeout=5)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Screenshot decode failed')
        if self.phone:
            if (frame.shape[1], frame.shape[0]) != self.screen_size:
                raise ValueError('Phone rotated or dimensions changed; stopping capture')
            frame = cv2.resize(frame, (900, 1600))
        return frame

    def save(self):
        temporary = self.folder / 'status.tmp'
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.folder / 'status.json')

    def stopped(self):
        return ((self.folder / 'STOP').exists()
                or (self.external_stop_path is not None and self.external_stop_path.exists())
                or shutil.disk_usage(self.folder).free < 5 * 1024**3
                or (self.maximum_runtime_seconds is not None
                    and time.monotonic() - self.started > self.maximum_runtime_seconds))

    def inspect(self, frame, region, previous, last_motion):
        left, top, right, bottom = region
        crop = frame[top:bottom, left:right]
        problem = image_problem(crop)
        if problem:
            raise ValueError(problem)
        reduced = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (320, 180))
        if previous is not None:
            changed = float(np.mean(cv2.absdiff(previous, reduced) > 12))
            if changed > 0.002:
                last_motion = time.monotonic()
            elif time.monotonic() - last_motion >= 2:
                raise ValueError('FROZEN_OR_STALLED_2_SECONDS')
        return reduced, last_motion

    def record(self, identity, already_expanded=False):
        if not re.fullmatch(r'C\d+(?:\.\d+)*(?:-[A-Za-z0-9]+)?', identity):
            raise ValueError('Invalid camera identity')
        task = {'status': 'WAITING_CLEAR_EXPANDED_SCENE', 'checks': []}
        self.state['cameras'][identity] = task
        self.save()
        process = None
        remote = ('/data/local/tmp/' if self.phone else '/sdcard/') + 'ihanoi_guard_' + self.folder.name + '.mp4'
        safe = re.sub(r'[^A-Za-z0-9_-]', '_', identity)
        destination = self.folder / (safe + '.mp4')
        pid = None
        try:
            raw = self.dump()
            descriptions = [node.get('content-desc', '') or node.get('text', '')
                            for node in ET.fromstring(raw).iter('node')]
            if not any(identity in value for value in descriptions):
                raise ValueError('Camera identity not verified')
            deadline = time.monotonic() + self.state['ready_wait_seconds']
            region = None
            ready = False
            while not already_expanded and time.monotonic() < deadline:
                if self.stopped():
                    raise ValueError('STOP_REQUESTED')
                frame = self.screenshot()
                if frame.shape[:2] != (1600, 900):
                    raise ValueError('Uncalibrated screen dimensions')
                if not image_problem(frame[185:650, 25:875]):
                    ready = True
                    break
                time.sleep(0.5)
            if not ready and not already_expanded:
                raise ValueError('Camera did not load clearly before deadline')
            if not already_expanded:
                self.adb('shell', 'input', 'tap', '850', '711' if self.phone else '685')
                time.sleep(1)
            expanded = self.screenshot()
            gray = cv2.cvtColor(expanded, cv2.COLOR_BGR2GRAY)
            if np.mean(gray[200:450] < 24) < 0.97 or np.mean(gray[1250:1400] < 24) < 0.97:
                raise ValueError('Expanded player layout not verified; refusing recording')
            region = scene_region(expanded)
            if region is None:
                raise ValueError('No clear expanded scene before deadline')
            previous = None
            last_motion = time.monotonic()
            stable_since = None
            stable = False
            while time.monotonic() < deadline:
                if self.stopped():
                    raise ValueError('STOP_REQUESTED')
                try:
                    frame = self.screenshot()
                    if frame.shape[:2] != (1600, 900):
                        raise ValueError('Screen dimensions changed')
                    previous, last_motion = self.inspect(frame, region, previous, last_motion)
                    if stable_since is None:
                        stable_since = time.monotonic()
                    if time.monotonic() - stable_since >= self.state['stable_seconds_before_recording']:
                        stable = True
                        break
                except ValueError as error:
                    task['last_readiness_problem'] = str(error)
                    stable_since = None
                    previous = None
                    last_motion = time.monotonic()
                self.save()
                time.sleep(0.5)
            if not stable:
                raise ValueError('Scene did not remain stable for five seconds before playback deadline')
            if self.adb('shell', 'pidof', 'screenrecord', timeout=5).strip():
                raise ValueError('Another screenrecord is running')
        except subprocess.CalledProcessError as error:
            if error.cmd[-1] != 'screenrecord' or error.returncode != 1:
                task.update(status='SKIPPED_NOT_READY', reason=str(error))
                self.save()
                return
        except Exception as error:
            task.update(status='SKIPPED_NOT_READY', reason=str(error))
            self.save()
            print(identity, task['status'], task['reason'], flush=True)
            return
        try:
            task.update(status='RECORDING_MONITORED', scene_region=list(region))
            self.save()
            with (self.folder / (safe + '.record.log')).open('wb') as log:
                width, height = self.screen_size
                if self.phone:
                    if not SCRCPY.is_file():
                        raise ValueError('Verified scrcpy installation is required for phone recording')
                    task.update(capture_method='SCRCPY_USB', timebase_verified=False,
                                timestamp_source='phone_encoder_not_native_camera',
                                source_screen_size=list(self.screen_size))
                    environment = os.environ.copy()
                    environment['ADB'] = self.adb_prefix[0]
                    environment['ADB_SERVER_SOCKET'] = 'tcp:localhost:5038'
                    process = subprocess.Popen(phone_record_command(self.serial, destination, self.seconds),
                        env=environment, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process = subprocess.Popen(self.adb_prefix + ['shell', 'screenrecord',
                        '--size', f'{width}x{height}', '--bit-rate', '6000000', '--time-limit', str(self.seconds), remote],
                        stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                time.sleep(1)
                if not self.phone:
                    pids = self.adb('shell', 'pidof', 'screenrecord').decode().split()
                    if len(pids) != 1 or not pids[0].isdigit():
                        raise ValueError('Cannot identify own recorder safely')
                    pid = pids[0]
                previous = None
                last_motion = time.monotonic()
                deadline = time.monotonic() + self.seconds
                while process.poll() is None and time.monotonic() < deadline:
                    if self.stopped():
                        raise ValueError('STOP_REQUESTED')
                    previous, last_motion = self.inspect(self.screenshot(), region, previous, last_motion)
                    task['checks'].append({'elapsed': round(self.seconds - (deadline - time.monotonic()), 2)})
                    self.save()
                    time.sleep(0.5)
                if process.wait(timeout=10) != 0:
                    raise ValueError('Recorder failed')
            if not self.phone:
                self.adb('pull', remote, str(destination), timeout=30)
            capture = cv2.VideoCapture(str(destination))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                frames = 0
                while True:
                    valid, frame = capture.read()
                    if not valid:
                        break
                    task['encoded_frame_size'] = [frame.shape[1], frame.shape[0]]
                    task['postcheck_frames'] = frames
                    if self.phone:
                        aspect_error = abs((frame.shape[1] / frame.shape[0]) /
                                           (self.screen_size[0] / self.screen_size[1]) - 1)
                        if frame.shape[1] < 720 or aspect_error > 0.01:
                            raise ValueError('POSTCHECK_PHONE_RESOLUTION_OR_ORIENTATION_CHANGED')
                        frame = cv2.resize(frame, (900, 1600))
                    elif (frame.shape[1], frame.shape[0]) != self.screen_size:
                        raise ValueError('POSTCHECK_SCREEN_SIZE_CHANGED')
                    left, top, right, bottom = region
                    problem = image_problem(frame[top:bottom, left:right])
                    if problem:
                        raise ValueError('POSTCHECK_' + problem)
                    frames += 1
                if not fps or frames / fps < self.seconds - 2:
                    raise ValueError('POSTCHECK_TRUNCATED_VIDEO')
                task.update(status='PENDING_HUMAN_QA', video=str(destination.relative_to(ROOT)),
                            frames=frames, recording_fps=fps)
            finally:
                capture.release()
        except Exception as error:
            task.update(status='CANCELLED_QUALITY_FAILURE', reason=str(error), video=None)
            if pid and process and process.poll() is None:
                self.adb('shell', 'kill', '-2', pid)
                process.wait(timeout=12)
            if self.phone and process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            if destination.exists():
                if not destination.resolve().is_relative_to(self.folder.resolve()):
                    raise ValueError('Refusing deletion outside this capture attempt')
                subprocess.run(['powershell', '-NoProfile', '-Command',
                    "Remove-Item -LiteralPath '" + str(destination).replace("'", "''") + "' -Force"],
                    check=True)
        finally:
            if process is not None and process.poll() is None:
                if self.phone:
                    process.terminate()
                    process.wait(timeout=10)
                else:
                    process.wait(timeout=self.seconds + 15)
            if not self.phone:
                self.adb('shell', 'rm', '-f', remote)
            self.save()
            print(identity, task['status'], task.get('reason', ''), flush=True)

    def return_to_listing(self):
        raw = self.dump()
        try:
            camera_rows(raw)
            return
        except ValueError:
            descriptions = [(node.get('content-desc', '') or node.get('text', ''))
                            for node in ET.fromstring(raw).iter('node')]
            if 'Chi tiết camera' not in descriptions:
                raise ValueError('Unexpected screen after attempt; no blind navigation')
        frame = self.screenshot()
        if frame.shape[:2] != (1600, 900):
            raise ValueError('Screen rotated; stopping navigation')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.mean(gray[1250:1400] < 24) > 0.97:
            self.adb('shell', 'input', 'tap', '850', '1490')
            time.sleep(0.5)
        self.adb('shell', 'input', 'tap', '50', '140' if self.phone else '100')
        time.sleep(0.5)
        camera_rows(self.dump())

    def remaining(self, previous_path=None, retry_failed=False, rewind=False):
        choices_path = CAPTURE / 'city_survey_20261029/human_choices.json'
        choices = json.loads(choices_path.read_text(encoding='utf-8'))
        previous = None
        if previous_path is not None:
            previous = json.loads(previous_path.read_text(encoding='utf-8'))
            if previous.get('status') not in ('STOPPED_SAFELY', 'FINISHED_SCAN_REVIEW_COVERAGE'):
                raise ValueError('Previous scan must have stopped before resuming')
            self.state['previous_status_file'] = str(previous_path.resolve().relative_to(ROOT))
        if retry_failed:
            previous = {'cameras': {}}
            for status_path in CAPTURE.glob('guarded_*/status.json'):
                saved = json.loads(status_path.read_text(encoding='utf-8'))
                for identity, row in saved.get('cameras', {}).items():
                    if row.get('status') == 'PENDING_HUMAN_QA' and row.get('video') and (ROOT / row['video']).is_file():
                        previous['cameras'][identity] = row
            self.state['retry_failed'] = True
        wanted = remaining_camera_ids(choices, previous, retry_failed=retry_failed)
        self.state.update(status='SCANNING_SINGLE_INSTANCE', requested_cameras=sorted(wanted))
        self.save()
        opened = set()
        signatures = set()
        try:
            if self.phone or rewind:
                self.return_to_listing()
                prior_signature = None
                for position in range(36):
                    if self.stopped():
                        break
                    raw = self.dump()
                    camera_rows(raw)
                    signature = tuple((node.get('content-desc'), node.get('bounds'))
                                      for node in ET.fromstring(raw).iter('node') if node.get('content-desc'))
                    if signature == prior_signature:
                        break
                    prior_signature = signature
                    self.adb('shell', 'input', 'swipe', '450', '450', '450', '1450', '350')
                    time.sleep(0.5)
            for step in range(240):
                if self.stopped() or wanted.issubset(self.state['cameras']):
                    break
                raw = self.dump()
                rows = camera_rows(raw)
                current = json.loads(choices_path.read_text(encoding='utf-8'))
                targets = [identity for identity in rows if identity in wanted
                           and identity not in self.state['cameras']
                           and current.get(identity, {}).get('choice') in ('THEO_DOI_DEN_VACH', 'CAN_XEM_LAI')
                           and center(rows[identity].get('ui_bounds'))]
                targets.sort(key=lambda identity: current[identity]['choice'] != 'THEO_DOI_DEN_VACH')
                if targets:
                    identity = targets[0]
                    point = center(rows[identity]['ui_bounds'])
                    self.adb('shell', 'input', 'tap', *map(str, point))
                    time.sleep(1)
                    self.record(identity)
                    self.return_to_listing()
                    signatures.clear()
                    continue
                section = next_section(raw, opened)
                if section:
                    name, point = section
                    opened.add(name)
                    self.adb('shell', 'input', 'tap', *map(str, point))
                    time.sleep(0.5)
                    continue
                signature = tuple(rows)
                if signature in signatures:
                    break
                signatures.add(signature)
                self.adb('shell', 'input', 'swipe', '450', '1450', '450', '450', '450')
                time.sleep(0.5)
            self.state['status'] = 'FINISHED_SCAN_REVIEW_COVERAGE'
        except Exception as error:
            self.state.update(status='STOPPED_SAFELY', reason=str(error))
        self.state['not_attempted'] = sorted(wanted - self.state['cameras'].keys())
        self.save()


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--camera')
    selection.add_argument('--remaining', action='store_true')
    parser.add_argument('--phone', action='store_true')
    parser.add_argument('--already-expanded', action='store_true')
    parser.add_argument('--restore-stay-on', type=int, choices=range(8))
    parser.add_argument('--previous-status', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--seconds', type=int, default=30, choices=range(10, 61))
    args = parser.parse_args()
    if args.previous_status and not args.remaining:
        parser.error('--previous-status requires --remaining')
    if args.retry_failed and not args.remaining:
        parser.error('--retry-failed requires --remaining')
    if args.already_expanded and not args.camera:
        parser.error('--already-expanded requires --camera')
    guard = Guard(args.seconds, phone=args.phone)
    print(str(guard.folder), flush=True)
    original_stay_on = None
    try:
        if args.phone:
            original_stay_on = args.restore_stay_on
            if original_stay_on is None:
                original_stay_on = int(guard.adb('shell', 'settings', 'get', 'global',
                                                'stay_on_while_plugged_in').strip())
            guard.state['original_stay_on'] = original_stay_on
            guard.save()
            guard.require_ihanoi()
            recorders = subprocess.run(guard.adb_prefix + ['shell', 'pidof', 'screenrecord'],
                                      capture_output=True, timeout=10)
            if recorders.returncode == 0 and recorders.stdout.strip():
                guard.state.update(status='BLOCKED_EXISTING_PHONE_RECORDER',
                                   reason='An existing phone screenrecord must be stopped before any new capture')
                guard.save()
                return
            if recorders.returncode not in (0, 1):
                raise ValueError('Cannot verify phone recorder state')
            guard.adb('shell', 'svc', 'power', 'stayon', 'usb')
        if args.remaining:
            guard.remaining(args.previous_status, retry_failed=args.retry_failed)
        else:
            guard.record(args.camera, already_expanded=args.already_expanded)
            guard.state['status'] = 'FINISHED'
        guard.save()
    finally:
        if args.phone and original_stay_on is not None:
            guard.adb('shell', 'settings', 'put', 'global', 'stay_on_while_plugged_in', str(original_stay_on))
            guard.state['stay_on_restored'] = True
            guard.save()


if __name__ == '__main__':
    main()
