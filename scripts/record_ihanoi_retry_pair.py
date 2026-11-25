import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.inventory_ihanoi_cameras import ADB, ROOT, camera_rows


FFMPEG = ROOT / '.tools/ffmpeg_runtime/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
TASKS = [('emulator-5564', 'C069.10-PTZ'), ('emulator-5584', 'C057.08-PTZ')]


def main():
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
    folder = ROOT / 'data/iHanoi_capture' / ('retry_pair_' + stamp)
    folder.mkdir(exist_ok=False)
    state = {'status': 'PREPARING', 'recording_limit_seconds': 45, 'automatic_reopen': False,
             'is_ground_truth': False, 'tasks': []}
    jobs = []

    def save():
        temporary = folder / 'status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(folder / 'status.json')

    def adb(serial, *arguments):
        return subprocess.run([str(ADB), '-s', serial, *arguments], check=True,
                              capture_output=True, timeout=30).stdout

    try:
        for serial, identity in TASKS:
            adb(serial, 'shell', 'uiautomator', 'dump', '/sdcard/ihanoi_pair_list.xml')
            raw = adb(serial, 'shell', 'cat', '/sdcard/ihanoi_pair_list.xml')
            (folder / (serial + '_before.xml')).write_bytes(raw)
            row = camera_rows(raw)[identity]
            bounds = [int(value) for value in re.findall(r'\d+', row['ui_bounds'])]
            if len(bounds) != 4 or bounds[1] < 284 or bounds[3] > 1554:
                raise ValueError('Camera row is not fully visible; refusing blind tap')
            task = {'serial': serial, 'camera_id': identity, 'name': row['name'],
                    'tap': [(bounds[0]+bounds[2])//2, (bounds[1]+bounds[3])//2],
                    'status': 'READY_ON_LIST'}
            state['tasks'].append(task)
        save()
        for task in state['tasks']:
            name = task['camera_id'].replace('.', '_').replace('-', '_') + '.mp4'
            task['file'] = name
            task['remote'] = '/sdcard/ihanoi_' + stamp + '_' + name
            log = (folder / (name + '.record.log')).open('wb')
            process = subprocess.Popen([str(ADB), '-s', task['serial'], 'shell', 'screenrecord',
                                        '--bit-rate', '4000000', '--time-limit', '45', task['remote']],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            jobs.append((task, process, log))
        time.sleep(1)
        for task in state['tasks']:
            adb(task['serial'], 'shell', 'input', 'tap', *map(str, task['tap']))
            task['status'] = 'RECORDING_OPENED_ONCE'
        state['status'] = 'RECORDING_TWO_INSTANCES'
        save()
        print(json.dumps({'folder': str(folder), 'status': state['status']}), flush=True)
        time.sleep(3)
        for task in state['tasks']:
            adb(task['serial'], 'shell', 'input', 'tap', '850', '685')
        for task, process, log in jobs:
            code = process.wait(timeout=70)
            log.close()
            task['record_return_code'] = code
            if code:
                task['status'] = 'RECORDING_FAILED'
                continue
            adb(task['serial'], 'pull', task['remote'], str(folder / task['file']))
            path = folder / task['file']
            task['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            task['bytes'] = path.stat().st_size
            task['status'] = 'PENDING_VISUAL_QA'
            for second in (10, 25, 40):
                subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-ss', str(second),
                                '-i', str(path), '-frames:v', '1', '-y',
                                str(folder / (path.stem + f'_{second}s.png'))],
                               check=True, capture_output=True, timeout=30)
        state['status'] = 'RECORDING_FINISHED_PENDING_QA'
        state['recording_active'] = False
        save()
        print(json.dumps({'folder': str(folder), 'status': state['status']}), flush=True)
    except Exception as error:
        for task, process, log in jobs:
            try:
                process.wait(timeout=65)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            log.close()
        state.update(status='STOPPED_SAFELY', reason=type(error).__name__ + ': ' + str(error),
                     recording_active=False)
        save()
        raise


if __name__ == '__main__':
    main()
