import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_kaggle2_20261027'
PYTHON = ROOT / '.tools/kaggle2_cli/Scripts/python.exe'
NOTEBOOK = 'v5nhdf/yolo26s-vehicle-light-20261027'


def main():
    started = set()
    deadline = time.monotonic() + 5*3600
    while time.monotonic() < deadline:
        state_path = REPORT / 'deployment_status.json'
        if not state_path.exists():
            time.sleep(30)
            continue
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['status'] == 'FAILED':
            return 1
        if not (REPORT / 'submission.json').exists():
            time.sleep(30)
            continue
        try:
            result = subprocess.run([str(PYTHON), str(ROOT / 'scripts/read_kaggle2_training_logs.py'), '--seconds', '20'],
                                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
        except subprocess.TimeoutExpired:
            time.sleep(60)
            continue
        snapshot_path = REPORT / 'live_stream_snapshot.log'
        snapshot_status = REPORT / 'live_stream_status.json'
        text = snapshot_path.read_text(encoding='utf-8') if snapshot_path.exists() else ''
        live = json.loads(snapshot_status.read_text(encoding='utf-8')) if snapshot_status.exists() else {}
        if result.returncode == 0:
            (REPORT / 'live_notebook.log').write_text(text, encoding='utf-8')
            for line in text.splitlines():
                if 'YOLO26_WORKER_TRAINING_STARTED ' in line:
                    try:
                        message = json.loads(line.split('YOLO26_WORKER_TRAINING_STARTED ', 1)[1])
                        if message.get('task') in {'vehicle', 'light'} and str(message.get('device', '')).startswith('cuda'):
                            started.add(message['task'])
                    except (ValueError, TypeError):
                        pass
            status = {'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                      'status': 'BOTH_GPU_TRAINING_START_VERIFIED' if len(started) == 2 else 'WAITING_FOR_TRAINING_WORKERS',
                      'cuda_training_workers_seen': sorted(started), 'notebook': NOTEBOOK,
                      'latest_progress': live.get('latest_progress'),
                      'log_source': 'Live SSE stream; one-shot logs are empty during active sessions',
                      'note': 'Start markers verify past worker start, not current success or completion'}
            progress = live.get('latest_progress') or {}
            for kind in ('vehicle', 'light'):
                if progress.get(kind, {}).get('epoch_completed', 0) > 0:
                    started.add(kind)
            status['cuda_training_workers_seen'] = sorted(started)
            if len(started) == 2:
                status['status'] = 'BOTH_GPU_TRAINING_START_VERIFIED'
            temporary = REPORT / 'worker_start_status.tmp'
            temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
            temporary.replace(REPORT / 'worker_start_status.json')
            if len(started) == 2:
                return 0
        time.sleep(60)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
