import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_fullscene_resume_20261028'
CAPTURE = ROOT / 'runs/project363/yolo26_kaggle2_20261027'
PYTHON = ROOT / '.tools/kaggle2_cli/Scripts/python.exe'


def main():
    deadline = time.monotonic() + 4*3600
    while time.monotonic() < deadline:
        status_path = REPORT / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status') == 'STOPPED_SAFELY':
            return 1
        if not (REPORT / 'submission.json').exists():
            time.sleep(30)
            continue
        try:
            submission = json.loads((REPORT / 'submission.json').read_text())
            from deploy_yolo26_fullscene_resume_20261028 import canonical_notebook
            reference = canonical_notebook(submission['url'])
            subprocess.run([str(PYTHON), str(ROOT / 'scripts/read_kaggle2_training_logs.py'),
                            '--seconds', '30', '--notebook', reference,
                            '--output-prefix', 'fullscene_live'], capture_output=True, timeout=120)
            source = CAPTURE / 'fullscene_live_status.json'
            live = json.loads(source.read_text()) if source.exists() else {}
            for suffix in ('status.json', 'snapshot.log'):
                path = CAPTURE / ('fullscene_live_' + suffix)
                if path.exists():
                    shutil.copyfile(path, REPORT / ('live_' + suffix))
            markers = live.get('training_start_markers', [])
            seen = {row['task'] for row in markers if row.get('task') in {'vehicle', 'light'}
                    and str(row.get('device', '')).startswith('cuda')}
            result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                      'status': 'BOTH_WORKERS_TRAINING_START_VERIFIED' if len(seen) == 2 else 'WAITING_WORKER_START',
                      'training_workers_seen': sorted(seen), 'training_start_markers': markers,
                      'latest_progress': live.get('latest_progress'),
                      'note': 'Start markers prove worker startup, not completion or current GPU utilization'}
            temporary = REPORT / 'worker_start_status.tmp'
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(REPORT / 'worker_start_status.json')
            if len(seen) == 2:
                return 0
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass
        time.sleep(45)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
