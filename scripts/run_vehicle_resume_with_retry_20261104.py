import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/resume_yolo26_vehicle_continue_20261104.py"
LOG = ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104_retry.log"
RESULTS = ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104/train/results.csv"
MAX_ATTEMPTS = 30


def completed_epochs():
    return len(RESULTS.read_text().splitlines()) - 1 if RESULTS.is_file() else 0


def main():
    for attempt in range(1, MAX_ATTEMPTS + 1):
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt} at {datetime.now():%F %T}, results rows {completed_epochs()} ===\n")
            log.flush()
            code = subprocess.call([sys.executable, str(SCRIPT)], stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
            log.write(f"=== exit code {code} at {datetime.now():%F %T}, results rows {completed_epochs()} ===\n")
        if code == 0:
            return
        time.sleep(180)
    raise SystemExit("Gave up after repeated failures")


if __name__ == "__main__":
    main()
