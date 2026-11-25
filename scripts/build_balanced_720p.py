import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    for script in ('filter_detector_datasets_720p.py', 'validate_filtered_720p.py'):
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script),
                        '--profile', 'balanced'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
