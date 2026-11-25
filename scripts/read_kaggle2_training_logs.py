import argparse
import json
import re
import sys
import time
import winreg
from datetime import datetime, timezone
from pathlib import Path

from kagglesdk import KaggleClient, KaggleEnv


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_kaggle2_20261027'


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=25)
    parser.add_argument('--notebook', default='v5nhdf/yolo26s-vehicle-light-20261027')
    parser.add_argument('--output-prefix', default='live_stream')
    args = parser.parse_args()
    if not re.fullmatch(r'v5nhdf/[a-z0-9-]+', args.notebook) or not re.fullmatch(r'[a-zA-Z0-9_]+', args.output_prefix):
        raise ValueError('Unexpected notebook or output name')
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    chunks = []
    states = []
    count = 0
    started = time.monotonic()
    http_status = None
    error_type = None
    truncated = False
    last_event_time = None
    try:
        with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
            http = client.http_client()
            response = http._session.get(http._endpoint + '/v1/kernels/logs/stream/' + args.notebook,
                                         headers={'Accept': 'text/event-stream, */*'}, auth=http._session.auth,
                                         stream=True, timeout=(15, 15))
            http_status = response.status_code
            response.raise_for_status()
            response.encoding = 'utf-8'
            with response:
                for line in response.iter_lines(chunk_size=1024, decode_unicode=True):
                    if time.monotonic()-started > args.seconds:
                        truncated = True
                        break
                    if not line.startswith('data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == 'END_OF_LOG':
                        break
                    try:
                        event = json.loads(raw)
                        if isinstance(event, dict):
                            last_event_time = event.get('time', last_event_time)
                        text = str(event.get('data', '')) if isinstance(event, dict) else str(event)
                    except ValueError:
                        text = raw
                    text = text.replace(token, '[REDACTED]')
                    chunks.append(text)
                    count += len(text)
                    if count > 20000000:
                        truncated = True
                        break
    except Exception as error:
        error_type = type(error).__name__
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', ''.join(chunks))
    if text:
        (REPORT / (args.output_prefix+'_snapshot.log')).write_text(text, encoding='utf-8')
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and ('vehicle' in value or 'light' in value):
            states.append(value)
    markers = []
    for line in text.splitlines():
        if 'YOLO26_WORKER_TRAINING_STARTED ' in line:
            try:
                markers.append(json.loads(line.split('YOLO26_WORKER_TRAINING_STARTED ', 1)[1]))
            except ValueError:
                pass
        else:
            try:
                value = json.loads(line)
                if isinstance(value, dict) and value.get('stage') == 'TRAINING_STARTED':
                    markers.append(value)
            except ValueError:
                pass
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'stream_http_status': http_status,
              'captured_characters': len(text), 'stream_end_or_timeout_type': error_type,
              'truncated_by_budget': truncated, 'last_server_event_time': last_event_time,
              'latest_progress': states[-1] if states else None, 'training_start_markers': markers}
    (REPORT / (args.output_prefix+'_status.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print('LATEST LOG TAIL:', flush=True)
    print(text[-11000:], flush=True)


if __name__ == '__main__':
    main()
