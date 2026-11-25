import argparse
import json
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.record_ihanoi_guarded import CAPTURE, ROOT, Guard


def successful_captures(capture_root=CAPTURE, project_root=ROOT):
    successful = {}
    for status_path in sorted(capture_root.glob('guarded_*/status.json')):
        try:
            saved = json.loads(status_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        for identity, row in saved.get('cameras', {}).items():
            if row.get('status') != 'PENDING_HUMAN_QA' or not row.get('video'):
                continue
            video = (project_root / row['video']).resolve()
            if video.is_relative_to(capture_root.resolve()) and video.is_file() and video.stat().st_size > 0:
                successful[identity] = dict(video=row['video'], status='PENDING_HUMAN_QA')
    return successful


class ContinuousCapture:
    def __init__(self, folder, seconds=30, retry_delay=30):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=False)
        self.seconds = seconds
        self.retry_delay = retry_delay
        self.state = dict(status='PREPARING', instances=1, device_type='emulator',
                          seconds_per_clip=seconds, retry_until_complete=True,
                          maximum_runtime_seconds=None, is_ground_truth=False,
                          completion_means='ONE_EXISTING_AUTOCHECKED_CLIP_PER_ELIGIBLE_CAMERA',
                          passes=[], successful_cameras={}, missing_cameras=[])
        self.save()

    def save(self):
        self.state['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = self.folder / 'status.tmp'
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.folder / 'status.json')

    def update_coverage(self):
        choices = json.loads((CAPTURE / 'city_survey_20261029/human_choices.json').read_text(encoding='utf-8'))
        eligible = {identity for identity, row in choices.items()
                    if row.get('choice') in ('THEO_DOI_DEN_VACH', 'CAN_XEM_LAI')}
        successful = successful_captures()
        self.state['eligible_cameras'] = sorted(eligible)
        self.state['successful_cameras'] = {identity: successful[identity]
                                            for identity in sorted(eligible & successful.keys())}
        self.state['missing_cameras'] = sorted(eligible - successful.keys())

    def stop_reason(self):
        if (self.folder / 'STOP').exists():
            return 'STOP_REQUESTED'
        if shutil.disk_usage(self.folder).free < 5 * 1024**3:
            return 'LOW_DISK_SPACE'
        return None

    def run(self):
        try:
            while True:
                self.update_coverage()
                reason = self.stop_reason()
                if reason:
                    self.state.update(status='STOPPED_SAFELY', reason=reason)
                    break
                if not self.state['missing_cameras']:
                    self.state['status'] = 'COMPLETE_PENDING_HUMAN_QA'
                    break
                guard = Guard(self.seconds)
                guard.maximum_runtime_seconds = None
                guard.external_stop_path = self.folder / 'STOP'
                guard.state['continuous_controller'] = str(self.folder.relative_to(ROOT))
                guard.save()
                self.state.update(status='SCANNING_SINGLE_INSTANCE',
                                  active_scan=str(guard.folder.relative_to(ROOT)))
                self.save()
                print('PASS', len(self.state['passes']) + 1, self.state['active_scan'],
                      'missing', len(self.state['missing_cameras']), flush=True)
                guard.remaining(retry_failed=True, rewind=True)
                self.state['passes'].append(dict(scan=self.state['active_scan'],
                                                status=guard.state['status'],
                                                reason=guard.state.get('reason'),
                                                attempted=len(guard.state['cameras'])))
                self.update_coverage()
                if guard.state['status'] == 'STOPPED_SAFELY':
                    self.state['last_scan_error'] = guard.state.get('reason')
                self.state['status'] = 'WAITING_TO_RETRY'
                self.save()
                print('COVERAGE', len(self.state['successful_cameras']), '/',
                      len(self.state['eligible_cameras']), flush=True)
                if not self.state['missing_cameras']:
                    continue
                deadline = time.monotonic() + self.retry_delay
                while time.monotonic() < deadline and not self.stop_reason():
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
        except Exception as error:
            self.state.update(status='STOPPED_SAFELY', reason=str(error))
            raise
        finally:
            self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=30, choices=range(10, 61))
    parser.add_argument('--retry-delay', type=int, default=30, choices=range(15, 301))
    args = parser.parse_args()
    timestamp = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
    controller = ContinuousCapture(CAPTURE / ('continuous_' + timestamp), args.seconds, args.retry_delay)
    print(controller.folder, flush=True)
    controller.run()


if __name__ == '__main__':
    main()
