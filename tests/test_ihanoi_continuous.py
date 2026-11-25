import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.continue_ihanoi_capture import ContinuousCapture, successful_captures


class ContinuousCaptureTests(unittest.TestCase):
    def test_success_requires_existing_nonempty_local_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / 'capture'
            scan = capture / 'guarded_test'
            scan.mkdir(parents=True)
            (scan / 'valid.mp4').write_bytes(b'previously checked video')
            (scan / 'empty.mp4').touch()
            rows = {identity: dict(status='PENDING_HUMAN_QA', video=video)
                    for identity, video in (
                        ('C001', 'capture/guarded_test/valid.mp4'),
                        ('C002', 'capture/guarded_test/missing.mp4'),
                        ('C003', 'capture/guarded_test/empty.mp4'))}
            (scan / 'status.json').write_text(json.dumps(dict(cameras=rows)), encoding='utf-8')
            self.assertEqual(set(successful_captures(capture, root)), {'C001'})

    def test_retries_failed_scan_until_coverage_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = ContinuousCapture(Path(temporary) / 'controller', retry_delay=0)
            observations = iter([['C001'], ['C001'], ['C001'], [], []])

            def update_coverage():
                controller.state['eligible_cameras'] = ['C001']
                controller.state['missing_cameras'] = next(observations)
                controller.state['successful_cameras'] = (
                    {} if controller.state['missing_cameras'] else {'C001': dict(video='clip.mp4')})

            with patch.object(controller, 'update_coverage', side_effect=update_coverage), \
                    patch.object(controller, 'stop_reason', return_value=None), \
                    patch('scripts.continue_ihanoi_capture.ROOT', Path(temporary)), \
                    patch('scripts.continue_ihanoi_capture.Guard') as guard_type:
                guard = guard_type.return_value
                guard.folder = Path(temporary) / 'guarded_test'
                guard.state = dict(status='STOPPED_SAFELY', reason='temporary stream outage', cameras={})
                controller.run()
            self.assertEqual(guard.remaining.call_count, 2)
            guard.remaining.assert_called_with(retry_failed=True, rewind=True)
            self.assertIsNone(guard.maximum_runtime_seconds)
            self.assertEqual(controller.state['status'], 'COMPLETE_PENDING_HUMAN_QA')

    def test_low_disk_stops_without_creating_recorder(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = ContinuousCapture(Path(temporary) / 'controller')
            with patch('scripts.continue_ihanoi_capture.shutil.disk_usage',
                       return_value=SimpleNamespace(free=1024)):
                self.assertEqual(controller.stop_reason(), 'LOW_DISK_SPACE')

    def test_explicit_stop_remains_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            controller = ContinuousCapture(Path(temporary) / 'controller')
            (controller.folder / 'STOP').touch()
            self.assertEqual(controller.stop_reason(), 'STOP_REQUESTED')


if __name__ == '__main__':
    unittest.main()
