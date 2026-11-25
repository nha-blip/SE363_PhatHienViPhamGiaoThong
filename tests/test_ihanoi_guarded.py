import unittest
from unittest.mock import patch

import numpy as np

from scripts.record_ihanoi_guarded import Guard, image_problem, scene_region, remaining_camera_ids, phone_record_command


class GuardedCaptureTests(unittest.TestCase):
    def test_textured_gray_scene_is_not_gray_corruption(self):
        gray = np.random.default_rng(0).integers(60, 210, (400, 600), dtype=np.uint8)
        frame = np.repeat(gray[:, :, None], 3, axis=2)
        self.assertIsNone(image_problem(frame))

    def test_phone_uses_scrcpy_not_failed_native_recorder(self):
        command = phone_record_command('phone', 'camera.mp4', 30)
        self.assertTrue(command[0].endswith('scrcpy.exe'))
        self.assertIn('--no-audio', command)
        self.assertIn('--no-control', command)
        self.assertIn('--no-window', command)
        self.assertIn('--time-limit=30', command)
        self.assertIn('--record=camera.mp4', command)
        self.assertNotIn('screenrecord', command)

    def test_phone_taps_scaled_to_physical_screen(self):
        guard = Guard.__new__(Guard)
        guard.phone = True
        guard.screen_size = (1080, 2412)
        guard.adb_prefix = ['adb', '-P', '5038', '-s', 'phone']
        with patch('scripts.record_ihanoi_guarded.subprocess.run') as run:
            guard.adb('shell', 'input', 'tap', '850', '711')
        self.assertEqual(run.call_args.args[0][-2:], ['1020', '1072'])

    def test_phone_refuses_other_apps(self):
        guard = Guard.__new__(Guard)
        guard.phone = True
        with patch.object(guard, 'adb', return_value=b'topResumedActivity=example.other/.Main'):
            with self.assertRaisesRegex(ValueError, 'not foreground'):
                guard.require_ihanoi()

    def test_emulator_refuses_other_apps(self):
        guard = Guard.__new__(Guard)
        guard.phone = False
        with patch.object(guard, 'adb', return_value=b'mResumedActivity: example.other/.Main'):
            with self.assertRaisesRegex(ValueError, 'not foreground'):
                guard.require_ihanoi()

    def test_retry_failed_preserves_successes_but_retries_failures(self):
        choices = {identity: {'choice': 'CAN_XEM_LAI'} for identity in ('C001', 'C002', 'C003')}
        choices['C004'] = {'choice': 'BO_QUA'}
        previous = {'cameras': {'C001': {'status': 'PENDING_HUMAN_QA'},
                                'C002': {'status': 'SKIPPED_NOT_READY'}}}
        self.assertEqual(remaining_camera_ids(choices, previous, retry_failed=True), {'C002', 'C003'})

    def test_resume_excludes_attempted_and_rejected_cameras(self):
        choices = {'C001': {'choice': 'THEO_DOI_DEN_VACH'},
                   'C002': {'choice': 'CAN_XEM_LAI'},
                   'C003': {'choice': 'BO_QUA'}}
        previous = {'cameras': {'C001': {'status': 'PENDING_HUMAN_QA'}}}
        self.assertEqual(remaining_camera_ids(choices, previous), {'C002'})

    def test_black_rejected(self):
        self.assertEqual(image_problem(np.zeros((400, 600, 3), np.uint8)), 'BLACK_FRAME')

    def test_gray_patch_rejected(self):
        frame = np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)
        frame[100:200, 150:300] = 128
        self.assertEqual(image_problem(frame), 'GRAY_REGION')

    def test_expanded_scene_detected(self):
        frame = np.zeros((1600, 900, 3), np.uint8)
        frame[575:1185] = np.random.default_rng(0).integers(30, 255, (610, 900, 3), dtype=np.uint8)
        region = scene_region(frame)
        self.assertGreater(region[3] - region[1], 480)

    def test_small_scene_rejected(self):
        frame = np.zeros((1600, 900, 3), np.uint8)
        frame[400:600] = 150
        with self.assertRaises(ValueError):
            scene_region(frame)


if __name__ == '__main__':
    unittest.main()
