import unittest

import numpy as np

from scripts.filter_ihanoi_black_videos import black_frame, camera_statistics


class IhanoiBlackFilterTests(unittest.TestCase):
    def test_ignores_bright_app_header_and_black_letterbox(self):
        frame = np.zeros((1600, 900, 3), dtype=np.uint8)
        frame[:200] = 255
        self.assertTrue(black_frame(frame))

    def test_keeps_real_scene_even_with_black_borders(self):
        frame = np.zeros((1600, 900, 3), dtype=np.uint8)
        frame[575:1185] = 80
        self.assertFalse(black_frame(frame))

    def test_keeps_dim_scene_with_visible_lights(self):
        frame = np.zeros((1600, 900, 3), dtype=np.uint8)
        frame[800:825, 300:330] = 180
        self.assertFalse(black_frame(frame))

    def test_gray_corruption_is_not_black_screen(self):
        frame = np.full((1600, 900, 3), 128, dtype=np.uint8)
        self.assertFalse(black_frame(frame))

    def test_unsupported_layout_is_not_auto_excluded(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertIsNone(camera_statistics(frame))
        self.assertFalse(black_frame(frame))
