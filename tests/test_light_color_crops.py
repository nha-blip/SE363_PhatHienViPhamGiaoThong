import unittest
from unittest.mock import patch

import numpy as np

from scripts.build_readable_crop_dataset import POLICY, color_cue, qualifies


class LightColorTests(unittest.TestCase):
    def test_bright_red_supports_red_not_green(self):
        region = np.zeros((8, 8, 3), dtype=np.uint8)
        region[2:4, 2:4] = [0, 0, 255]
        self.assertTrue(color_cue(region, 0)['supported'])
        self.assertFalse(color_cue(region, 2)['supported'])

    def test_white_and_off_have_no_color_evidence(self):
        self.assertFalse(color_cue(np.full((8, 8, 3), 255, dtype=np.uint8), 0)['supported'])
        self.assertFalse(color_cue(np.full((8, 8, 3), 255, dtype=np.uint8), 3)['supported'])

    def test_mixed_colors_do_not_pass(self):
        region = np.zeros((8, 8, 3), dtype=np.uint8)
        region[0:2] = [0, 0, 255]
        region[2:4] = [0, 255, 0]
        self.assertFalse(color_cue(region, 0)['supported'])

    def test_relaxed_cue_does_not_override_manual_rejection(self):
        obj = {'class': 0, 'target_box': [0, 0, 5, 10], 'crop_laplacian_variance': 0, 'color_cue': {'supported': True}}
        with patch.dict(POLICY['light'], allow_color_cue=True):
            self.assertTrue(qualifies('light', obj, None))
            self.assertFalse(qualifies('light', obj, 'UNREADABLE'))
