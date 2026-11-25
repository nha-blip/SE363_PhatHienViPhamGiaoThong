import unittest

from scripts.prepare_readable_crop_pilot import contains, intersects, safe_crop


class SafeCropTests(unittest.TestCase):
    def test_disjoint_bad_box_can_be_avoided(self):
        focus = [200, 200, 250, 250]
        bad = [270, 210, 280, 220]
        crop = safe_crop(focus, [focus, bad], {1})
        self.assertIsNotNone(crop)
        self.assertTrue(contains(crop, focus))
        self.assertFalse(intersects(crop, bad))

    def test_overlapping_bad_box_skips_crop(self):
        focus = [200, 200, 250, 250]
        self.assertIsNone(safe_crop(focus, [focus, [220, 220, 230, 230]], {1}))

    def test_crop_does_not_leave_partial_good_object(self):
        focus = [200, 200, 250, 250]
        other = [290, 180, 350, 240]
        crop = safe_crop(focus, [focus, other], set())
        self.assertTrue(contains(crop, focus))
        self.assertTrue(not intersects(crop, other) or contains(crop, other))
