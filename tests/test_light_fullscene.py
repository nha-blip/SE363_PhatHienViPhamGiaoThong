import unittest
from scripts.build_light_fullscene_20261028 import geometry, transform


class FullsceneGeometryTests(unittest.TestCase):
    def test_720p_unchanged(self):
        rows = [(0, 0.5, 0.5, 0.1, 0.2)]
        self.assertEqual(transform(rows, 1280, 720), rows)

    def test_full_hd_scales_without_changing_normalized_boxes(self):
        rows = [(2, 0.5, 0.25, 0.02, 0.04)]
        self.assertEqual(transform(rows, 1920, 1080), rows)

    def test_small_image_padded_not_enlarged(self):
        self.assertEqual(geometry(640, 360), (640, 360, 320, 180))
        result = transform([(1, 0.5, 0.5, 0.1, 0.2)], 640, 360)[0]
        self.assertEqual(result, (1, 0.5, 0.5, 0.05, 0.1))

    def test_square_preserves_entire_frame(self):
        self.assertEqual(geometry(1080, 1080), (720, 720, 280, 0))
        self.assertEqual(transform([(3, 0.5, 0.5, 1.0, 1.0)], 1080, 1080),
                         [(3, 0.5, 0.5, 720/1280, 1.0)])
