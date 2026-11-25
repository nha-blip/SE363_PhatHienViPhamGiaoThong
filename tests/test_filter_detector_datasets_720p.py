import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.filter_detector_datasets_720p import POLICY, analyze, label_path, parse_labels, transform_box


class Filter720pTests(unittest.TestCase):
    def make_sample(self, folder, width=1280, height=720, sharp=True, rows=None):
        root = Path(folder)
        image = root / 'images/train/sample.png'
        image.parent.mkdir(parents=True)
        label = label_path(image)
        label.parent.mkdir(parents=True)
        if sharp:
            grid = ((np.indices((height, width)).sum(axis=0) // 5) % 2 * 255).astype(np.uint8)
        else:
            grid = np.full((height, width), 128, dtype=np.uint8)
        cv2.imwrite(str(image), grid)
        label.write_text(rows if rows is not None else '0 0.5 0.5 0.25 0.25\n')
        return {'kind': 'vehicle', 'split': 'train', 'source': 'test', 'image': str(image)}

    def test_no_upscale_for_low_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder, 640, 360)
            self.assertIn('NATIVE_BELOW_720P', analyze(item, {})['reasons'])

    def test_sharp_native_720p_is_kept_without_modification(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder)
            original = Path(item['image']).read_bytes()
            result = analyze(item, {})
            self.assertEqual(result['decision'], 'KEEP')
            self.assertEqual(Path(item['image']).read_bytes(), original)

    def test_blur_heuristic_rejects_flat_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder, sharp=False)
            result = analyze(item, {})
            self.assertIn('GLOBAL_BLUR_HEURISTIC', result['reasons'])

    def test_mixed_image_not_kept_with_unlabeled_small_object(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder, rows='0 0.5 0.5 0.25 0.25\n1 0.2 0.2 0.003 0.004\n')
            result = analyze(item, {})
            self.assertEqual(result['decision'], 'REJECT')
            self.assertEqual(len(result['objects']), 2)
            self.assertIn('CONTAINS_TOO_SMALL_OBJECT_AT_720P', result['reasons'])

    def test_letterbox_transform(self):
        self.assertEqual(transform_box([0, 0, 1280, 960], 0.75, 0.75, 160, 0), [160, 0, 1120, 720])

    def test_balanced_keeps_all_labels_when_one_object_is_useful(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder, rows='0 0.5 0.5 0.25 0.25\n1 0.2 0.2 0.003 0.004\n')
            with patch.dict(POLICY, require_one_useful_object=True):
                result = analyze(item, {})
            self.assertEqual(result['decision'], 'KEEP')
            self.assertEqual(len(result['objects']), 2)

    def test_balanced_low_resolution_is_padded_not_upscaled(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder, 640, 360)
            output = Path(folder) / 'filtered'
            for category in ('images', 'labels'):
                (output / 'vehicle' / category / 'train').mkdir(parents=True)
            with patch.dict(POLICY, min_native_width=1, min_native_height=1, require_one_useful_object=True):
                with patch('scripts.filter_detector_datasets_720p.OUTPUT', output):
                    result = analyze(item, {}, export=True)
            self.assertEqual(result['decision'], 'KEEP')
            self.assertEqual(cv2.imread(result['output_image']).shape[:2], (720, 1280))
            values = list(map(float, Path(result['output_label']).read_text().split()[1:]))
            self.assertEqual(values, [0.5, 0.5, 0.125, 0.125])

    def test_balanced_rejects_frame_without_large_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder, rows='0 0.5 0.5 0.003 0.004\n')
            with patch.dict(POLICY, require_one_useful_object=True):
                self.assertEqual(analyze(item, {})['decision'], 'REJECT')

    def test_invalid_geometry_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            parse_labels(b'0 0.5 0.5 0 0.1', ['car'], 1280, 720)

    def test_export_has_exact_720p_and_correct_letterbox_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            item = self.make_sample(folder, 1280, 960)
            output = Path(folder) / 'filtered'
            for category in ('images', 'labels'):
                (output / 'vehicle' / category / 'train').mkdir(parents=True)
            original = Path(item['image']).read_bytes()
            with patch('scripts.filter_detector_datasets_720p.OUTPUT', output):
                result = analyze(item, {}, export=True)
            self.assertEqual(result['decision'], 'KEEP')
            image = cv2.imread(result['output_image'])
            self.assertEqual(image.shape[:2], (720, 1280))
            values = Path(result['output_label']).read_text().split()
            self.assertEqual(values[0], '0')
            self.assertEqual(list(map(float, values[1:])), [0.5, 0.5, 0.1875, 0.25])
            self.assertEqual(Path(item['image']).read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
