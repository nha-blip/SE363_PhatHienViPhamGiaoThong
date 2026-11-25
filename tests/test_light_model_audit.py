import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_light_geometry_20261029 import geometry_candidates
from scripts.audit_yolo26_light_20261029 import predictions, size_bucket
from scripts.autorestart_yolo26_kaggle2 import prepare_resume


class LightAuditTests(unittest.TestCase):
    def test_geometry_candidate_is_not_a_confirmed_label_error(self):
        boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)
        identities = np.array([0])
        predicted = np.array([[10, 10, 20, 40, 0.9, 0]], dtype=np.float32)
        candidates = geometry_candidates(boxes, identities, predicted)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['gt_index'], 0)
        self.assertAlmostEqual(candidates[0]['pred_area_to_gt'], 3)
        np.testing.assert_array_equal(boxes, [[10, 10, 20, 20]])

    def test_geometry_candidate_requires_matching_class(self):
        boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)
        identities = np.array([0])
        predicted = np.array([[10, 10, 20, 40, 0.9, 1]], dtype=np.float32)
        self.assertEqual(geometry_candidates(boxes, identities, predicted), [])

    def test_size_bucket_boundaries(self):
        self.assertEqual([size_bucket(value) for value in (3.9, 4, 7.9, 8, 15.9, 16)],
                         ['<4px', '4-8px', '4-8px', '8-16px', '8-16px', '>=16px'])

    def test_empty_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(predictions(Path(folder)/'missing.txt', 1280, 720).shape, (0, 6))

    def test_native_coordinates_and_fixed_confidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'prediction.txt'
            path.write_text('2 0.5 0.5 0.1 0.2 0.8\n0 0.5 0.5 0.1 0.2 0.1\n')
            result = predictions(path, 1000, 500)
            self.assertEqual(result.shape, (1, 6))
            self.assertEqual(result[0, :4].tolist(), [450, 200, 550, 300])
            self.assertEqual(result[0, 5], 2)

    def test_failed_worker_blocks_blind_autoresume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'completion.json').write_text(json.dumps({'failed_tasks': ['vehicle']}))
            with self.assertRaisesRegex(RuntimeError, 'diagnose'):
                prepare_resume(root, {'vehicle': {}})
