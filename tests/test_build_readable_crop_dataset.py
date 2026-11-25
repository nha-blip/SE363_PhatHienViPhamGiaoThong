import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_readable_crop_dataset import load_decisions, qualifies, yolo_rows


class ReadableDatasetTests(unittest.TestCase):
    def test_small_unknown_vehicle_rejected(self):
        self.assertFalse(qualifies('vehicle', {'target_box': [0, 0, 20, 40], 'crop_laplacian_variance': 300}, None))

    def test_human_clear_can_override_size(self):
        self.assertTrue(qualifies('light', {'target_box': [0, 0, 3, 10]}, 'CLEAR'))

    def test_human_rejection_overrides_large_sharp_object(self):
        self.assertFalse(qualifies('vehicle', {'target_box': [0, 0, 200, 400], 'crop_laplacian_variance': 300}, 'UNREADABLE'))

    def test_full_box_normalization(self):
        rows = yolo_rows([{'class': 2, 'target_box': [10, 20, 30, 60]}], [0], [0, 0, 40, 80])
        self.assertEqual(rows.split(), ['2', '0.5000000000', '0.5000000000', '0.5000000000', '0.5000000000'])

    def test_partial_box_rejected(self):
        with self.assertRaises(ValueError):
            yolo_rows([{'class': 0, 'target_box': [0, 0, 50, 50]}], [0], [10, 10, 100, 100])

    def test_latest_crop_rejections_are_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / 'qa.json'
            crop = Path(folder) / 'crop.json'
            original.write_text(json.dumps({'reviews': [{'id': 'vehicle_134', 'audit_identity': 'test', 'object_index': 3, 'decision': 'CLEAR'}]}))
            crop.write_text(json.dumps({'status': 'COMPLETE_CROP_READABILITY_REVIEW', 'sample_reviews': [{'sample_id': 'vehicle_134', 'decision': 'REJECT_UNREADABLE'}]}))
            with patch('scripts.build_readable_crop_dataset.QA', original), patch('scripts.build_readable_crop_dataset.CROP_QA', crop), patch('scripts.build_readable_crop_dataset.FINAL_QA', Path(folder) / 'absent.json'):
                self.assertEqual(load_decisions()[('test', 3)], 'UNREADABLE')
