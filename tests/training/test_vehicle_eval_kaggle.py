import importlib.util
import io
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('vehicle_eval', ROOT / 'config/kaggle/yolo26_vehicle_eval_epoch27_20261030/eval.py')
vehicle_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vehicle_eval)


class VehicleEvalTest(unittest.TestCase):
    def test_match_frame_separates_missed_wrong_and_ok(self):
        truths = [((0, 0, 10, 10), 0), ((20, 0, 30, 10), 3), ((50, 0, 60, 10), 1)]
        predictions = [((0, 0, 10, 10), 0.9, 0), ((20, 0, 30, 10), 0.8, 0)]
        self.assertEqual(vehicle_eval.match_frame(truths, predictions), ['OK', 'WRONG', 'MISSED'])

    def test_each_truth_is_matched_once(self):
        truths = [((0, 0, 10, 10), 0)]
        predictions = [((0, 0, 10, 10), 0.9, 0), ((0, 0, 10, 10), 0.8, 0)]
        self.assertEqual(vehicle_eval.match_frame(truths, predictions), ['OK'])

    def test_low_overlap_is_missed(self):
        self.assertEqual(vehicle_eval.match_frame([((0, 0, 10, 10), 0)], [((8, 8, 18, 18), 0.9, 0)]), ['MISSED'])

    def test_match_detail_reports_predicted_class_and_unmatched(self):
        truths = [((0, 0, 10, 10), 0), ((20, 0, 30, 10), 3), ((50, 0, 60, 10), 1)]
        predictions = [((0, 0, 10, 10), 0.9, 3), ((20, 0, 30, 10), 0.8, 3), ((90, 0, 99, 10), 0.7, 2)]
        self.assertEqual(vehicle_eval.match_detail(truths, predictions), ([3, 3, None], [2]))

    def test_class_zero_prediction_counts_as_matched(self):
        truths = [((0, 0, 10, 10), 0)]
        predictions = [((0, 0, 10, 10), 0.9, 0), ((0, 0, 10, 10), 0.8, 1)]
        self.assertEqual(vehicle_eval.match_detail(truths, predictions), ([0], [1]))

    def test_size_bucket_uses_scaled_short_side(self):
        self.assertEqual(vehicle_eval.size_bucket((0, 0, 14, 100), 0.5), '<8px')
        self.assertEqual(vehicle_eval.size_bucket((0, 0, 16, 100), 0.5), '8-16px')
        self.assertEqual(vehicle_eval.size_bucket((0, 0, 64, 100), 0.5), '>=32px')

    def test_matrix_columns_are_true_classes(self):
        matrix = [[5, 0, 0, 2, 1], [0, 3, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 4, 0], [2, 7, 0, 3, 0]]
        summary = vehicle_eval.summarize_matrix(matrix, vehicle_eval.NAMES)
        self.assertEqual(summary['car']['total'], 8)
        self.assertEqual(summary['car']['predicted_as']['truck'], 1)
        self.assertEqual(summary['truck']['predicted_as']['car'], 2)
        self.assertEqual(summary['motorcycle']['missed_as_background'], 7)
        self.assertEqual(summary['car']['false_positive_from_background'], 1)

    def test_val_members_never_select_train_or_test(self):
        original = vehicle_eval.VAL_IMAGES
        vehicle_eval.VAL_IMAGES = 1
        self.addCleanup(setattr, vehicle_eval, 'VAL_IMAGES', original)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for name in ('vehicle/images/val/a.jpg', 'vehicle/labels/val/a.txt', 'vehicle/images/train/b.jpg',
                         'vehicle/labels/train/b.txt', 'vehicle/images/test/c.jpg', 'vehicle/labels/test/c.txt',
                         'vehicle/data.yaml'):
                archive.writestr(name, 'x')
        with zipfile.ZipFile(buffer) as archive:
            names = {member.filename for member in vehicle_eval.val_members(archive)}
        self.assertEqual(names, {'vehicle/images/val/a.jpg', 'vehicle/labels/val/a.txt'})


if __name__ == '__main__':
    unittest.main()
