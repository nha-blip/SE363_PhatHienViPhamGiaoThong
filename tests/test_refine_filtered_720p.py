import unittest

from scripts.refine_filtered_720p import reasons


class StrictFilterTests(unittest.TestCase):
    def row(self, kind='vehicle', global_score=100, crop_score=300, box=None):
        return {'decision': 'KEEP', 'reasons': [], 'kind': kind,
                'metrics': {'global_laplacian_variance': global_score},
                'objects': [{'target_box': box or [0, 0, 40, 50], 'crop_laplacian_variance': crop_score}]}

    def test_visible_motion_blur_case_no_longer_passes(self):
        row = self.row(global_score=63.389, crop_score=21.343)
        self.assertIn('STRICT_OBJECT_BLUR_HEURISTIC', reasons(row))

    def test_dark_unclear_light_no_longer_passes(self):
        row = self.row('light', global_score=31.019, crop_score=10.132)
        self.assertIn('STRICT_GLOBAL_BLUR_HEURISTIC', reasons(row))

    def test_strong_large_object_is_kept(self):
        self.assertEqual(reasons(self.row()), [])

    def test_mixed_frame_rejected_without_dropping_boxes(self):
        row = self.row()
        row['objects'].append({'target_box': [100, 100, 110, 120], 'crop_laplacian_variance': 1000})
        self.assertIn('STRICT_SMALL_OBJECT', reasons(row))
        self.assertEqual(len(row['objects']), 2)

    def test_previous_rejection_remains_rejected(self):
        row = self.row()
        row.update(decision='REJECT', reasons=['NATIVE_BELOW_720P'])
        self.assertEqual(reasons(row), ['NATIVE_BELOW_720P'])


if __name__ == '__main__':
    unittest.main()
