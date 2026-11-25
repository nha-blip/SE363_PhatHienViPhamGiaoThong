import copy
import unittest

from scripts.prepare_light_corrections import apply_action, yolo_lines


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.record = {'id': 'sample', 'width': 100, 'height': 50, 'objects': [
            {'source_class': 'off', 'proposed_class': 'off', 'bbox_xyxy': [10, 10, 30, 40], 'valid': True}]}

    def test_relabel_without_source_mutation(self):
        original = copy.deepcopy(self.record)
        result, log = apply_action(self.record, {'object_number': 1, 'action': 'CHANGE_CLASS', 'expected_class': 'off', 'new_class': 'red'})
        self.assertEqual(self.record, original)
        self.assertEqual(result['objects'][0]['proposed_class'], 'red')
        self.assertEqual(log['before']['source_class'], 'off')
        self.assertEqual(yolo_lines(result), ['0 0.2000000000 0.5000000000 0.2000000000 0.6000000000'])

    def test_removal_does_not_create_automatic_negative(self):
        result, log = apply_action(self.record, {'object_number': 1, 'action': 'REMOVE_NON_TRAFFIC_OBJECT', 'expected_class': 'off'})
        self.assertIsNone(log['after'])
        with self.assertRaises(ValueError):
            yolo_lines(result)

    def test_clamp_only_small_rounding(self):
        self.record['issues'] = ['INVALID_BBOX']
        self.record['objects'][0]['valid'] = False
        self.record['objects'][0]['bbox_xyxy'][0] = -0.001296
        action = {'object_number': 1, 'action': 'CLAMP_ROUNDING_AT_IMAGE_EDGE', 'expected_class': 'off'}
        result, _ = apply_action(self.record, action)
        self.assertEqual(result['objects'][0]['bbox_xyxy'][0], 0)
        self.assertEqual(result['source_audit_issues'], ['INVALID_BBOX'])
        self.assertNotIn('INVALID_BBOX', result['issues'])
        self.assertTrue(result['requires_correction_review'])
        self.record['objects'][0]['bbox_xyxy'][0] = -1
        with self.assertRaises(ValueError):
            apply_action(self.record, action)

    def test_unmapped_object_blocks_whole_image(self):
        self.record['objects'].append({'proposed_class': None, 'bbox_xyxy': [1, 1, 2, 2]})
        with self.assertRaises(ValueError):
            yolo_lines(self.record)

    def test_zero_height_is_not_exported(self):
        self.record['objects'][0]['bbox_xyxy'] = [10, 20, 30, 20]
        with self.assertRaises(ValueError):
            yolo_lines(self.record)

    def test_source_precondition(self):
        with self.assertRaises(ValueError):
            apply_action(self.record, {'object_number': 1, 'action': 'REMOVE_NON_TRAFFIC_OBJECT', 'expected_class': 'red'})


if __name__ == '__main__':
    unittest.main()
