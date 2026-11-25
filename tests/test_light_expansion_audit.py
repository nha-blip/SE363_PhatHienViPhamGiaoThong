import unittest

from scripts.audit_light_expansion import mapped_class, valid_box


class LightExpansionAuditTests(unittest.TestCase):
    def test_direct_classes(self):
        for name in ('red', 'yellow', 'green', 'off'):
            self.assertEqual(mapped_class(name), name)

    def test_arrow_color_mapping(self):
        self.assertEqual(mapped_class('arrow_straight_left_green'), 'green')
        self.assertEqual(mapped_class('circle_yellow'), 'yellow')

    def test_special_states_remain_unmapped(self):
        for name in ('wait_on', 'arrow_left_red_yellow', 'circle_red_yellow', 'unknown'):
            self.assertIsNone(mapped_class(name))

    def test_box_bounds(self):
        self.assertTrue(valid_box([0, 0, 100, 50], 100, 50))
        for box in ([3, 3, 3, 5], [-1, 0, 5, 5], [0, 0, 101, 50], [0, 0, float('nan'), 2]):
            self.assertFalse(valid_box(box, 100, 50))


if __name__ == '__main__':
    unittest.main()
