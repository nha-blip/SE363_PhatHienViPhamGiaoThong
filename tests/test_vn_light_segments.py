import unittest

import numpy as np

from scripts import build_vn_light_segments as segments


def lamp(bgr, size=20):
    crop = np.zeros((60, 20, 3), np.uint8)
    crop[10:10 + size, 5:15] = bgr
    return crop


class LitStateTests(unittest.TestCase):
    def test_bright_colors_are_recognized(self):
        self.assertEqual(segments.lit_state(lamp((0, 0, 255))), 'red')
        self.assertEqual(segments.lit_state(lamp((0, 200, 255))), 'yellow')
        self.assertEqual(segments.lit_state(lamp((80, 255, 0))), 'green')

    def test_dark_head_is_unknown_not_off(self):
        self.assertEqual(segments.lit_state(np.full((60, 20, 3), 30, np.uint8)), 'unknown')

    def test_mixed_colors_are_unknown(self):
        crop = np.zeros((60, 20, 3), np.uint8)
        crop[5:25, 5:15] = (0, 0, 255)
        crop[35:55, 5:15] = (80, 255, 0)
        self.assertEqual(segments.lit_state(crop), 'unknown')

    def test_empty_crop_is_unknown(self):
        self.assertEqual(segments.lit_state(np.zeros((0, 0, 3), np.uint8)), 'unknown')


class SegmentTests(unittest.TestCase):
    def test_short_uncertain_gap_keeps_one_row_but_gap_frame_is_not_labeled(self):
        result = segments.state_segments([(0, 'red'), (1, 'red'), (2, 'unknown'), (3, 'red'), (4, 'green'), (5, 'lost')])
        self.assertEqual(result, [{'state': 'red', 'frames': [0, 1, 3]}, {'state': 'green', 'frames': [4]}])

    def test_long_uncertain_gap_starts_a_new_segment(self):
        observations = [(0, 'red')] + [(frame, 'unknown') for frame in range(1, 9)] + [(9, 'red')]
        self.assertEqual(segments.state_segments(observations, maximum_gap=6),
                         [{'state': 'red', 'frames': [0]}, {'state': 'red', 'frames': [9]}])

    def test_color_change_after_gap_is_not_merged(self):
        result = segments.state_segments([(0, 'red'), (1, 'unknown'), (2, 'green')])
        self.assertEqual([row['state'] for row in result], ['red', 'green'])

    def test_no_certain_frames_gives_no_segments(self):
        self.assertEqual(segments.state_segments([(0, 'lost'), (1, 'unknown')]), [])


class GroupHeadTests(unittest.TestCase):
    def track(self, identity, drawn, box):
        return {'id': identity, 'drawn': drawn, 'box': box, 'boxes': {0: box, 1: box}}

    def test_redrawn_head_merges_and_human_box_wins(self):
        groups = segments.group_heads([self.track('C_1', False, (451, 241, 460, 265)), self.track('C_2', True, (450, 238, 461, 268))])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['id'], 'C_2')
        self.assertEqual(groups[0]['merged_rows'], ['C_2', 'C_1'])

    def test_different_heads_stay_separate(self):
        groups = segments.group_heads([self.track('C_1', False, (10, 10, 20, 40)), self.track('C_2', False, (300, 10, 310, 40))])
        self.assertEqual(len(groups), 2)


class TrackBoxTests(unittest.TestCase):
    def scene(self, dx=0, dy=0):
        generator = np.random.default_rng(363)
        image = generator.integers(0, 255, (300, 500, 3), dtype=np.uint8)
        return np.roll(image, (dy, dx), axis=(0, 1))

    def test_box_stays_on_head_in_a_static_scene(self):
        box = (400, 100, 420, 160)
        result = segments.track_box({0: self.scene()}, self.scene(), box)
        self.assertEqual(result[0], box)

    def test_box_follows_a_camera_shift(self):
        result = segments.track_box({0: self.scene(dx=-30, dy=20)}, self.scene(), (400, 100, 420, 160))
        self.assertEqual(result[0], (370, 120, 390, 180))

    def test_unrelated_frame_is_lost_not_guessed(self):
        other = np.random.default_rng(1).integers(0, 255, (300, 500, 3), dtype=np.uint8)
        self.assertEqual(segments.track_box({0: other}, self.scene(), (400, 100, 420, 160)), {})


class ResolveRowsTests(unittest.TestCase):
    def test_only_kept_rows_become_heads(self):
        kept, rejected = segments.resolve_rows([
            {'id': 'a', 'camera': 'C', 'decision': 'KEEP', 'objects': [{}], 'notes': ''},
            {'id': 'b', 'camera': 'C', 'decision': 'NEEDS_FIX', 'objects': [{}], 'notes': 'ko phải đèn'},
            {'id': 'c', 'camera': 'C', 'decision': 'PENDING', 'objects': [{}], 'notes': 'ko có đèn'}])
        self.assertEqual([row['id'] for row in kept], ['a'])
        self.assertEqual([row['notes'] for row in rejected], ['ko phải đèn', 'ko có đèn'])


if __name__ == '__main__':
    unittest.main()
