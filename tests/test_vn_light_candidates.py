import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts import prepare_vn_light_candidates as candidates
from scripts.review_light_dataset import ReviewStore


class SceneBandTests(unittest.TestCase):
    def test_picks_camera_view_over_app_header(self):
        rows = [80]*160 + [0]*400 + [120]*600 + [0]*300 + [60]*40 + [0]*100
        self.assertEqual(candidates.scene_band(rows), (560, 1160))

    def test_small_dark_gaps_do_not_split_the_scene(self):
        rows = [0]*10 + [100]*150 + [5]*2 + [100]*150 + [0]*10
        self.assertEqual(candidates.scene_band(rows), (10, 312))

    def test_black_screen_has_no_scene(self):
        self.assertIsNone(candidates.scene_band([0]*1600))


class StableBandTests(unittest.TestCase):
    def test_loading_frames_do_not_change_the_camera_view(self):
        bands = {0: (0, 1600), 25: (120, 1500), 50: (573, 1186), 75: (574, 1186), 100: (573, 1187), 125: (573, 1186)}
        band, frames = candidates.stable_band(bands, 1600)
        self.assertEqual(band, (573, 1186))
        self.assertEqual(frames, [50, 75, 100, 125])

    def test_full_page_layout_is_not_a_camera_view(self):
        self.assertEqual(candidates.stable_band({0: (0, 1600), 25: (0, 1600), 50: (0, 1600)}, 1600), (None, []))

    def test_map_page_reaching_screen_bottom_is_not_a_camera_view(self):
        bands = {0: (731, 1600), 25: (731, 1600), 50: (731, 1600), 75: (573, 1186), 100: (573, 1186), 125: (573, 1186)}
        self.assertEqual(candidates.stable_band(bands, 1600), ((573, 1186), [75, 100, 125]))
        self.assertEqual(candidates.stable_band({0: (731, 1600), 25: (731, 1600), 50: (731, 1600)}, 1600), (None, []))

    def test_black_video_has_no_camera_view(self):
        self.assertEqual(candidates.stable_band({0: None, 25: None, 50: None}, 1600), (None, []))

    def test_too_few_matching_frames_are_rejected(self):
        self.assertEqual(candidates.stable_band({0: (573, 1186), 25: (100, 500)}, 1600), (None, []))


class ClusterTests(unittest.TestCase):
    def test_overlapping_boxes_merge_into_one_head(self):
        rows = [(0, (760, 320, 815, 478), 0.7, 2), (0, (765, 284, 830, 471), 0.42, 2), (25, (762, 332, 814, 477), 0.6, 0)]
        clusters = candidates.cluster_detections(rows)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]['frames'], {0, 25})
        self.assertEqual(clusters[0]['class'], 2)

    def test_separate_heads_stay_separate(self):
        rows = [(0, (10, 10, 30, 60), 0.5, 0), (0, (300, 10, 320, 60), 0.5, 0)]
        self.assertEqual(len(candidates.cluster_detections(rows)), 2)

    def test_single_weak_frame_is_not_proposed(self):
        clusters = candidates.cluster_detections([(0, (10, 10, 30, 60), 0.22, 0)])
        self.assertEqual(candidates.select_clusters(clusters), [])

    def test_keeps_only_strongest_clusters(self):
        rows = [(frame, (position*100, 0, position*100+20, 50), 0.5, 0)
                for position in range(5) for frame in range(position + 1)]
        selected = candidates.select_clusters(candidates.cluster_detections(rows))
        self.assertEqual([len(row['frames']) for row in selected], [5, 4, 3])


class CameraClipTests(unittest.TestCase):
    def test_selected_first_and_rejected_cameras_skipped(self):
        clips, skipped = candidates.camera_clips({
            'C058.08-PTZ': {'choice': 'CAN_XEM_LAI', 'notes': 'nền đen'},
            'C069.10-PTZ': {'choice': 'THEO_DOI_DEN_VACH', 'notes': ''},
            'C070.10-PTZ': {'choice': 'BO_QUA', 'notes': ''}})
        self.assertEqual([row[1] for row in clips], ['C069.10-PTZ', 'C058.08-PTZ'])
        self.assertEqual(skipped, [])


class ReviewFlowTests(unittest.TestCase):
    def store(self, folder, row):
        path = Path(folder) / 'bbox_review.json'
        path.write_text(json.dumps({'status': 'PENDING_HUMAN_QA', 'is_ground_truth': False, 'allow_bbox_edit': True,
                                    'allow_class_edit': True, 'allow_focus_removal': True, 'reviews': [row]}), encoding='utf-8')
        return ReviewStore(path)

    def rows(self, folder, box, class_name):
        image = Path(folder) / 'frame.jpg'
        Image.new('RGB', (200, 100)).save(image)
        return candidates.review_row('C1_1', 'C1', image, image, box, class_name, 1.0, None, width=200, height=100)

    def test_missed_head_must_be_drawn_and_classified(self):
        with tempfile.TemporaryDirectory() as folder:
            store = self.store(folder, self.rows(folder, None, None))
            with self.assertRaises(ValueError):
                store.save(0, 'KEEP', 'SIGNAL_HEAD', False, '')
            store.save_geometry(0, 0, [10.0, 10.0, 30.0, 60.0])
            store.save_class(0, 'yellow')
            store.save(0, 'KEEP', 'SIGNAL_HEAD', True, '')
            saved = store.rows[0]
            self.assertEqual(saved['objects'][0]['proposed_class'], 'yellow')
            self.assertEqual(saved['objects'][0]['bbox_xyxy'], [10.0, 10.0, 30.0, 60.0])

    def test_no_missed_head_can_be_dismissed(self):
        with tempfile.TemporaryDirectory() as folder:
            store = self.store(folder, self.rows(folder, None, None))
            store.set_removed(0, True)
            store.save(0, 'REMOVE_BOX', 'UNREVIEWED', False, '')
            self.assertEqual(store.data['status'], 'HUMAN_QA_COMPLETED_REQUIRES_VALIDATION')

    def test_proposal_stays_non_ground_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            store = self.store(folder, self.rows(folder, [10.0, 10.0, 30.0, 60.0], 'red'))
            store.save(0, 'KEEP', 'SIGNAL_HEAD', True, '')
            self.assertFalse(store.data['is_ground_truth'])


if __name__ == '__main__':
    unittest.main()
