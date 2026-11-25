import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts.review_ihanoi_survey import CHOICE_LABELS, LocalVideo, Review, save_choice


class IhanoiReviewTests(unittest.TestCase):
    def test_notes_save_preserves_other_camera_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'choices.json'
            path.write_text(json.dumps({'OLD': {'choice': 'CHI_TRACKING', 'notes': 'old'}}))
            result = save_choice(path, 'NEW', 'THEO_DOI_DEN_VACH', 'Đèn và vạch rõ')
            self.assertEqual(result['OLD']['notes'], 'old')
            self.assertEqual(result['NEW']['notes'], 'Đèn và vạch rõ')
            self.assertFalse(result['NEW']['is_ground_truth'])
            with self.assertRaises(ValueError):
                save_choice(path, 'NEW', 'INVALID', '')

    def test_local_video_read_seek_and_close(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (96, 64))
            if not writer.isOpened():
                self.skipTest('MJPG writer unavailable')
            for index in range(30):
                writer.write(np.full((64, 96, 3), index*5, dtype=np.uint8))
            writer.release()
            reader = LocalVideo(path)
            try:
                self.assertAlmostEqual(reader.duration, 3)
                times = [reader.read()[1] for attempt in range(5)]
                self.assertEqual(times, sorted(times))
                reader.seek(1.5)
                frame, seconds = reader.read()
                self.assertEqual(frame.shape, (64, 96, 3))
                self.assertAlmostEqual(seconds, 1.5, places=1)
            finally:
                reader.close()
            self.assertFalse(reader.capture.isOpened())

    def test_failed_video_open_releases_handle(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch('scripts.review_ihanoi_survey.cv2.VideoCapture', return_value=capture):
            with self.assertRaises(ValueError):
                LocalVideo('missing.mp4')
        capture.release.assert_called_once()

    def test_ui_notes_column_and_autosave_on_camera_change(self):
        try:
            window = tk.Tk()
        except tk.TclError:
            self.skipTest('Tk display unavailable')
        window.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                (output/'survey_index.json').write_text(json.dumps({'cameras': [
                    {'camera_id': 'A', 'name': 'Camera A', 'status': 'WAITING_SURVEY'},
                    {'camera_id': 'B', 'name': 'Camera B', 'status': 'WAITING_SURVEY'}]}))
                with patch('scripts.review_ihanoi_survey.OUTPUT', output):
                    app = Review(window)
                    window.update()
                    self.assertIn('notes', app.table['columns'])
                    app.table.selection_set('A')
                    app.selected()
                    app.choice.set(CHOICE_LABELS['THEO_DOI_DEN_VACH'])
                    app.note.set('Đèn rõ, có vạch')
                    app.table.selection_set('B')
                    app.selected()
                    saved = json.loads(app.path.read_text(encoding='utf-8'))
                    self.assertEqual(saved['A']['notes'], 'Đèn rõ, có vạch')
                    self.assertEqual(saved['A']['choice'], 'THEO_DOI_DEN_VACH')
                    self.assertEqual(app.table.set('A', 'notes'), 'Đèn rõ, có vạch')
                    app.choose('BO_QUA')
                    self.assertEqual(app.table.set('B', 'choice'), 'Loại')
                    app.close_video()
        finally:
            window.destroy()
