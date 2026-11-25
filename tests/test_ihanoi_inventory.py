import unittest

from scripts.inventory_ihanoi_cameras import camera_rows, unopened_section


class IhanoiInventoryTests(unittest.TestCase):
    def test_expands_visible_sections_only_once(self):
        xml = '<hierarchy><node content-desc="Phường Tây Hồ" clickable="true" bounds="[0,400][900,500]"/></hierarchy>'
        self.assertEqual(unopened_section(xml, set()), ('Phường Tây Hồ', 450, 450))
        self.assertIsNone(unopened_section(xml, {'Phường Tây Hồ'}))

    def test_camera_identity_and_name(self):
        xml = '<hierarchy><node content-desc="Camera giao thông"/><node clickable="true" content-desc="Road - C069.10-PTZ&#10;Address" bounds="[0,400][900,600]"/></hierarchy>'
        rows = camera_rows(xml)
        self.assertEqual(list(rows), ['C069.10-PTZ'])
        self.assertEqual(rows['C069.10-PTZ']['address'], 'Address')

    def test_deduplicates_visible_camera(self):
        xml = '<hierarchy><node content-desc="Camera giao thông"/><node clickable="true" content-desc="C069.10-PTZ"/><node clickable="true" content-desc="C069.10-PTZ"/></hierarchy>'
        self.assertEqual(len(camera_rows(xml)), 1)

    def test_stops_on_other_app_screen(self):
        with self.assertRaisesRegex(ValueError, 'listing'):
            camera_rows('<hierarchy><node content-desc="Login"/></hierarchy>')
