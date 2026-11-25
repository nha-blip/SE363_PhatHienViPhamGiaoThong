import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import deploy_light_ab_20261031 as deploy


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('light_ab', ROOT / 'config/kaggle/yolo26_light_ab_20261031/train.py')
light_ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(light_ab)


class EntryTests(unittest.TestCase):
    def test_full_scene_entries_keep_repeats(self):
        root = Path('/data/light')
        entries = light_ab.absolute_entries(['./images/train/a.jpg', './images/train/a.jpg', '', './images/train/b.jpg'], root)
        self.assertEqual(entries, [str(root / 'images/train/a.jpg')] * 2 + [str(root / 'images/train/b.jpg')])

    def test_lisa_yellow_frames_repeat_three_times(self):
        with tempfile.TemporaryDirectory() as folder:
            labels = Path(folder)
            (labels / 'y.txt').write_text('1 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n')
            (labels / 'r.txt').write_text('0 0.5 0.5 0.1 0.1\n')
            entries = light_ab.lisa_entries([Path('/x/y.jpg'), Path('/x/r.jpg')], labels)
        self.assertEqual(entries, [str(Path('/x/r.jpg'))] + [str(Path('/x/y.jpg'))] * 3)

    def test_expected_b_sampler_size_follows_lisa_counts(self):
        self.assertEqual(light_ab.FULLSCENE_TRAIN_ENTRIES + light_ab.LISA_TRAIN_FRAMES + 2 * 88, 11030)


class ArchiveTests(unittest.TestCase):
    def archive(self, *names):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for name in names:
                archive.writestr(name, 'x')
        return zipfile.ZipFile(buffer)

    def test_accepts_expected_root(self):
        members = light_ab.checked_members(self.archive('a_lisa/images/train/a.jpg'), 'a_lisa')
        self.assertEqual([row.filename for row in members], ['a_lisa/images/train/a.jpg'])

    def test_rejects_other_root_and_traversal(self):
        with self.assertRaises(ValueError):
            light_ab.checked_members(self.archive('other/a.jpg'), 'a_lisa')
        with self.assertRaises(ValueError):
            light_ab.checked_members(self.archive('a_lisa/../../a.jpg'), 'a_lisa')


class ExperimentConfigTests(unittest.TestCase):
    def test_two_experiments_use_two_gpus_and_budget_is_ordered(self):
        self.assertEqual(sorted(light_ab.EXPERIMENTS.values()), [0, 1])
        self.assertLess(light_ab.TRAIN_BUDGET_SECONDS, light_ab.SESSION_BUDGET_SECONDS)
        self.assertLess(light_ab.SESSION_BUDGET_SECONDS, deploy.SESSION_TIMEOUT_SECONDS)

    def test_lisa_archive_excludes_manifests_and_uses_stage_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ('train', 'holdout'):
                for category, name in (('images', 'a.jpg'), ('labels', 'a.txt')):
                    (root / category / split).mkdir(parents=True)
                    (root / category / split / name).write_text('x')
            (root / 'manifest.json').write_text('{}')
            names = [name for _, name in deploy.lisa_members(root)]
        self.assertEqual(len(names), 4)
        self.assertTrue(all(name.startswith('a_lisa/') for name in names))


if __name__ == '__main__':
    unittest.main()
