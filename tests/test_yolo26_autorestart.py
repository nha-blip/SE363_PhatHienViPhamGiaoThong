import unittest

from scripts.autorestart_yolo26_kaggle2 import budgets, resume_reason, selected_member, PREFIX
from scripts.deploy_yolo26_fullscene_resume_20261028 import canonical_notebook


class AutoRestartTests(unittest.TestCase):
    def metadata(self, **overrides):
        return {'completed_epochs': 20, 'target_epochs': 100, 'patience': 15,
                'best_epoch': 18, 'optimizer_present': True, **overrides}

    def test_resume_budget_stop(self):
        self.assertEqual(resume_reason(self.metadata()), 'RESUME')

    def test_target_done_is_not_restarted(self):
        self.assertEqual(resume_reason(self.metadata(completed_epochs=100)), 'TARGET_REACHED')

    def test_early_stopping_is_not_overridden(self):
        self.assertEqual(resume_reason(self.metadata(completed_epochs=33)), 'EARLY_STOPPED')

    def test_patience_zero_disables_early_stop(self):
        self.assertEqual(resume_reason(self.metadata(patience=0, best_epoch=1)), 'RESUME')

    def test_missing_optimizer_and_empty_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            resume_reason(self.metadata(optimizer_present=False))
        with self.assertRaises(ValueError):
            resume_reason(self.metadata(completed_epochs=0))

    def test_quota_budget(self):
        self.assertIsNone(budgets(7199))
        for seconds in (7200, 3*3600, 20*3600):
            result = budgets(seconds)
            self.assertLess(result['train_budget_seconds'], result['session_budget_seconds'])
            self.assertLess(result['session_budget_seconds'], result['session_timeout_seconds'])
            self.assertLess(result['session_timeout_seconds'], seconds)

    def test_no_dataset_or_arbitrary_file_recovered(self):
        self.assertTrue(selected_member(PREFIX+'vehicle/checkpoints/last.pt', ['vehicle']))
        self.assertFalse(selected_member(PREFIX+'light/checkpoints/last.pt', ['vehicle']))
        self.assertFalse(selected_member(PREFIX+'vehicle/dataset/images/train/a.jpg', ['vehicle']))
        self.assertFalse(selected_member(PREFIX+'vehicle/../../escape.pt', ['vehicle']))

    def test_canonical_slug_comes_from_actual_url(self):
        self.assertEqual(canonical_notebook('https://www.kaggle.com/code/v5nhdf/actual-slug'), 'v5nhdf/actual-slug')
        with self.assertRaises(ValueError):
            canonical_notebook('https://example.org/code/v5nhdf/actual-slug')
