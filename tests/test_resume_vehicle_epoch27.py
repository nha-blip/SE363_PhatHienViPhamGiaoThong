import unittest

from scripts.resume_vehicle_epoch27_20261030 import (
    EPOCH_SECONDS, FINAL_EVAL_SECONDS, STAGING_SECONDS, safe_budgets)


class SafeBudgetTests(unittest.TestCase):
    def test_last_started_epoch_finishes_before_watchdog(self):
        for free in (12860, 20000, 108000):
            result = safe_budgets(free)
            latest_end = STAGING_SECONDS + result['train_budget_seconds'] + EPOCH_SECONDS + FINAL_EVAL_SECONDS
            self.assertLessEqual(latest_end, result['session_budget_seconds'])
            self.assertLess(result['session_budget_seconds'], result['session_timeout_seconds'])
            self.assertLess(result['session_timeout_seconds'], free)

    def test_refuses_when_one_epoch_cannot_finish(self):
        self.assertIsNone(safe_budgets(7000))


if __name__ == '__main__':
    unittest.main()
