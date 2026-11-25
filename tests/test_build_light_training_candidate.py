import tempfile
import unittest
from pathlib import Path

from scripts.build_light_training_candidate import choose_holdout, class_counts, repeat_count


class TrainingCandidateTests(unittest.TestCase):
    def test_atomic_groups_and_sources_in_both_partitions(self):
        rows = []
        for source in ('source_a', 'source_b'):
            for group in ('one', 'two'):
                for number in range(3):
                    rows.append({'source': source, 'sequence_group': source + '/' + group,
                                 'id': source + group + str(number),
                                 'class_counts': {'red': 2, 'yellow': 1, 'green': 2, 'off': 1}})
        holdout, score = choose_holdout(rows)
        self.assertEqual(holdout, choose_holdout(list(reversed(rows)))[0])
        self.assertGreaterEqual(score, 0)
        for source in ('source_a', 'source_b'):
            self.assertEqual(sum(group.startswith(source + '/') for group in holdout), 1)
        train = {row['id'] for row in rows if row['sequence_group'] not in holdout}
        val = {row['id'] for row in rows if row['sequence_group'] in holdout}
        self.assertFalse(train & val)

    def test_no_feasible_source_split_is_rejected(self):
        rows = [{'source': 'only', 'sequence_group': 'one', 'class_counts': {'red': 1}}]
        with self.assertRaises(ValueError):
            choose_holdout(rows)

    def test_repeat_is_bounded_per_image_not_per_object(self):
        self.assertEqual(repeat_count({'red': 100, 'green': 3}), 1)
        self.assertEqual(repeat_count({'yellow': 10, 'off': 20}), 3)
        self.assertEqual(repeat_count({}), 1)

    def test_label_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'label.txt'
            path.write_text('1 0.5 0.5 0.2 0.3\n3 0.5 0.5 0.1 0.1\n')
            self.assertEqual(class_counts(path), {'yellow': 1, 'off': 1})
            path.write_text('0 0.99 0.5 0.2 0.3\n')
            with self.assertRaises(ValueError):
                class_counts(path)


if __name__ == '__main__':
    unittest.main()
