from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from split_dataset import split_entries  # noqa: E402


class SplitDatasetTest(unittest.TestCase):
    def test_split_entries_uses_7_1_2_ratio(self):
        entries = [f"sample_{idx} {idx % 3}" for idx in range(10)]

        train_entries, val_entries, test_entries = split_entries(entries, seed=7)

        self.assertEqual(len(train_entries), 7)
        self.assertEqual(len(val_entries), 1)
        self.assertEqual(len(test_entries), 2)
        self.assertEqual(sorted(train_entries + val_entries + test_entries), sorted(entries))


if __name__ == "__main__":
    unittest.main()
