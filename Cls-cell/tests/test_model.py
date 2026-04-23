from pathlib import Path
import sys
import unittest

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model import Classifier  # noqa: E402


class ModelTest(unittest.TestCase):
    def test_classifier_returns_probabilities_for_each_class(self):
        classifier = Classifier(in_dim=1024, n_classes=7)
        inputs = torch.randn(4, 1024)

        logits, probabilities = classifier(inputs)

        self.assertEqual(logits.shape, (4, 7))
        self.assertEqual(probabilities.shape, (4, 7))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
