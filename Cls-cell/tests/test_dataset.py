from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset import CervixDataset  # noqa: E402


class DatasetTest(unittest.TestCase):
    def test_cervix_dataset_loads_image_and_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "cell.png"
            Image.new("RGB", (256, 256), color=(120, 80, 40)).save(image_path)

            split_file = tmp_path / "train.txt"
            split_file.write_text(f"{image_path} 2\n", encoding="utf-8")

            dataset = CervixDataset(str(split_file))
            image_tensor, label, source_path = dataset[0]

            self.assertEqual(tuple(image_tensor.shape), (3, 224, 224))
            self.assertEqual(label, 2)
            self.assertEqual(source_path, str(image_path))


if __name__ == "__main__":
    unittest.main()
