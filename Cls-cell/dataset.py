from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class CervixDataset(Dataset):
    def __init__(self, split_file):
        self.split_file = Path(split_file)
        with self.split_file.open("r", encoding="utf-8") as handle:
            self.samples = [line.strip() for line in handle if line.strip()]

        normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(244),
            T.CenterCrop(224),
            normalize,
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index].rsplit(" ", 1)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, int(label), image_path


def build_dataloader(split_file, batch_size, shuffle, num_workers):
    dataset = CervixDataset(split_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
