# from fastai.vision.all import Path, get_image_files
from typing import Any, Optional, Callable, Tuple
import os
from PIL import Image, ImageFile, UnidentifiedImageError
import pickle
from dinov2.data.datasets.extended import ExtendedVisionDataset
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

def quality_control(file_path, ratio_thresh=0.85):
    image = cv2.imread(file_path)
    b, g, r = cv2.split(image)
    count_greater_than_240_or_less_than_10 = ((b > 240) | (g > 240) | (r > 240) | (b < 10) | (g < 10) | (r < 10)).sum()
    current_ratio = count_greater_than_240_or_less_than_10/((image.shape[0]*image.shape[1]))
    return (current_ratio < ratio_thresh)

class RecursiveImageDatasetNas(ExtendedVisionDataset):
    def __init__(self,
                 root: str,
                 verify_images: bool = True,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)
        self.nas = {
            1: "/path/to/nas1/cervical/patches/",
            2: "/path/to/nas2/cervical/patches/",
            3: "/path/to/nas3/cervical/patches/"
        }
        with open(root, 'rb') as f:
            data = pickle.load(f)

        self.image_paths = data
        self.verify_images = verify_images

    def get_image_data(self, index: int) -> bytes:  # should return an image as an array

        image_path_line = self.image_paths[index]
        image_path = image_path_line[0]
        nas = image_path_line[1]
        path_base = self.nas[nas]
        image_path = os.path.join(path_base, image_path)
        try:
            img = Image.open(image_path).convert(mode="RGB")
        except (OSError, IOError, UnidentifiedImageError, RuntimeError) as e:
            raise RuntimeError(f"Cannot read image at {image_path}") from e
        return img

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        start_idx = index
        while True:
            try:
                image = self.get_image_data(index)
                if self.verify_images:
                    image_path_line = self.image_paths[index]
                    image_path = image_path_line[0]
                    nas = image_path_line[1]
                    path_base = self.nas[nas]
                    full_image_path = os.path.join(path_base, image_path)
                    if quality_control(full_image_path):
                        target = self.get_target(index)
                        if self.transforms is not None:
                            image, target = self.transforms(image, target)
                        return image, target
                    else:
                        print(f"Skipping image {full_image_path} due to quality control or corruption.")
                        index = (index + 1) % len(self.image_paths)
                        if index == start_idx:
                            raise RuntimeError("No valid images found in dataset")
                else:
                    target = self.get_target(index)
                    if self.transforms is not None:
                        image, target = self.transforms(image, target)
                    return image, target
            except (OSError, IOError, UnidentifiedImageError, RuntimeError) as e:
                print(f"Skipping image at index {index} due to an error: {e}")
                index = (index + 1) % len(self.image_paths)
                if index == start_idx:
                    raise RuntimeError("No valid images found in dataset")

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)