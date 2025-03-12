from fastai.vision.all import Path, get_image_files, verify_images

from typing import Any, Optional, Callable, Tuple
import os
from PIL import Image
import pickle
from dinov2.data.datasets.extended import ExtendedVisionDataset



class RecursiveImageDatasetNas(ExtendedVisionDataset):
    def __init__(self,
                #  root_common: "/jhcnas1/jincheng/cervical/patches/",
                 root: str,
                 verify_images: bool = False,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)
        self.nas = {
            1: "/jhcnas1/jincheng/cervical/patches/",
            2: "/jhcnas2/home/jincheng/cervical/patches/",
            3: "/jhcnas3/Cervical/CervicalData_NEW/Processed_Data/PATCH_DATA/Patch_1200_All/patches/"
        }
        with open(root, 'rb') as f:
            data = pickle.load(f)

        self.image_paths = data
        # self.root = Path(root).expanduser()
        # image_paths = get_image_files(self.root)
        # invalid_images = set()
        # if verify_images:
        #     print("Verifying images. This ran at ~100 images/sec/cpu for me. Probably depends heavily on disk perf.")
        #     invalid_images = set(verify_images(image_paths))
        #     print("Skipping invalid images:", invalid_images)
        # self.image_paths = [p for p in image_paths if p not in invalid_images]


    def get_image_data(self, index: int) -> bytes:  # should return an image as an array

        image_path_line = self.image_paths[index]
        image_path = image_path_line[0]
        nas = image_path_line[1]
        path_base = self.nas[nas]
        image_path = os.path.join(path_base, image_path)
        img = Image.open(image_path).convert(mode="RGB")

        return img

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)
    





class RecursiveImageDataset(ExtendedVisionDataset):
    def __init__(self,
                #  root_common: "/jhcnas1/jincheng/cervical/patches/",
                 root: str,
                 verify_images: bool = False,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:

        super().__init__(root, transforms, transform, target_transform)
        self._root_common =  "/jhcnas1/jincheng/cervical/patches/"
        self.root = Path(root).expanduser()
        image_paths = get_image_files(self.root)
        invalid_images = set()
        if verify_images:
            print("Verifying images. This ran at ~100 images/sec/cpu for me. Probably depends heavily on disk perf.")
            invalid_images = set(verify_images(image_paths))
            print("Skipping invalid images:", invalid_images)
        self.image_paths = [p for p in image_paths if p not in invalid_images]


    def get_image_data(self, index: int) -> bytes:  # should return an image as an array

        image_path = self.image_paths[index]
        image_path = os.path.join(self._root_common, image_path)
        img = Image.open(image_path).convert(mode="RGB")

        return img

    def get_target(self, index: int) -> Any:
        return 0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)