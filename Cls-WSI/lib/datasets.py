import os.path as osp

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import json


class CCS_JSON_TOP(Dataset):
    """
    A dataset class for handling Cervix Whole Slide Imaging (WSI) data in JSON format.

    Args:
        mode (str): Mode of operation ('train', 'test', 'val', 'infer').
        model (str): Model type to be used (default is 'dinov2').
        train_set (str): Path to the training set JSON file.
        test_set (str): Path to the test set JSON file.
        val_set (str): Path to the validation set JSON file.
        infer_set (str): Path to the inference set JSON file.
        selection_K (int): Number of positive class labels to select (default is 100).
    """

    def __init__(self, mode="train", model="dinov2", selection_K=100, train_set=None, test_set=None, val_set=None, infer_set=None):
        super().__init__()

        # Store selection_K as an instance variable
        self.selection_K = selection_K

        # Select the appropriate label file based on the mode
        label_file = self._get_label_file(mode, train_set, test_set, val_set, infer_set)
        # Load the slide paths from the label file
        with open(label_file, encoding='utf-8') as f:
            self.total_slides = f.readlines()

        # Define transformations
        self.transform = self._get_transform()

    def _get_label_file(self, mode, train_set, test_set, val_set, infer_set):
        """Get the correct label file based on the mode."""
        if mode == "train":
            return train_set
        elif mode == "test":
            return test_set
        elif mode == "val":
            return val_set
        elif mode == "infer":
            return infer_set
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _get_transform(self):
        """Define normalization and transformations."""
        normalize = T.Normalize([0.5], [0.5])
        return T.Compose([
            T.ToTensor(),
            T.Resize(244),
            T.CenterCrop(224),
            normalize,
        ])

    def __len__(self):
        """Return the total number of slides in the dataset."""
        return len(self.total_slides)

    def __getitem__(self, index):
        """
        Retrieve the item at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[Tensor, int, str]: A tuple containing the cropped images tensor,
                                       the label, and the JSON path.
        """
        slide = self.total_slides[index]
        json_path, label = slide.split(' ')
        json_data = self._load_json_data(json_path)
        selected_data = self._select_data(json_data)

        cropped_images = self._crop_images(selected_data)

        if len(cropped_images) == 0:
            print("Empty:", json_path)
            final_img_dic = torch.zeros(1, 3, 224, 224)  # Return a zero tensor if no images were found
            return final_img_dic, int(label), json_path
        else:
            final_img_dic = torch.stack(cropped_images)  # Stack the images into a tensor
            return final_img_dic, int(label), json_path

    def _load_json_data(self, json_path):
        """Load and parse the JSON data from the specified path."""
        with open(json_path, "r") as file:
            data = json.load(file)

        json_data = []
        for item in data:
            file_name = item['file_name']
            labels = item['label']
            scores = item['score']
            bboxes = item['bbox']

            for i, l in enumerate(labels):
                json_data.append((file_name, labels[i], scores[i], bboxes[i]))

        return json_data

    def _select_data(self, json_data):
        """Select data based on positive class labels."""
        selected_data = []

        # Sort json_data by label and score
        json_data.sort(key=lambda x: (x[1], x[2]), reverse=True)

        for i in range(1, 7):  # Assuming class labels are from 1 to 6
            category_count = 0
            for js_data in json_data:
                if js_data[1] == i:
                    selected_data.append((js_data[0], js_data[3]))  # (file_name, bbox)
                    category_count += 1
                if category_count >= self.selection_K:  # Use the instance variable
                    break

        return selected_data

    def _crop_images(self, selected_data):
        """Crop images based on the selected data."""
        cropped_images = []
        for sele_data in selected_data:
            image_path = sele_data[0]
            image = Image.open(image_path)
            bbox = sele_data[1]

            # Extract bounding box coordinates
            x, y, w, h = bbox  # bbox is [x, y, width, height]
            x = max(0, x)
            y = max(0, y)
            w = min(1200 - x, w)
            h = min(1200 - y, h)

            # Crop and pad the image
            cropped_image = image.crop((x, y, x + w, y + h))
            width = int(w)
            height = int(h)
            edge_length = max(width, height)
            new_img = Image.new('RGB', (edge_length, edge_length), (0, 0, 0))
            new_img.paste(cropped_image, ((edge_length - width) // 2, (edge_length - height) // 2))

            # Transform the image and add to the list
            img = self.transform(new_img)
            cropped_images.append(img)

        return cropped_images


class CCS_feat_TOP(Dataset):
    def __init__(self, mode="train", train_set = None, test_set = None, val_set = None):
        super().__init__()

        if mode == "train":
            label_file = train_set
        elif mode == "test":
            label_file = test_set
        elif mode == "val":
            label_file = val_set

        with open(label_file, encoding='utf-8') as f:
            self.total_slides = f.readlines()

    def __len__(self):
        return len(self.total_slides)


    def __getitem__(self, index):

        slide = self.total_slides[index] 
        slide_path, label = slide.split(' ')
        feat = torch.load(slide_path)
        return feat, int(label), slide_path
    


class Cervix(Dataset):
    def __init__(self, mode="train", model="dinov2", train_set = None, test_set = None, val_set = None):
        super().__init__()

        if mode == "train":
            label_file = train_set
        elif mode == "test":
            label_file = test_set
        elif mode == "val":
            label_file = val_set


        with open(label_file, encoding='utf-8') as f:
            self.total_datas = f.readlines()


        normalize = T.Normalize([0.5], [0.5])

        self.transform = T.Compose([
                T.ToTensor(), 
                T.Resize(244), 
                T.CenterCrop(224), 
                normalize, 
            ])

    def __len__(self):
        return len(self.total_datas)
    

    def __getitem__(self, index):
        line = self.total_datas[index] 
        img_path, label = line.split(' ')
        img = Image.open(img_path)
        img = self.transform(img)
        return img, int(label), img_path


def get_dataloader(batch_size=16, shuffle=True, num_workers=4, selection_K=100, mode="train", 
                   model="dinov2", dataset=None, train_set=None, test_set=None, val_set=None, infer_set=None):
    """
    Create a DataLoader for the specified dataset.

    Args:
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        selection_K (int): Number of cells to select (specific to certain datasets).
        mode (str): Mode of operation (train, test, val, infer).
        model (str): Model type to be used (e.g., 'dinov2').
        dataset (str): Name of the dataset to load.
        train_set (str): Path to the training set.
        test_set (str): Path to the test set.
        val_set (str): Path to the validation set.
        infer_set (str): Path to the inference set.

    Returns:
        DataLoader: A DataLoader object for the specified dataset.
    """
    # Dataset selection based on the specified dataset name
    if dataset == 'CCS_JSON_TOP':
        dataset = CCS_JSON_TOP(mode, model, selection_K, train_set, test_set, val_set, infer_set)
    elif dataset == 'CCS_feat_TOP':
        dataset = CCS_feat_TOP(mode, train_set, test_set, val_set)
    elif dataset == 'Cervix':
        dataset = Cervix(mode, model, train_set, test_set, val_set)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create DataLoader from the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader
