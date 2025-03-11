# Self-supervised pretraining for foundation models

This section of code contains tools for pretraining vision models on cervical image datasets across multiple NAS servers using DINOv2. For our case, we have three NAS servers with image patches that we want to use for pretraining. The custom data loader in this codebase reads images from these servers and prepares them for training.

## Overview

The project consists of two main components:

1. **Patch Collector Script**: Collects image patch paths from multiple directories and stores them in a pickle file
2. **Custom DataLoader**: A modified DINOv2 dataset class that reads images from multiple NAS servers

## Setup and Usage

### Step1: Collecting Image Patches

First, use `patch collector.py` script to gather all image paths from your NAS directories:

```bash
python patch_collector.py \
  --input_dirs /path/to/nas1/cervical/patches/ \
              /path/to/nas2/cervical/patches/ \
              /path/to/nas3/cervical/patches/ \
  --output_file /path/to/patch_path.pkl
```

This script:

- Walks through each input directory
- Finds all JPG images in the folders
- Records the relative path and source directory index (1, 2, or 3)
- Saves this information in a pickle file

### Step2. Configure the NAS Paths

The `RecursiveImageDatasetNas` class from `data/datasets/recursive_image_dataset.py` needs to know the base paths for each NAS index. Edit the class definition in line 28-30 in your codebase to match your NAS directory paths.

```python
...
        self.nas = {
            1: "/path/to/nas1/cervical/patches/",
            2: "/path/to/nas2/cervical/patches/",
            3: "/path/to/nas3/cervical/patches/"
        }
...
```

### Step3. Configure Training

Create a YAML configuration file for DINOv2 pretraining. The example config file is placed in `configs/train/vitl14_cervix.yaml`. Update the configuration file with your desired settings:

### Step4. Run Pretraining

Launch the DINOv2 pretraining with your configuration:

```bash
python dinov2/run/train/train.py \
    --nodes 2 \
    --config-file SSL-patch\configs\train\vitl14_cervix.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR>
```

## References

This implementation builds on [DINOv2](https://github.com/facebookresearch/dinov2) for self-supervised visual representation learning.
