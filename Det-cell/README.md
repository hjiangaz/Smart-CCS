# Cell-level Detection

This section of code is built on [DDETR](https://github.com/fundamentalvision/Deformable-DETR).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Build the cytology cell detection dataset in [COCO 2017 format](https://cocodataset.org/#format-data):

```text
coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Training

```bash
sh ./configs/train.sh
```

Set multi-GPU training, `num_classes`, batch size, and dataset paths.

### Evaluation

```bash
sh ./configs/test.sh
```

### Inference

Input a manifest that lists all patch paths extracted from a WSI. Each record should follow this format:

```python
{
    'id': 0,
    'file_name': 'path_to_patch.jpg',
    'height': 1200,
    'width': 1200,
    'folder': 'patch folder'
}
```

Download the trained detector from [Drive](https://drive.google.com/drive/folders/1KbYIU5AjbTG8kIG-CjLM5aftvQtkgib3?usp=sharing), then run:

```bash
sh ./configs/extract.sh
```

The output stores detection results for each WSI in JSON format with `bbox`, `class`, and `score`, which avoids large intermediate storage and improves efficiency for WSI inference. Example output:

```json
{
  "image_id": 644610,
  "file_name": "path_to_patch.jpg",
  "label": [6],
  "score": [0.3639935255050659],
  "bbox": [[362.7785949707031, 876.9736938476562, 115.10406494140625, 170.00372314453125]]
}
```
