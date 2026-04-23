# Cell-level Classification

Download cell datasets such as [Herlev](https://mde-lab.aegean.gr/index.php/downloads/) and [SIPaKMeD](https://www.cs.uoi.gr/~marina/sipakmed.html), then prepare a manifest file with one sample per line:

```text
/path/to/image_001.png 0
/path/to/image_002.png 3
```

## Dataset Split (7:1:2)

```bash
bash scripts/split.sh
```

Or run the split script directly:

```bash
python split_dataset.py --input path/to/cells.txt --output_dir splits --seed 2020
```

This command generates `splits/train.txt`, `splits/val.txt`, and `splits/test.txt` using a 7:1:2 ratio.

## Pretrained Model

Download the pretrained CCS checkpoint from [Google Drive](https://drive.google.com/drive/folders/1KbYIU5AjbTG8kIG-CjLM5aftvQtkgib3?usp=sharing).

## Training

```bash
bash scripts/train.sh
```

## Testing

```bash
bash scripts/test.sh
```
