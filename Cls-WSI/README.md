# WSI-level Classification

## Installation

```bash
pip install -r requirements.txt
```

## Standard Inference

```bash
bash scripts/test.sh
```

## TTA Inference


**Step 1 – Build prototype memory bank** :

```bash
python build_memory_bank.py --config configs/tta_config.yaml --output path/to/proto_bank.pth
```

**Step 2 – TTA inference**:

```bash
bash scripts/test_TTA.sh
```
