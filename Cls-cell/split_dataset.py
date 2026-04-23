import argparse
from pathlib import Path
import random


def split_entries(entries, seed=2020):
    entries = list(entries)
    random.Random(seed).shuffle(entries)

    total = len(entries)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.1)

    train_entries = entries[:train_end]
    val_entries = entries[train_end:val_end]
    test_entries = entries[val_end:]
    return train_entries, val_entries, test_entries


def write_split(entries, path):
    Path(path).write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Split a cell classification manifest into 7:1:2 text files")
    parser.add_argument("--input", required=True, type=str, help="Path to a manifest file with `image_path label` per line.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory used to save train.txt, val.txt, and test.txt.")
    parser.add_argument("--seed", default=2026, type=int)
    args = parser.parse_args()

    input_path = Path(args.input)
    entries = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_entries, val_entries, test_entries = split_entries(entries, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_split(train_entries, output_dir / "train.txt")
    write_split(val_entries, output_dir / "val.txt")
    write_split(test_entries, output_dir / "test.txt")

    print(f"Saved train/val/test splits to {output_dir}")


if __name__ == "__main__":
    main()
