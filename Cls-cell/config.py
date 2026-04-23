import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Cell classification fine-tuning for Smart-CCS")
    parser.add_argument("--backbone_model", default="CCS", choices=["CCS", "dinov2"], type=str)
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to the pretrained backbone checkpoint.")
    parser.add_argument("--classifier_ckpt", default=None, type=str, help="Path to a saved fine-tuned checkpoint.")
    parser.add_argument("--train_set", default=None, type=str)
    parser.add_argument("--val_set", default=None, type=str)
    parser.add_argument("--test_set", default=None, type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epoch", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--n_classes", default=7, type=int)
    parser.add_argument("--in_dim", default=1024, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()
