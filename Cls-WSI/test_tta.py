
import copy
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import yaml
import pandas as pd

from lib.models import load_backbone, load_classifier
from lib.datasets import get_dataloader
from lib.tta_ccs import CCSPrototypeAlignment
from lib.augmentations import apply_tta_augmentation


CLASS_NAMES = ["NILM", "ASC-US", "LSIL", "ASC-H", "HSIL", "SCC", "AGC"]



def parse_args():
    parser = argparse.ArgumentParser(description="Smart-CCS TTA Inference")
    parser.add_argument("--config", default='configs/tta_config.yaml', help="Path to TTA YAML config file.")
    return parser.parse_args()



def extract_features(backbone: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Extract backbone features from a single WSI's cell images (no gradient)."""
    imgs = imgs.cuda()
    with torch.no_grad():
        feats = backbone(imgs)
    return feats.cpu()


def create_output_folder(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)



def inference_tta(cfg: dict):
    """Run TTA inference over all test WSIs specified in cfg."""
    for k, v in cfg.items():
        print(f"  {k} = {v}")

    create_output_folder(cfg["output_folder"])

    backbone = load_backbone(model=cfg["backbone_model"], ckpt=cfg["checkpoint"])
    backbone.eval()

    classifier_ref = load_classifier(
        classifier_mode=cfg["classifier"],
        in_dim=cfg["in_dim"],
        n_classes=cfg["n_classes"],
        dropout=cfg.get("dropout"),
        act=cfg.get("act", "relu"),
        selection_K=cfg["selection_K"],
    )
    classifier_ref.load_state_dict(
        torch.load(cfg["classifier_ckpt"], map_location="cpu")
    )

    bank_path = cfg.get("memory_bank_path")
    if not bank_path or not os.path.exists(bank_path):
        raise FileNotFoundError(
            f"Prototype bank not found at '{bank_path}'. "
            "Run build_memory_bank.py first."
        )
    payload    = torch.load(bank_path, map_location="cpu")
    proto_bank = payload["proto_bank"]      # {class_id: tensor (w, feat_dim)}
    print(f"[SmartCCS-TTA] Prototype bank loaded from: {bank_path}")

    feat_dataloader = get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=4,
        selection_K=cfg["selection_K"],
        mode="infer",
        model=cfg["backbone_model"],
        dataset=cfg["dataset"],
        infer_set=cfg["infer_set"],
    )

    results = []

    test_batch_size = cfg.get("test_batch_size", 16)

    buf_feats:     list = []   
    buf_feats_aug: list = []
    buf_names:     list = []

    def _make_tta() -> CCSPrototypeAlignment:
        """Fresh student + teacher adapter, reset before every batch."""
        return CCSPrototypeAlignment(
            classifier   = copy.deepcopy(classifier_ref),
            proto_bank   = proto_bank,
            proj_dim     = cfg.get("proj_dim", 128),
            in_dim       = cfg.get("in_dim", 1024),
            temperature  = cfg.get("temperature", 0.1),
            ema_momentum = cfg.get("ema_momentum", 0.999),
            lr           = cfg.get("tta_lr", 1e-5),
            n_classes    = cfg["n_classes"],
            device       = "cuda",
        )

    def _flush_batch() -> None:
        """Run one joint adaptation + prediction step, append results."""
        tta        = _make_tta()
        probs_list = tta.adapt_batch_and_predict(buf_feats, buf_feats_aug)

        for slide_name, probs in zip(buf_names, probs_list):
            pred_idx       = torch.argmax(probs, dim=1).item()
            detailed_class = CLASS_NAMES[pred_idx]
            coarse_class   = "NOR" if detailed_class == "NILM" else "ABN"
            results.append([slide_name, coarse_class])
            if cfg.get("only_screening", False):
                print(f"  Slide ID: {slide_name}, Screening: {coarse_class}")
            else:
                print(f"  Slide ID: {slide_name}, Subtype: {detailed_class}, Screening: {coarse_class}")

        buf_feats.clear()
        buf_feats_aug.clear()
        buf_names.clear()

    for imgs, _, slide_path in feat_dataloader:
        slide_name = os.path.basename(slide_path[0]).split(".")[0]
        print(f"[SmartCCS-TTA] Processing: {slide_name}")

        cell_imgs = imgs[0]   # (n, 3, H, W)

        cell_feats     = extract_features(backbone, cell_imgs)              # (n, feat_dim)
        cell_feats_aug = extract_features(backbone,
                             apply_tta_augmentation(cell_imgs.clone()))     # (n, feat_dim)

        buf_feats.append(cell_feats)
        buf_feats_aug.append(cell_feats_aug)
        buf_names.append(slide_name)

        if len(buf_feats) >= test_batch_size:
            print(f"[SmartCCS-TTA] Adapting batch of {len(buf_feats)} WSIs ...")
            _flush_batch()

    if buf_feats:
        print(f"[SmartCCS-TTA] Adapting final batch of {len(buf_feats)} WSIs ...")
        _flush_batch()

    base_name    = os.path.splitext(os.path.basename(cfg["infer_set"]))[0]
    output_fname = f"{base_name}_Smart-CCS_TTA_results.xlsx"
    output_path  = os.path.join(cfg["output_folder"], output_fname)

    df = pd.DataFrame(results, columns=["Slide ID", "Screening Results"])
    df.to_excel(output_path, index=False)
    print(f"[SmartCCS-TTA] Results saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    inference_tta(cfg)
