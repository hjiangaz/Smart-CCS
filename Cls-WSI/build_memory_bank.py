
import os
import argparse

import torch
import torch.nn.functional as F
import yaml

from lib.models import load_backbone, load_classifier
from lib.datasets import get_dataloader



def parse_args():
    parser = argparse.ArgumentParser(description="Build Smart-CCS TTA prototype memory bank")
    parser.add_argument("--config",  required=True,  help="Path to TTA YAML config file.")
    parser.add_argument("--output",  default=None,   help="Override output .pth path.")
    return parser.parse_args()



def build_memory_bank(cfg: dict, output_path: str):
    """Extract features from retrospective data and build class-wise prototype bank."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[SmartCCS-TTA] Building prototype memory bank ...")
    print(f"  Retrospective set : {cfg['retro_set']}")
    print(f"  Top-w per class   : {cfg['top_w']}")
    print(f"  Output path       : {output_path}")

    backbone = load_backbone(model=cfg["backbone_model"], ckpt=cfg["checkpoint"])
    backbone.eval()

    classifier = load_classifier(
        classifier_mode=cfg["classifier"],
        in_dim=cfg["in_dim"],
        n_classes=cfg["n_classes"],
        dropout=cfg.get("dropout"),
        act=cfg.get("act", "relu"),
        selection_K=cfg["selection_K"],
    )
    classifier.load_state_dict(torch.load(cfg["classifier_ckpt"], map_location="cpu"))
    classifier.to(device).eval()

    retro_loader = get_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=4,
        selection_K=cfg["selection_K"],
        mode="infer",
        model=cfg["backbone_model"],
        dataset=cfg["dataset"],
        infer_set=cfg["retro_set"],
    )

    all_probs      = []   
    all_mean_feats = []   

    with torch.no_grad():
        for imgs, _, slide_path in retro_loader:
            slide_name = os.path.basename(slide_path[0]).split(".")[0]
            print(f"  Processing: {slide_name}", end="\r")

            imgs = imgs[0].to(device)           
            feats = backbone(imgs)             
            mean_feat = feats.mean(dim=0)      

            logits, *_ = classifier(feats)    
            probs = F.softmax(logits, dim=1).squeeze(0)  

            all_probs.append(probs.cpu())
            all_mean_feats.append(mean_feat.cpu())

    print(f"\n  Total WSIs processed: {len(all_probs)}")

    all_probs      = torch.stack(all_probs,      dim=0)  
    all_mean_feats = torch.stack(all_mean_feats, dim=0)

    top_w     = cfg["top_w"]
    n_classes = cfg["n_classes"]
    proto_bank = {}

    for c in range(n_classes):
        class_probs  = all_probs[:, c]                 
        k            = min(top_w, len(class_probs))
        _, top_idx   = class_probs.topk(k, largest=True)
        proto_bank[c] = all_mean_feats[top_idx]           
        print(f"  Class {c}: selected {k} prototypes (max prob={class_probs.max():.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_payload = {
        "proto_bank": proto_bank,
        "n_classes":  n_classes,
        "top_w":      top_w,
        "feat_dim":   all_mean_feats.shape[1],
    }
    torch.save(save_payload, output_path)
    print(f"\n[SmartCCS-TTA] Prototype bank saved to: {output_path}")


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_path = args.output or cfg.get("memory_bank_path") or "proto_bank.pth"
    build_memory_bank(cfg, output_path)
