import torch

from dinov2.models import build_model_CCS


def load_ccs_backbone(ckpt):
    if ckpt is None:
        raise ValueError("A CCS checkpoint path is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, embed_dim = build_model_CCS(device=device, gpu_num=1, ckpt_path=ckpt)
    return backbone, embed_dim


def load_dinov2_backbone():
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    backbone.eval()
    return backbone, 1024


def load_backbone(model_name, ckpt=None):
    if model_name == "CCS":
        return load_ccs_backbone(ckpt)
    if model_name == "dinov2":
        return load_dinov2_backbone()
    raise ValueError(f"Unsupported backbone model: {model_name}")
