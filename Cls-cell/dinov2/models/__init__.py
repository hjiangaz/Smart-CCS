import logging

import torch

from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model_CCS(device, gpu_num, ckpt_path):
    vit_kwargs = dict(
        img_size=224,
        patch_size=14,
        init_values=1.0e-05,
        ffn_layer="mlp",
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    teacher = vits.__dict__["vit_large"](**vit_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")["teacher"]
    new_ckpt = {}
    for key, value in ckpt.items():
        if "backbone" in key:
            key = ".".join(key.split(".")[1:])
            new_ckpt[key] = value
    teacher.load_state_dict(new_ckpt, strict=True)
    teacher.to(device)
    if gpu_num > 1:
        teacher = torch.nn.parallel.DataParallel(teacher)
    teacher.eval()
    return teacher, teacher.embed_dim
