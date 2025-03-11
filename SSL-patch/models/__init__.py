# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
from . import vision_transformer as vits
from torchvision import transforms

logger = logging.getLogger("dinov2")
def build_model_jb(device, gpu_num, ckpt_path):
    vit_kwargs = dict(
        img_size=224,
        patch_size=14,
        init_values=1.0e-05,
        ffn_layer='mlp',
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    teacher = vits.__dict__['vit_large'](**vit_kwargs)
    ckpt = torch.load(ckpt_path)['teacher']
    new_ckpt = {}
    for k, v in ckpt.items():
        if 'backbone' in k:
            k = '.'.join(k.split('.')[1:])
            new_ckpt[k] = v
    msg = teacher.load_state_dict(new_ckpt,strict=True)
    print(msg) 
    teacher.to(device)
    if gpu_num > 1:
        teacher = torch.nn.parallel.DataParallel(teacher)
    teacher.eval()
    return teacher, teacher.embed_dim


def build_model_jh(device, gpu_num, ckpt_path):
    vit_kwargs = dict(
        img_size=224,
        patch_size=14,
        init_values=1.0e-05,
        ffn_layer='swiglufused',
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    teacher = vits.__dict__['vit_large'](**vit_kwargs)
    ckpt = torch.load(ckpt_path)['teacher']
    new_ckpt = {}
    for k, v in ckpt.items():
        if 'backbone' in k:
            k = '.'.join(k.split('.')[1:])
            new_ckpt[k] = v
    msg = teacher.load_state_dict(new_ckpt,strict=True)
    print(msg) 
    teacher.to(device)
    if gpu_num > 1:
        teacher = torch.nn.parallel.DataParallel(teacher)
    teacher.eval()
    return teacher, teacher.embed_dim




def build_transform_jb():
    # Use timm's names
    # We prefer input size (512, 512), level 0;
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    return normalize


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def build_model(args, only_teacher=False, img_size=224):
    args.arch = remove_suffix(args.arch, "_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
