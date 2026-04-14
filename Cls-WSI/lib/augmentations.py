import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF



def apply_rotation(imgs: torch.Tensor, degrees: float = 15.0) -> torch.Tensor:

    angle = random.uniform(-degrees, degrees)
    # TF.rotate supports 4-D tensors directly (fills with 0 = normalized black).
    return TF.rotate(imgs, angle)


def apply_scaling(imgs: torch.Tensor, scale_range: tuple = (0.9, 1.1)) -> torch.Tensor:

    scale = random.uniform(*scale_range)
    _, _, h, w = imgs.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    imgs = F.interpolate(imgs, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # center_crop pads with 0 if the resized image is smaller than (h, w)
    imgs = TF.center_crop(imgs, [h, w])
    return imgs


def apply_color_jitter(
    imgs: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.1,
    hue: float = 0.05,
) -> torch.Tensor:

    result = imgs.clone()
    for i in range(result.shape[0]):
        img = result[i]

        # Randomize the order of color transforms
        ops = list(range(4))
        random.shuffle(ops)

        for op in ops:
            if op == 0 and brightness > 0:          # brightness
                bf = 1.0 + random.uniform(-brightness, brightness)
                img = img * bf

            elif op == 1 and contrast > 0:           # contrast
                cf = 1.0 + random.uniform(-contrast, contrast)
                mean = img.mean()
                img = mean + (img - mean) * cf

            elif op == 2 and saturation > 0:         # saturation
                sf = 1.0 + random.uniform(-saturation, saturation)
                gray = img.mean(dim=0, keepdim=True)
                img = gray + (img - gray) * sf

            elif op == 3 and hue > 0:                # hue (channel-offset approx.)
                hf = random.uniform(-hue, hue)
                img = img + hf

        result[i] = img
    return result


def apply_gaussian_noise(imgs: torch.Tensor, std: float = 0.005) -> torch.Tensor:

    return imgs + torch.randn_like(imgs) * std



def apply_tta_augmentation(imgs: torch.Tensor, aug_cfg: dict = None) -> torch.Tensor:

    aug_cfg = aug_cfg or {}

    if torch.rand(1).item() > 0.5:
        imgs = torch.flip(imgs, dims=[-1])          # horizontal flip

    if torch.rand(1).item() > 0.5:
        imgs = torch.flip(imgs, dims=[-2])          # vertical flip

    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        imgs = torch.rot90(imgs, k=k, dims=[-2, -1])  # 90° multiples

    rot_cfg = aug_cfg.get("rotation", {})
    if rot_cfg.get("enabled", False):
        imgs = apply_rotation(imgs, degrees=rot_cfg.get("degrees", 15))

    scl_cfg = aug_cfg.get("scaling", {})
    if scl_cfg.get("enabled", False):
        imgs = apply_scaling(imgs, scale_range=tuple(scl_cfg.get("scale_range", [0.9, 1.1])))

    cj_cfg = aug_cfg.get("color_jitter", {})
    if cj_cfg.get("enabled", False):
        imgs = apply_color_jitter(
            imgs,
            brightness=cj_cfg.get("brightness", 0.2),
            contrast=cj_cfg.get("contrast",   0.2),
            saturation=cj_cfg.get("saturation", 0.1),
            hue=cj_cfg.get("hue",        0.05),
        )

    gn_cfg = aug_cfg.get("gaussian_noise", {})
    if gn_cfg.get("enabled", False):
        imgs = apply_gaussian_noise(imgs, std=gn_cfg.get("std", 0.005))

    return imgs
