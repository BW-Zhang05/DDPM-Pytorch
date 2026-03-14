import random
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class SummaryWriterWrapper:
    def __init__(self, dir: str):
        self.writer = SummaryWriter(log_dir=dir)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def scalar(self, tag, value, step):
        self.writer.add_scalar(tag, float(value), step)

    def image(self, tag, image, step):
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        self.writer.add_image(tag, image, step, dataformats='HWC')

    def images(self, tag, images, step):
        self.image(tag, tile_imgs(images), step=step)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
    assert pad_pixels >= 0 and 0 <= pad_val <= 255

    imgs = np.asarray(imgs)
    assert imgs.dtype == np.uint8
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    n, h, w, c = imgs.shape
    assert c == 1 or c == 3, 'Expected 1 or 3 channels'

    if num_col <= 0:
        ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
        num_row = ceil_sqrt_n
        num_col = ceil_sqrt_n
    else:
        assert n % num_col == 0
        num_row = int(np.ceil(n / num_col))

    imgs = np.pad(
        imgs,
        pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)),
        mode='constant',
        constant_values=pad_val
    )
    h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
    imgs = imgs.reshape(num_row, num_col, h, w, c)
    imgs = imgs.transpose(0, 2, 1, 3, 4)
    imgs = imgs.reshape(num_row * h, num_col * w, c)

    if pad_pixels > 0:
        imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
    if c == 1:
        imgs = imgs[..., 0]
    return imgs


def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
    Image.fromarray(tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val, num_col=num_col)).save(filename)


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x: torch.Tensor, *, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(torch.clamp(cdf_delta, min=1e-12)))
    )
    return log_probs


def get_warmed_up_lr(max_lr: float, warmup: int, global_step: int) -> float:
    if warmup == 0:
        return max_lr
    return max_lr * min(float(global_step) / float(warmup), 1.0)


def rms(parameters: Iterable[torch.Tensor]) -> float:
    params = [p for p in parameters if p is not None]
    if not params:
        return 0.0
    sum_sq = 0.0
    count = 0
    for p in params:
        sum_sq += float(torch.sum(p.detach() ** 2))
        count += p.numel()
    return float(np.sqrt(sum_sq / max(count, 1)))
