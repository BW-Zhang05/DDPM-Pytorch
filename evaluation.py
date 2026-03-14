import os
import sys
from typing import Iterable, Optional, Tuple

import fire
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_tf import metrics
from diffusion_tf.data import get_dataset
from diffusion_tf.diffusion_utils import GaussianDiffusion, get_beta_schedule as get_beta_schedule_v1
from diffusion_tf.diffusion_utils_2 import GaussianDiffusion2, get_beta_schedule as get_beta_schedule_v2
from diffusion_tf.models.unet import UNet
from diffusion_tf.train_utils import load_checkpoint


def _build_model_cifar(*, image_size: int, in_ch: int, model_var_type: str, dropout: float):
    out_ch = in_ch * 2 if model_var_type == 'learned' else in_ch
    return UNet(
        in_ch=in_ch,
        out_ch=out_ch,
        ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=dropout,
        image_size=image_size,
    )


def _build_model_block(*, image_size: int, in_ch: int, block_size: int, dropout: float):
    effective_ch = in_ch * (block_size ** 2)
    effective_size = image_size // block_size
    return UNet(
        in_ch=effective_ch,
        out_ch=effective_ch,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=dropout,
        image_size=effective_size,
    )


def _denoise_block(model, x, t, block_size: int):
    if block_size != 1:
        x = F.pixel_unshuffle(x, block_size)
    out = model(x, t)
    if block_size != 1:
        out = F.pixel_shuffle(out, block_size)
    return out


def _iter_fake_batches(
    *,
    diffusion,
    denoise_fn,
    num_samples: int,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> Iterable[torch.Tensor]:
    remaining = num_samples
    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        sample_shape = (cur_bs, 3, image_size, image_size)
        samples = diffusion.p_sample_loop(denoise_fn, shape=sample_shape, device=device)
        yield samples
        remaining -= cur_bs


def _extract_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _with_progress(data: Iterable, *, total: int, desc: str):
    pbar = tqdm(total=total, desc=desc, unit='img', dynamic_ncols=True)
    try:
        for batch in data:
            x = _extract_batch(batch)
            bs = int(x.shape[0])
            pbar.update(bs)
            yield batch
    finally:
        pbar.close()


def evaluate(
    checkpoint,
    data_dir: Optional[str] = None,
    dataset: Optional[str] = None,
    image_size: Optional[int] = None,
    block_size: Optional[int] = None,
    num_samples: int = 1000,
    metrics_batch_size: int = 64,
    fid_num_samples: Optional[int] = None,
    is_num_samples: Optional[int] = None,
    fid_real_batch_size: int = 64,
    is_splits: int = 10,
    inception_weights: Optional[str] = None,
    eval_fid: int = 1,
    eval_is: int = 1,
    use_ema: int = 1,
    device: Optional[str] = None,
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt = load_checkpoint(checkpoint, map_location=device)
    config = ckpt.get('config', {})

    dataset = dataset or config.get('dataset')
    data_dir = data_dir or config.get('data_dir')

    if not eval_fid and not eval_is:
        raise ValueError('At least one of eval_fid or eval_is must be enabled.')

    fid_num = int(num_samples if fid_num_samples is None else fid_num_samples)
    is_num = int(num_samples if is_num_samples is None else is_num_samples)
    max_fake = max(fid_num if eval_fid else 0, is_num if eval_is else 0)

    use_cifar_path = ('model_mean_type' in config) or ('model_var_type' in config)

    if use_cifar_path:
        image_size = int(config.get('image_size', image_size or 32))
        model_var_type = config.get('model_var_type', 'fixedlarge')
        model_mean_type = config.get('model_mean_type', 'eps')
        loss_type = config.get('loss_type', 'mse')
        num_diffusion_timesteps = int(config.get('num_diffusion_timesteps', 1000))
        beta_start = float(config.get('beta_start', 0.0001))
        beta_end = float(config.get('beta_end', 0.02))
        beta_schedule = config.get('beta_schedule', 'linear')

        model = _build_model_cifar(
            image_size=image_size,
            in_ch=3,
            model_var_type=model_var_type,
            dropout=0.0,
        ).to(device)
        state = ckpt.get('ema') if use_ema and ('ema' in ckpt) else ckpt['model']
        model.load_state_dict(state)
        model.eval()

        diffusion = GaussianDiffusion2(
            betas=get_beta_schedule_v2(
                beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                num_diffusion_timesteps=num_diffusion_timesteps,
            ),
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
        )

        denoise_fn = lambda _x, _t: model(_x, _t)
    else:
        image_size = int(config.get('image_size', image_size or 256))
        block_size = int(config.get('block_size', block_size or 1))
        loss_type = config.get('loss_type', 'noisepred')
        num_diffusion_timesteps = int(config.get('num_diffusion_timesteps', 1000))
        beta_start = float(config.get('beta_start', 0.0001))
        beta_end = float(config.get('beta_end', 0.02))
        beta_schedule = config.get('beta_schedule', 'linear')

        model = _build_model_block(
            image_size=image_size,
            in_ch=3,
            block_size=block_size,
            dropout=0.0,
        ).to(device)
        state = ckpt.get('ema') if use_ema and ('ema' in ckpt) else ckpt['model']
        model.load_state_dict(state)
        model.eval()

        diffusion = GaussianDiffusion(
            betas=get_beta_schedule_v1(
                beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                num_diffusion_timesteps=num_diffusion_timesteps,
            ),
            loss_type=loss_type,
        ).to(device)

        denoise_fn = lambda _x, _t: _denoise_block(model, _x, _t, block_size)

    if eval_fid:
        if dataset is None or data_dir is None:
            raise ValueError('data_dir and dataset are required for FID evaluation.')

    print('Evaluation config:')
    print(f'  dataset: {dataset}')
    print(f'  data_dir: {data_dir}')
    print(f'  image_size: {image_size}')
    if not use_cifar_path:
        print(f'  block_size: {block_size}')
    print(f'  num_samples: {num_samples}')
    print(f'  fid_num_samples: {fid_num if eval_fid else "N/A"}')
    print(f'  is_num_samples: {is_num if eval_is else "N/A"}')
    print(f'  use_ema: {bool(use_ema and ("ema" in ckpt))}')
    print(f'  device: {device}')

    if eval_fid:
        ds, _ = get_dataset(dataset, data_dir, image_size, randflip=False)
        real_loader = DataLoader(
            ds,
            batch_size=fid_real_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        real_iter = _with_progress(real_loader, total=fid_num, desc='Real Inception')
    else:
        real_iter = None

    fake_iter = _iter_fake_batches(
        diffusion=diffusion,
        denoise_fn=denoise_fn,
        num_samples=max_fake,
        batch_size=metrics_batch_size,
        image_size=image_size,
        device=device,
    )
    fake_iter = _with_progress(fake_iter, total=max_fake, desc='Fake Inception')

    inception_model = metrics.load_inception_v3(device=device, weights_path=inception_weights)
    real_act = None
    if eval_fid:
        real_act, _ = metrics.collect_inception_outputs(
            inception_model,
            real_iter,
            device=device,
            max_samples=fid_num,
        )

    fake_act, fake_logits = metrics.collect_inception_outputs(
        inception_model,
        fake_iter,
        device=device,
        max_samples=max_fake,
    )

    fid = None
    if eval_fid:
        fid = metrics.compute_fid_from_activations(real_act, fake_act)

    is_score = None
    if eval_is:
        is_score = metrics.compute_inception_score_from_logits(fake_logits, splits=is_splits)

    if eval_fid:
        print(f'FID: {fid:.4f}')
    if eval_is and is_score is not None:
        is_mean, is_std = is_score
        print(f'Inception Score: {is_mean:.4f} ± {is_std:.4f}')


if __name__ == '__main__':
    fire.Fire()
