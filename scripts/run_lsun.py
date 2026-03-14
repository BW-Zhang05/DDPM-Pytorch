"""
LSUN (PyTorch)

python3 scripts/run_lsun.py train --data_dir /path/to/lsun --dataset lsun_church --exp_name lsun_church_ddpm
python3 scripts/run_lsun.py sample --checkpoint checkpoints/lsun_church_ddpm/step_10000.pt --dataset lsun_church
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fire
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_tf import utils
from diffusion_tf import metrics
from diffusion_tf.data import get_dataset
from diffusion_tf.diffusion_utils import get_beta_schedule, GaussianDiffusion
from diffusion_tf.models.unet import UNet
from diffusion_tf.train_utils import EMA, save_checkpoint, load_checkpoint


def _build_model(*, image_size, in_ch, block_size, dropout):
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


def _denoise_fn(model, x, t, block_size):
    if block_size != 1:
        x = F.pixel_unshuffle(x, block_size)
    out = model(x, t)
    if block_size != 1:
        out = F.pixel_shuffle(out, block_size)
    return out


def _iter_fake_batches(*, diffusion, denoise_fn, num_samples, batch_size, image_size, device):
    remaining = num_samples
    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        sample_shape = (cur_bs, 3, image_size, image_size)
        samples = diffusion.p_sample_loop(denoise_fn, shape=sample_shape, device=device)
        yield samples
        remaining -= cur_bs


def train(
    exp_name,
    data_dir,
    log_dir='logs',
    ckpt_dir='checkpoints',
    resume=None,
    dataset='lsun_church',
    optimizer='adam',
    batch_size=64,
    grad_clip=1.0,
    lr=2e-5,
    warmup=5000,
    num_diffusion_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule='linear',
    loss_type='noisepred',
    dropout=0.0,
    randflip=1,
    block_size=1,
    image_size=256,
    num_workers=4,
    max_steps=100000,
    log_interval=100,
    sample_interval=1000,
    save_interval=1000,
    ema_decay=0.9999,
    sample_batch_size=16,
    eval_fid=0,
    eval_is=0,
    fid_num_samples=1000,
    is_num_samples=1000,
    metrics_batch_size=16,
    fid_real_batch_size=16,
    is_splits=10,
    inception_weights=None,
    device=None,
):
    config = dict(
        exp_name=exp_name,
        data_dir=data_dir,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        resume=resume,
        dataset=dataset,
        optimizer=optimizer,
        batch_size=batch_size,
        grad_clip=grad_clip,
        lr=lr,
        warmup=warmup,
        num_diffusion_timesteps=num_diffusion_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        loss_type=loss_type,
        dropout=dropout,
        randflip=randflip,
        block_size=block_size,
        image_size=image_size,
        num_workers=num_workers,
        max_steps=max_steps,
        log_interval=log_interval,
        sample_interval=sample_interval,
        save_interval=save_interval,
        ema_decay=ema_decay,
        sample_batch_size=sample_batch_size,
        eval_fid=eval_fid,
        eval_is=eval_is,
        fid_num_samples=fid_num_samples,
        is_num_samples=is_num_samples,
        metrics_batch_size=metrics_batch_size,
        fid_real_batch_size=fid_real_batch_size,
        is_splits=is_splits,
        inception_weights=inception_weights,
        device=str(device) if device else None,
    )
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ds, _ = get_dataset(dataset, data_dir, image_size, bool(randflip))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    diffusion = GaussianDiffusion(
        betas=get_beta_schedule(beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps),
        loss_type=loss_type,
    ).to(device)

    model = _build_model(image_size=image_size, in_ch=3, block_size=block_size, dropout=dropout).to(device)
    ema = EMA(model, decay=ema_decay)

    if optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer}')

    step = 0
    if resume:
        ckpt = load_checkpoint(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            ema.shadow = ckpt['ema']
        if 'optimizer' in ckpt:
            optim.load_state_dict(ckpt['optimizer'])
        step = int(ckpt.get('step', 0))

    run_dir = os.path.join(log_dir, exp_name)
    ckpt_run_dir = os.path.join(ckpt_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    writer = utils.SummaryWriterWrapper(run_dir)

    inception_model = None
    real_act = None
    if eval_fid or eval_is:
        inception_model = metrics.load_inception_v3(device=device, weights_path=inception_weights)
        if eval_fid:
            real_loader = DataLoader(
                ds,
                batch_size=fid_real_batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
            )
            real_act, _ = metrics.collect_inception_outputs(
                inception_model,
                real_loader,
                device=device,
                max_samples=fid_num_samples,
            )

    model.train()
    data_iter = iter(dl)
    pbar = tqdm(total=max_steps, initial=step, dynamic_ncols=True, desc='train')
    while step < max_steps:
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, _ = next(data_iter)

        x = x.to(device)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        losses = diffusion.p_losses(lambda _x, _t: _denoise_fn(model, _x, _t, block_size), x_start=x, t=t)
        loss = losses.mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if warmup > 0:
            cur_lr = utils.get_warmed_up_lr(lr, warmup, step)
            for group in optim.param_groups:
                group['lr'] = cur_lr
        optim.step()
        ema.update(model)

        if step % log_interval == 0:
            writer.scalar('train/loss', float(loss.item()), step)
            writer.flush()
            pbar.set_postfix(
                loss=f'{loss.item():.4f}',
                lr=f"{optim.param_groups[0]['lr']:.2e}",
            )

        if step % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                sample_shape = (sample_batch_size, 3, image_size, image_size)
                samples = diffusion.p_sample_loop(
                    lambda _x, _t: _denoise_fn(model, _x, _t, block_size),
                    shape=sample_shape,
                    device=device,
                )
                samples = (samples.clamp(-1, 1) + 1) * 127.5
                samples = samples.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                utils.save_tiled_imgs(os.path.join(run_dir, f'samples_{step}.png'), samples)

                if eval_fid or eval_is:
                    max_fake = 0
                    if eval_fid:
                        max_fake = max(max_fake, int(fid_num_samples))
                    if eval_is:
                        max_fake = max(max_fake, int(is_num_samples))
                    fake_iter = _iter_fake_batches(
                        diffusion=diffusion,
                        denoise_fn=lambda _x, _t: _denoise_fn(model, _x, _t, block_size),
                        num_samples=max_fake,
                        batch_size=metrics_batch_size,
                        image_size=image_size,
                        device=device,
                    )
                    fake_act, fake_logits = metrics.collect_inception_outputs(
                        inception_model,
                        fake_iter,
                        device=device,
                        max_samples=max_fake,
                    )
                    if eval_fid:
                        fid = metrics.compute_fid_from_activations(real_act, fake_act)
                        writer.scalar('metrics/fid', fid, step)
                        pbar.write(f'[step {step}] FID: {fid:.4f}')
                    if eval_is:
                        is_mean, is_std = metrics.compute_inception_score_from_logits(fake_logits, splits=is_splits)
                        writer.scalar('metrics/is_mean', is_mean, step)
                        writer.scalar('metrics/is_std', is_std, step)
                        pbar.write(f'[step {step}] IS: {is_mean:.4f} ± {is_std:.4f}')
            model.train()

        if step % save_interval == 0:
            save_checkpoint(
                os.path.join(ckpt_run_dir, f'step_{step}.pt'),
                {
                    'model': model.state_dict(),
                    'ema': ema.shadow,
                    'optimizer': optim.state_dict(),
                    'step': step,
                    'config': config,
                },
            )

        step += 1
        pbar.update(1)

    pbar.close()
    writer.close()


def sample(
    checkpoint,
    out_dir='samples',
    num_samples=16,
    batch_size=16,
    data_dir=None,
    dataset='lsun_church',
    image_size=256,
    block_size=1,
    eval_fid=0,
    eval_is=0,
    fid_num_samples=None,
    is_num_samples=None,
    metrics_batch_size=16,
    fid_real_batch_size=16,
    is_splits=10,
    inception_weights=None,
    device=None,
):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt = load_checkpoint(checkpoint, map_location=device)
    config = ckpt.get('config', {})
    image_size = int(config.get('image_size', image_size))
    block_size = int(config.get('block_size', block_size))
    dataset = config.get('dataset', dataset)
    data_dir = config.get('data_dir', data_dir)
    num_diffusion_timesteps = int(config.get('num_diffusion_timesteps', 1000))
    beta_start = float(config.get('beta_start', 0.0001))
    beta_end = float(config.get('beta_end', 0.02))
    beta_schedule = config.get('beta_schedule', 'linear')
    loss_type = config.get('loss_type', 'noisepred')

    model = _build_model(image_size=image_size, in_ch=3, block_size=block_size, dropout=0.0).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    diffusion = GaussianDiffusion(
        betas=get_beta_schedule(beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps),
        loss_type=loss_type,
    ).to(device)

    os.makedirs(out_dir, exist_ok=True)
    all_samples = []
    cached_batches = []
    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            cur_bs = min(batch_size, remaining)
            sample_shape = (cur_bs, 3, image_size, image_size)
            samples = diffusion.p_sample_loop(
                lambda _x, _t: _denoise_fn(model, _x, _t, block_size),
                shape=sample_shape,
                device=device,
            )
            if eval_fid or eval_is:
                cached_batches.append(samples.cpu())
            samples = (samples.clamp(-1, 1) + 1) * 127.5
            samples = samples.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            all_samples.append(samples)
            remaining -= cur_bs

    all_samples = np.concatenate(all_samples, axis=0)
    utils.save_tiled_imgs(os.path.join(out_dir, 'samples.png'), all_samples)

    if eval_fid or eval_is:
        if data_dir is None:
            raise ValueError('data_dir is required for FID/IS evaluation.')
        fid_num = num_samples if fid_num_samples is None else int(fid_num_samples)
        is_num = num_samples if is_num_samples is None else int(is_num_samples)
        max_fake = max(fid_num if eval_fid else 0, is_num if eval_is else 0)

        def _fake_iter():
            yielded = 0
            for batch in cached_batches:
                if yielded >= max_fake:
                    break
                if yielded + batch.shape[0] > max_fake:
                    batch = batch[: max_fake - yielded]
                yielded += batch.shape[0]
                yield batch
            if yielded < max_fake:
                extra_iter = _iter_fake_batches(
                    diffusion=diffusion,
                    denoise_fn=lambda _x, _t: _denoise_fn(model, _x, _t, block_size),
                    num_samples=max_fake - yielded,
                    batch_size=metrics_batch_size,
                    image_size=image_size,
                    device=device,
                )
                for batch in extra_iter:
                    yield batch

        inception_model = metrics.load_inception_v3(device=device, weights_path=inception_weights)
        real_act = None
        if eval_fid:
            ds, _ = get_dataset(dataset, data_dir, image_size, randflip=False)
            real_loader = DataLoader(
                ds,
                batch_size=fid_real_batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            real_act, _ = metrics.collect_inception_outputs(
                inception_model,
                real_loader,
                device=device,
                max_samples=fid_num,
            )
        fake_act, fake_logits = metrics.collect_inception_outputs(
            inception_model,
            _fake_iter(),
            device=device,
            max_samples=max_fake,
        )
        if eval_fid:
            fid = metrics.compute_fid_from_activations(real_act, fake_act)
            print(f'FID: {fid:.4f}')
        if eval_is:
            is_mean, is_std = metrics.compute_inception_score_from_logits(fake_logits, splits=is_splits)
            print(f'Inception Score: {is_mean:.4f} ± {is_std:.4f}')


if __name__ == '__main__':
    fire.Fire()
