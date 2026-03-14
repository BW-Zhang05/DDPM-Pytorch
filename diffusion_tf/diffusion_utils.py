import numpy as np
import torch
from torch import nn

from . import nn as nn_utils


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_like(shape, device, repeat=False, dtype=torch.float32):
    if repeat:
        noise = torch.randn((1, *shape[1:]), device=device, dtype=dtype)
        return noise.repeat(shape[0], *([1] * (len(shape) - 1)))
    return torch.randn(shape, device=device, dtype=dtype)


class GaussianDiffusion(nn.Module):
    def __init__(self, *, betas, loss_type):
        super().__init__()
        self.loss_type = loss_type

        assert isinstance(betas, np.ndarray)
        betas = betas.astype(np.float64) 
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.tensor(alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alphas_cumprod_prev', torch.tensor(alphas_cumprod_prev, dtype=torch.float32))

        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1. - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.tensor(np.log(1. - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.tensor(np.sqrt(1. / alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.tensor(np.sqrt(1. / alphas_cumprod - 1), dtype=torch.float32))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', torch.tensor(posterior_variance, dtype=torch.float32))
        self.register_buffer('posterior_log_variance_clipped', torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef1', torch.tensor(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod), dtype=torch.float32))
        self.register_buffer('posterior_mean_coef2', torch.tensor((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod), dtype=torch.float32))

    @staticmethod
    def _extract(a, t, x_shape):
        bs = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(bs, *([1] * (len(x_shape) - 1)))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, denoise_fn, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = denoise_fn(x_noisy, t)

        if self.loss_type == 'noisepred':
            losses = nn_utils.meanflat((noise - x_recon) ** 2)
        else:
            raise NotImplementedError(self.loss_type)
        return losses

    def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool):
        if self.loss_type == 'noisepred':
            x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_fn(x, t))
        else:
            raise NotImplementedError(self.loss_type)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, denoise_fn, *, x, t, clip_denoised=True, repeat_noise=False):
        model_mean, _, model_log_variance = self.p_mean_variance(denoise_fn, x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device=x.device, repeat=repeat_noise, dtype=x.dtype)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, *, shape, device=None):
        if device is None:
            device = next(denoise_fn.parameters()).device if hasattr(denoise_fn, 'parameters') else torch.device('cpu')
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=t)
        return img

    @torch.no_grad()
    def p_sample_loop_trajectory(self, denoise_fn, *, shape, device=None, repeat_noise_steps=-1):
        if device is None:
            device = next(denoise_fn.parameters()).device if hasattr(denoise_fn, 'parameters') else torch.device('cpu')
        imgs = []
        img = noise_like(shape, device=device, repeat=repeat_noise_steps >= 0, dtype=torch.float32)
        for i in reversed(range(self.num_timesteps)):
            repeat = repeat_noise_steps >= 0 and (self.num_timesteps - i) <= repeat_noise_steps
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=t, repeat_noise=repeat)
            imgs.append(img)
        times = torch.arange(self.num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
        return times, imgs

    @torch.no_grad()
    def interpolate(self, denoise_fn, *, x1, x2, lam: float, t: int):
        t_batched = torch.full((x1.shape[0],), int(t), device=x1.device, dtype=torch.long)
        xt1 = self.q_sample(x1, t=t_batched)
        xt2 = self.q_sample(x2, t=t_batched)
        xt_interp = (1 - lam) * xt1 + lam * xt2
        img = xt_interp
        for i in reversed(range(t + 1)):
            t_i = torch.full((x1.shape[0],), i, device=x1.device, dtype=torch.long)
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=t_i)
        return img
