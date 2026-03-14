import numpy as np
import torch
from torch import nn

from . import nn as nn_utils
from . import utils


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


class GaussianDiffusion2(nn.Module):
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):
        super().__init__()
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        assert isinstance(betas, np.ndarray)
        betas = betas.astype(np.float64) 
        self.betas = betas # 每一步的加噪系数
        assert (betas > 0).all() and (betas <= 1).all() # 0-1之间
        timesteps, = betas.shape # 加噪的步数
        self.num_timesteps = int(timesteps)

        alphas = 1. - betas # 每一步的alpha系数,这里alphas是alpha的平方
        self.alphas_cumprod = np.cumprod(alphas, axis=0) # 累乘 (alpha1 * ... alpha_t)^2
        self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1]) # 累乘 (alpha1 * ... alpha_(t-1))^2

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) # 累乘 alpha1 * ... alpha_t
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod) # 根号下(1-(alpha1 * ... alpha_t)^2)
        self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)

        # 反向过程用到的系数
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape): # 根据时间步 t,从系数数组 a 里取出对应的值,并reshape成可以和图片 x 直接计算的形状
        bs = t.shape[0]
        a = torch.as_tensor(a, device=t.device, dtype=torch.float32)
        out = a.gather(0, t)
        return out.reshape(bs, *([1] * (len(x_shape) - 1)))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None): # 对应加噪公式
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool, return_pred_xstart: bool):
        model_output = denoise_fn(x, t)

        if self.model_var_type == 'learned':
            B, C, H, W = x.shape
            model_output, model_log_variance = torch.chunk(model_output, 2, dim=1)
            model_variance = torch.exp(model_log_variance)
        elif self.model_var_type in ['fixedsmall', 'fixedlarge']:
            model_variance, model_log_variance = {
                'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
                'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, x.shape).expand_as(x)
            model_log_variance = self._extract(model_log_variance, t, x.shape).expand_as(x)
        else:
            raise NotImplementedError(self.model_var_type)

        def _maybe_clip(x_):
            return torch.clamp(x_, -1., 1.) if clip_denoised else x_

        if self.model_mean_type == 'xprev':
            pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type == 'xstart':
            pred_xstart = _maybe_clip(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        elif self.model_mean_type == 'eps':
            pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, pred_xstart
        return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        return (
            self._extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            self._extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def p_sample(self, denoise_fn, *, x, t, clip_denoised=True, return_pred_xstart: bool):
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, x=x, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return (sample, pred_xstart) if return_pred_xstart else sample

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, *, shape, device=None):
        if device is None:
            device = next(denoise_fn.parameters()).device if hasattr(denoise_fn, 'parameters') else torch.device('cpu')
        img = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(denoise_fn=denoise_fn, x=img, t=t, return_pred_xstart=False)
        return img

    @torch.no_grad()
    def p_sample_loop_progressive(self, denoise_fn, *, shape, device=None, include_xstartpred_freq=50):
        if device is None:
            device = next(denoise_fn.parameters()).device if hasattr(denoise_fn, 'parameters') else torch.device('cpu')
        img = torch.randn(shape, device=device)
        num_recorded_xstartpred = self.num_timesteps // include_xstartpred_freq
        xstartpreds = torch.zeros((shape[0], num_recorded_xstartpred, *shape[1:]), device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            sample, pred_xstart = self.p_sample(denoise_fn=denoise_fn, x=img, t=t, return_pred_xstart=True)
            img = sample
            if i % include_xstartpred_freq == 0:
                idx = i // include_xstartpred_freq
                if idx < num_recorded_xstartpred:
                    xstartpreds[:, idx] = pred_xstart
        return img, xstartpreds

    def _vb_terms_bpd(self, denoise_fn, x_start, x_t, t, *, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, x=x_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = nn_utils.meanflat(kl) / np.log(2.)

        decoder_nll = -utils.discretized_gaussian_log_likelihood(x_start, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = nn_utils.meanflat(decoder_nll) / np.log(2.)

        output = torch.where(t == 0, decoder_nll, kl)
        return (output, pred_xstart) if return_pred_xstart else output

    def training_losses(self, denoise_fn, x_start, t, noise=None): # x_start 原始图像
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise) # 加噪到t时刻后的图像

        if self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, return_pred_xstart=False)
        elif self.loss_type == 'mse':
            assert self.model_var_type != 'learned'
            target = {
                'xprev': self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                'xstart': x_start,
                'eps': noise
            }[self.model_mean_type]
            model_output = denoise_fn(x_t, t) # 进入U-Net
            losses = nn_utils.meanflat((target - model_output) ** 2)
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def _prior_bpd(self, x_start):
        B, T = x_start.shape[0], self.num_timesteps
        t = torch.full((B,), T - 1, device=x_start.device, dtype=torch.long)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.)
        return nn_utils.meanflat(kl_prior) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, *, clip_denoised=True):
        B, T = x_start.shape[0], self.num_timesteps
        terms_bpd = torch.zeros((B, T), device=x_start.device)
        mse_bt = torch.zeros((B, T), device=x_start.device)

        for t_ in reversed(range(T)):
            t_b = torch.full((B,), t_, device=x_start.device, dtype=torch.long)
            new_vals_b, pred_xstart = self._vb_terms_bpd(
                denoise_fn, x_start=x_start, x_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
                clip_denoised=clip_denoised, return_pred_xstart=True)
            new_mse_b = nn_utils.meanflat((pred_xstart - x_start) ** 2)
            terms_bpd[:, t_] = new_vals_b
            mse_bt[:, t_] = new_mse_b

        prior_bpd_b = self._prior_bpd(x_start)
        total_bpd_b = torch.sum(terms_bpd, dim=1) + prior_bpd_b
        return total_bpd_b, terms_bpd, prior_bpd_b, mse_bt
