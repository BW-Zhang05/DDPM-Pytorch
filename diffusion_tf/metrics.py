import math
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchvision import models


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_inception_v3(*, device: torch.device, weights_path: Optional[str] = None):
    if weights_path:
        model = models.inception_v3(weights=None)
        state = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state)
    else:
        try:
            from torchvision.models import Inception_V3_Weights

            model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except Exception:
            # Older torchvision fallback
            model = models.inception_v3(pretrained=True)
    model.eval().to(device)
    return model


def _prepare_inception_input(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    x = x.clamp(0.0, 1.0)
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _extract_batch(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _iter_batches(
    data: Iterable,
    *,
    max_samples: Optional[int] = None,
) -> Iterable[torch.Tensor]:
    seen = 0
    for batch in data:
        x = _extract_batch(batch)
        if max_samples is None:
            yield x
            continue
        remaining = max_samples - seen
        if remaining <= 0:
            break
        if x.shape[0] > remaining:
            x = x[:remaining]
        seen += x.shape[0]
        yield x


def collect_inception_outputs(
    model,
    data: Iterable,
    *,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    logits = []

    feature_buf = []

    def _hook(_module, _inp, output):
        feature_buf.append(output.detach())

    hook = model.avgpool.register_forward_hook(_hook)

    with torch.no_grad():
        for x in _iter_batches(data, max_samples=max_samples):
            x = x.to(device)
            x = _prepare_inception_input(x)
            _ = model(x)
            feat = feature_buf.pop(0)
            feat = torch.flatten(feat, 1)
            feats.append(feat.cpu().numpy())
            logits.append(_.cpu().numpy())

    hook.remove()

    feats = np.concatenate(feats, axis=0) if feats else np.empty((0, 2048), dtype=np.float32)
    logits = np.concatenate(logits, axis=0) if logits else np.empty((0, 1000), dtype=np.float32)
    return feats, logits


def compute_fid_from_activations(
    real_act: np.ndarray,
    fake_act: np.ndarray,
) -> float:
    mu1 = np.mean(real_act, axis=0)
    mu2 = np.mean(fake_act, axis=0)
    sigma1 = np.cov(real_act, rowvar=False)
    sigma2 = np.cov(fake_act, rowvar=False)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def compute_inception_score_from_logits(
    logits: np.ndarray,
    *,
    splits: int = 10,
    eps: float = 1e-16,
) -> Tuple[float, float]:
    if logits.shape[0] == 0:
        return float('nan'), float('nan')

    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    n = probs.shape[0]
    split_size = max(n // splits, 1)
    scores = []
    for i in range(0, n, split_size):
        part = probs[i:i + split_size]
        if part.shape[0] == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_fid_is(
    *,
    real_data: Optional[Iterable] = None,
    fake_data: Iterable,
    device: torch.device,
    max_real: Optional[int] = None,
    max_fake: Optional[int] = None,
    inception_weights: Optional[str] = None,
    compute_fid: bool = True,
    compute_is: bool = True,
    is_splits: int = 10,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    model = load_inception_v3(device=device, weights_path=inception_weights)

    real_act = None
    if compute_fid:
        if real_data is None:
            raise ValueError('real_data is required for FID.')
        real_act, _ = collect_inception_outputs(
            model,
            real_data,
            device=device,
            max_samples=max_real,
        )

    fake_act, fake_logits = collect_inception_outputs(
        model,
        fake_data,
        device=device,
        max_samples=max_fake,
    )

    fid = None
    if compute_fid:
        fid = compute_fid_from_activations(real_act, fake_act)

    is_score = None
    if compute_is:
        is_score = compute_inception_score_from_logits(fake_logits, splits=is_splits)

    return fid, is_score
