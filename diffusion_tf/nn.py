import math
import torch


def meanflat(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=list(range(1, x.ndim)))


def sumflat(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=list(range(1, x.ndim)))


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    if timesteps.ndim != 1:
        raise ValueError("timesteps should be 1-D")
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
