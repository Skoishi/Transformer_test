from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 128
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def sequence_exact_match(pred: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> float:
    """Compute exact string match rate in a batch.

    pred, tgt: [B, T]
    """
    # Trim at EOS if present: here we simply compare padded sequences exactly
    mask = (tgt != pad_id)
    # For exact string match, we require equality on all non-pad positions and same length
    eq = (pred == tgt) | ~mask
    per_seq = eq.all(dim=1)
    return per_seq.float().mean().item()


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions (for decoder). Returns [1, size, size]."""
    attn_shape = (1, size, size)
    subsequent = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
    return ~subsequent


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






