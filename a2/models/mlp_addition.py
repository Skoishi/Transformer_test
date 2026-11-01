from __future__ import annotations

"""Simple non-seq2seq MLP baseline.

This model treats each input example as two integers (a,b) parsed from the source
string form "a+b" (without spaces) and predicts the sum string all at once.

Approach:
- Parse source tokens (digit chars plus '+'). Extract the two numbers.
- Build a fixed-size input vector from (a, b). Two modes (``MLPConfig.input_mode``):
    "onehot" (default): each of the ``max_digits`` decimal digits of a and of b as a
        one-hot over 0-9, i.e. 2 * max_digits * 10 binary inputs. This is the raw
        input with no hand-crafted structure, so what the network learns is entirely
        up to it -- the regime of Zhang et al. (2017), "Understanding deep learning
        requires rethinking generalization".
    "scalar": a small hand-crafted feature vector
        [ log10(a+1), log10(b+1), log10(a+b+1), a/10^max_digits, b/10^max_digits ]
        plus optional per-digit count features (counts of each digit in a and in b).
        Note that log10(a+b+1) leaks the answer's magnitude.
- A feed-forward network (``num_layers`` x [Linear, ReLU]) maps the input to a hidden
  representation, and one final Linear layer emits logits for *every* output position
  at once: [hidden] -> [max_output_len * vocab_size], reshaped to [max_output_len, V].
  Each output position therefore gets its own set of output weights.

Limitations:
- Ignores auto-regressive structure; predicts all output positions independently.
- Serves only as a weak baseline for comparison to seq-to-seq models.

Interface:
forward(src: LongTensor[B,S], src_lens: LongTensor[B], tgt: LongTensor[B,T]) -> FloatTensor[B,T-1,V]
Matches the Transformer forward output so training scripts can be adapted easily.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    vocab_size: int
    max_digits: int  # number of digits per operand (sets the one-hot width / feature scaling)
    hidden_dim: int = 128
    num_layers: int = 2
    input_mode: Literal["onehot", "scalar"] = "onehot"
    digit_feat: bool = True  # scalar mode only: include per-digit count features for a and b
    normalize: bool = False  # scalar mode only: batch-normalize input features
    max_output_len: int = 8  # maximum (T-1) output length the head can emit


class MLPAddition(nn.Module):
    def __init__(
        self,
        config: MLPConfig,
        pad_id: int,
        sos_id: int,
        eos_id: int,
        plus_id: int,
        digit0_id: int,
    ) -> None:
        """
        plus_id / digit0_id: vocabulary ids of '+' and '0'. Digits '0'..'9' are assumed to
        occupy ids digit0_id .. digit0_id+9 (true for ``a2.vocab.default_vocab``).
        """
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.plus_id = plus_id
        self.digit0_id = digit0_id

        if config.input_mode == "onehot":
            in_dim = 2 * config.max_digits * 10
        elif config.input_mode == "scalar":
            base_feat_dim = 5  # log10(a+1), log10(b+1), log10(a+b+1), norm a, norm b
            digit_feat_dim = 20  # counts for digits 0-9 for a and b (2*10)
            in_dim = base_feat_dim + (digit_feat_dim if config.digit_feat else 0)
        else:
            raise ValueError(f"Unknown input_mode: {config.input_mode!r}")
        self.in_dim = in_dim

        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(nn.ReLU())
            dim = config.hidden_dim
        self.mlp = nn.Sequential(*layers)
        # Per-position output head: one Linear producing logits for all positions at once.
        self.out_proj = nn.Linear(dim, config.max_output_len * config.vocab_size)

    def _parse_numbers(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse integer pair (a,b) from tokenized source sequences.

        Source format: <sos> DIGITS '+' DIGITS <eos/pad>
        We decode by walking tokens, skipping specials/pad, splitting at the plus sign.
        """
        a_vals: list[int] = []
        b_vals: list[int] = []
        for seq in src:  # seq: [S]
            a = 0
            b = 0
            part_b = False
            for tok in seq.tolist():
                if tok in (self.pad_id, self.sos_id, self.eos_id):
                    continue
                if tok == self.plus_id:
                    part_b = True
                    continue
                d = tok - self.digit0_id
                if 0 <= d <= 9:
                    if part_b:
                        b = b * 10 + d
                    else:
                        a = a * 10 + d
            a_vals.append(a)
            b_vals.append(b)
        a_tensor = torch.tensor(a_vals, device=src.device, dtype=torch.long)
        b_tensor = torch.tensor(b_vals, device=src.device, dtype=torch.long)
        return a_tensor, b_tensor

    def _digits(self, x: torch.Tensor) -> torch.Tensor:
        """Least-significant-first decimal digits of x, [B, max_digits]."""
        cols = []
        tmp = x.clone()
        for _ in range(self.config.max_digits):
            cols.append(tmp % 10)
            tmp = tmp // 10
        return torch.stack(cols, dim=1)

    def _build_features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.config.input_mode == "onehot":
            digits = torch.cat([self._digits(a), self._digits(b)], dim=1)  # [B, 2*max_digits]
            onehot = torch.nn.functional.one_hot(digits, num_classes=10).float()  # [B, 2*max_digits, 10]
            return onehot.flatten(1)

        # scalar mode: hand-crafted numeric features
        a_f = a.float()
        b_f = b.float()
        s_f = (a + b).float()
        denom = 10 ** self.config.max_digits
        base = torch.stack([
            torch.log10(a_f + 1.0),
            torch.log10(b_f + 1.0),
            torch.log10(s_f + 1.0),
            a_f / denom,
            b_f / denom,
        ], dim=1)

        if self.config.normalize:
            base = (base - base.mean(dim=0, keepdim=True)) / (base.std(dim=0, keepdim=True) + 1e-8)

        if not self.config.digit_feat:
            return base

        def digit_counts(x: torch.Tensor) -> torch.Tensor:
            # returns [B,10]: how many times each digit occurs in x
            counts = torch.zeros(x.size(0), 10, device=x.device)
            ones = torch.ones(x.size(0), 1, device=x.device)
            for d in self._digits(x).unbind(dim=1):
                counts.scatter_add_(1, d.unsqueeze(1), ones)
            return counts

        digit_feats = torch.cat([digit_counts(a), digit_counts(b)], dim=1)
        if self.config.normalize:
            digit_feats = (digit_feats - digit_feats.mean(dim=0, keepdim=True)) / (digit_feats.std(dim=0, keepdim=True) + 1e-8)
        return torch.cat([base, digit_feats], dim=1)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:  # noqa: D401
        a, b = self._parse_numbers(src)
        feats = self._build_features(a, b)
        h = self.mlp(feats)  # [B, hidden]
        B = h.size(0)
        # Teacher-forcing target length: predict positions 1..T-1 (everything after SOS)
        Lt = tgt.size(1) - 1
        if Lt > self.config.max_output_len:
            raise ValueError(f"Target length {Lt} exceeds max_output_len={self.config.max_output_len}")
        logits = self.out_proj(h).view(B, self.config.max_output_len, self.config.vocab_size)
        return logits[:, :Lt]
