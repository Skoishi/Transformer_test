"""
Assignment: Implement a Decoder-Only Transformer with primitives (no shortcuts)

What you will build
- A decoder-only Transformer that conditions on the source as a prefix and predicts target tokens.
- Interface MUST match what the training script and tests expect:
    forward(src: LongTensor[B,S], src_lens: LongTensor[B], tgt: LongTensor[B,T]) -> FloatTensor[B,T-1,V]
- Return logits only for target positions (aligned with tgt[:, 1:]).

1) BANNED APIs
- nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder
- nn.MultiheadAttention
- torch.nn.functional.scaled_dot_product_attention (and any attention one-liners)
- nn.Transformer.generate_square_subsequent_mask

2) ALLOWED building blocks (use these primitives)
- Layers: nn.Embedding, nn.Linear, nn.LayerNorm, nn.ReLU
- Tensors/ops: torch.matmul, torch.bmm, torch.einsum, torch.tril/triu, torch.where, tensor.view/reshape/transpose
- Functions: torch.nn.functional.softmax (or manual softmax via exp/sum), F.gelu/F.relu
- Misc: register_buffer, torch.no_grad, math.sqrt
- Provided: positional_encoding below (sinusoidal)

3) What to implement (step-by-step)
- Multi-head self-attention (MHA) from scratch:
    • Q,K,V = Linear(d_model->d_model)(x) each
    • Reshape: (B,L,d) -> (B,H,L,Hd) with H=nhead, Hd=d//H; use view + transpose
    • Scores = (Q @ K^T)/sqrt(Hd) with shapes (B,H,L,L)
    • Add masks: attn_mask [L,L] (0 allowed, -inf blocked), key_padding_mask [B,L] (True=PAD) applied on key axis
    • attn = softmax(scores, dim=-1); context = attn @ V -> (B,H,L,Hd)
    • Merge heads back to (B,L,d) and project with Linear(d->d)
- Decoder block (Pre-LN style):
    • y = x + SelfAttn(LayerNorm(x))
    • z = y + Linear(ReLU(Linear(LayerNorm(y))))
- Stack N identical decoder blocks
- Embeddings: a single shared nn.Embedding for src and tgt + sinusoidal positions
- Forward path and shapes:
    • Inputs: src[B,S], tgt[B,T]
    • tgt_in = tgt[:, :-1]  (teacher forcing)
    • X = [embed(src) + pos[:,:S], embed(tgt_in) + pos[:,S:S+T-1]] concatenated on time dim -> [B,L,d]
    • Build masks (see below), run X through decoder stack
    • Take Y_tgt = output[:, S:, :] and project -> logits[B, T-1, V]

Masking (prefix-LM style) -- provided for you in build_prefix_causal_mask; understand it, don't rewrite it
- Let Ls=S, Lt=T-1, total L=Ls+Lt. The attention mask M[L,L] holds 0 where attention is allowed and -inf where it is disallowed:
  • src->src block (top-left Ls x Ls): fully visible (no mask, zeros)
  • src->tgt block (top-right Ls x Lt): disallow (fill with -inf)
  • tgt->src block (bottom-left Lt x Ls): fully visible (zeros)
  • tgt->tgt block (bottom-right Lt x Lt): causal (disallow j>i)
- Additionally apply key_padding_mask over concatenated sequence so attention to PAD positions is masked out for all queries.

Correctness/Interface requirements
- Positional encodings are computed in forward for the full length L = S + T - 1.
- Respect padding_idx everywhere (no attention to PAD; loss ignores PAD outside this module).
- Do NOT compute logits for src positions; only for target positions (T-1 steps) aligned with gold tgt[:, 1:].
- Shapes must exactly match tests/train script expectations.

Tip: Start by passing tests/test_transformer.py (python -m pytest -q); then train with
python -m a2.scripts.train_transformer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional encoding: [1, length, d_model]."""
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


@dataclass
class TransformerConfig:
    """Configuration for the simplified decoder-only Transformer.

    Note: 'dropout' and 'max_len' are kept for CLI/backward compatibility but are not used
    by the simplified implementation (no dropout; positions computed per forward).
    """
    vocab_size: int
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.0
    max_len: int = 128


class MultiHeadSelfAttention(nn.Module):
    """Implement scaled-dot self-attention with multiple heads using only Linear/softmax/matmul.

    Inputs:
      x: FloatTensor[B, L, d_model]
    attn_mask: FloatTensor[L, L] with 0 for allowed, -inf for blocked
      key_padding_mask: BoolTensor[B, L] with True where positions are PAD (to be ignored)
    Output:
      y: FloatTensor[B, L, d_model]
    """

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** 0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Hd)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: implement multi-head attention.
        # Steps:
        # 1) Project x to q,k,v then reshape to (B, nhead, L, head_dim)
        # 2) Compute scores = (q @ k^T) / sqrt(head_dim)
        # 3) Add attn_mask and key_padding_mask (broadcast to heads)
        # 4) probs = softmax(scores, dim=-1)
        # 5) context = probs @ v -> reshape to (B, L, d_model)
        # 6) return out_proj(context)

        B, L, _ = x.shape

        # 1) TODO: Project to Q, K, V and reshape/split heads
        # (B, L, d_model) -> (B, nhead, L, head_dim)
        Q_x=self.q_proj(x)
        K_x=self.k_proj(x)
        V_x=self.v_proj(x)
        Q_x=self._split_heads(Q_x)
        K_x=self._split_heads(K_x)
        V_x=self._split_heads(V_x)
        # 2) TODO: Scaled dot-product attention scores: (B, nhead, L, L)
        scores=torch.matmul(Q_x, K_x.transpose(-2, -1)) / self.scale
        # 3) Add masks (expect [L, L]) (done for you)
        if attn_mask is not None:
            scores = scores + attn_mask.view(1, 1, L, L)
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] with True at PAD (to be ignored as keys)
            # convert to additive mask over key dimension
            kp = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            scores = scores.masked_fill(kp, float("-inf"))

        # Safeguard: if an entire row is -inf (shouldn't normally happen), replace with zeros before softmax
        if torch.isinf(scores).all(dim=-1).any():  # shape (B,H,L)
            # Set problematic rows to zeros so softmax yields uniform distribution
            mask_all_inf = torch.isinf(scores).all(dim=-1, keepdim=True)  # (B,H,L,1)
            scores = torch.where(mask_all_inf, torch.zeros_like(scores), scores)

        # 4) TODO: Softmax over keys
        probs=F.softmax(scores, dim=-1)
        # 5) TODO: Context and merge heads
        context=torch.matmul(probs, V_x)
        context=self._merge_heads(context)
        # 6) TODO: Output projection
        out=self.out_proj(context)

        return out


class DecoderBlock(nn.Module):
    """Pre-LN decoder block: x -> SA -> MLP with residuals."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadSelfAttention(d_model, nhead)

        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # TODO: implement Pre-LN residual block.
        # y = x + SelfAttn(LayerNorm(x))
        # z = y + Linear(ReLU(Linear(LayerNorm(y))))

        # TODO: Self-attention with Pre-LN and residual
        y = x + self.sa(self.ln1(x), attn_mask, key_padding_mask)
        # TODO: MLP with Pre-LN and residual
        z = y + self.fc2(self.act(self.fc1(self.ln2(y))))
        return z

class TransformerSeq2Seq(nn.Module):
    """Decoder-only Transformer (prefix LM) to match the training script interface.

    forward(src, src_lens, tgt) returns logits for tgt[:, 1:] only.
    """

    def __init__(self, config: TransformerConfig, pad_id: int, sos_id: int, eos_id: int) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        # Single shared embedding for both src and tgt
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=pad_id)

        self.layers = nn.ModuleList(
            [DecoderBlock(config.d_model, config.nhead, config.dim_feedforward) for _ in range(config.num_layers)]
        )
        # Weight-tied output projection (no bias), tied to token embedding
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.tok_embed.weight

    @staticmethod
    def build_prefix_causal_mask(Ls: int, Lt: int, device: torch.device) -> torch.Tensor:
        """Return FloatTensor[L, L] with 0 for allowed, -inf for disallowed.

        Layout: [src(0:Ls) | tgt(0:Lt)]
        - src->src: allow all (0)
        - src->tgt: disallow (top-right = -inf)
        - tgt->src: allow all (0)
        - tgt->tgt: causal (strictly lower triangle allowed)
        """
        L = Ls + Lt
        m = torch.zeros(L, L, device=device)
        if Lt > 0:
            # Bottom-right causal mask for target: set strictly upper triangle to -inf
            # (Avoid 0 * -inf which yields NaNs by using boolean indexing instead of multiplication.)
            causal_mask = torch.triu(torch.ones(Lt, Lt, device=device, dtype=torch.bool), diagonal=1)
            block = m[Ls:, Ls:]
            block[causal_mask] = float("-inf")
            m[Ls:, Ls:] = block
            # Top-right: src cannot see target (entire block disallowed)
            if Ls > 0:
                m[:Ls, Ls:] = float("-inf")
        return m

    @staticmethod
    def combine_key_padding_masks(src_pad: torch.Tensor, tgt_pad: torch.Tensor) -> torch.Tensor:
        """Concatenate key padding masks for src[B,S] and tgt_in[B,T-1] -> [B, S+T-1]."""
        if tgt_pad.numel() == 0:
            return src_pad
        return torch.cat([src_pad, tgt_pad], dim=1)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # TODO: implement the full forward as described in the header.
        # Outline:
        # 1) Move/slice positional encoding to device/length L = S + (T-1)
        # 2) src_emb = tok_embed(src) + pos[:, :S]
        #    tgt_in = tgt[:, :-1]; tgt_emb = tok_embed(tgt_in) + pos[:, S:S+T-1]
        # 3) X = torch.cat([src_emb, tgt_emb], dim=1)  # [B, L, d]
        # 4) Build masks:
        #    - key_padding_mask: src.eq(pad_id) and tgt_in.eq(pad_id) -> concat [B, L]
        #    - attn_mask: build_prefix_causal_mask(S, T-1)
        # 5) Run through each DecoderBlock with masks
        # 6) Take Y_tgt = output[:, S:, :] and project to logits via output_proj
        # 7) Return logits (B, T-1, V)

        device = src.device
        B, S = src.shape
        _, T = tgt.shape
        Lt = max(T - 1, 0)
        L = S + Lt
        
        # 1-2) Embed with positions computed on the fly; target uses teacher-forced input (tgt[:, :-1]) (done for you)
        pos = positional_encoding(L, self.config.d_model, device)
        src_pos = pos[:, :S, :]
        tgt_pos = pos[:, S:S + Lt, :]

        src_emb = self.tok_embed(src) + src_pos
        tgt_in = tgt[:, :-1]
        tgt_emb = self.tok_embed(tgt_in) + tgt_pos

        # 3) Concatenate prefix (src) and target-in embeddings (done for you)
        X = torch.cat([src_emb, tgt_emb], dim=1)  # [B, L, d_model]

        # 4) Build masks (done for you)
        src_pad = src.eq(self.pad_id)  # [B, S]
        tgt_pad = tgt_in.eq(self.pad_id) if Lt > 0 else torch.zeros(B, 0, dtype=torch.bool, device=device)
        key_padding_mask = self.combine_key_padding_masks(src_pad, tgt_pad)  # [B, L]
        attn_mask = self.build_prefix_causal_mask(S, Lt, device)  # [L, L]

        # 5) TODO: Run through decoder blocks

        # 6) TODO: Slice target states and project to logits

        # 7) TODO: Return logits only for target positions: [B, T-1, V]
        return logits
