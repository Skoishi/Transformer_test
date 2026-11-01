import torch

from a2.models.transformer_seq2seq import TransformerSeq2Seq, TransformerConfig


def mask_to_bool(m: torch.Tensor) -> torch.Tensor:
    # True indicates blocked/forbidden (i.e., -inf), False allowed (0)
    return torch.isneginf(m)


def test_prefix_mask_shapes_and_blocks():
    """Sanity check: build mask across small (Ls, Lt) combos and verify overall shape L x L."""
    device = torch.device("cpu")
    # Basic small cases
    for Ls, Lt in [(0, 1), (1, 0), (2, 2), (3, 1)]:
        m = TransformerSeq2Seq.build_prefix_causal_mask(Ls, Lt, device)
        assert m.shape == (Ls + Lt, Ls + Lt)


def test_prefix_mask_semantics_src_tgt_blocks():
    """Check block structure except target self-causality:
    - src->src allowed; src->tgt blocked; tgt->src allowed.
    """
    device = torch.device("cpu")
    Ls, Lt = 2, 3
    m = TransformerSeq2Seq.build_prefix_causal_mask(Ls, Lt, device)
    blocked = mask_to_bool(m)
    # src->src allowed
    assert not blocked[0:Ls, 0:Ls].any()
    # src->tgt blocked
    assert blocked[0:Ls, Ls:Ls+Lt].all()
    # tgt->src allowed
    assert not blocked[Ls:Ls+Lt, 0:Ls].any()


def test_prefix_mask_semantics_causal_on_tgt():
    """Within the target block (tgt->tgt), enforce strict causal masking (no looking ahead)."""
    device = torch.device("cpu")
    Ls, Lt = 1, 4
    m = TransformerSeq2Seq.build_prefix_causal_mask(Ls, Lt, device)
    blocked = mask_to_bool(m)
    # In tgt->tgt block, strictly upper triangle must be blocked (j > i)
    br = blocked[Ls:, Ls:]
    assert torch.all(br == torch.triu(torch.ones(Lt, Lt, dtype=torch.bool), diagonal=1))
