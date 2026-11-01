import torch

from a2.vocab import default_vocab
from a2.data import AdditionDataset, collate_fn, generate_dataset
from a2.models.transformer_seq2seq import TransformerSeq2Seq, TransformerConfig

def test_transformer_forward():
    """Smoke test: Transformer forward returns logits for target steps with correct shapes.
    Ensures the simplified decoder-only Transformer aligns with training/test interface.
    """
    vocab = default_vocab()
    ds = AdditionDataset(generate_dataset(8, max_digits=2), vocab)
    src, src_lens, tgt, _ = collate_fn([ds[0], ds[1]], pad_id=vocab.pad_id)
    model = TransformerSeq2Seq(
        TransformerConfig(vocab_size=vocab.size, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, max_len=64),
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
    )
    logits = model(src, src_lens, tgt)
    assert logits.shape[0] == src.shape[0]
    assert logits.shape[-1] == vocab.size






