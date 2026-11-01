import torch

from a2.data import AdditionDataset, collate_fn
from a2.models.mlp_addition import MLPAddition, MLPConfig
from a2.vocab import default_vocab


def make_model(**cfg) -> MLPAddition:
    vocab = default_vocab()
    config = MLPConfig(vocab_size=vocab.size, max_digits=3, hidden_dim=32, num_layers=2, **cfg)
    return MLPAddition(
        config,
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
        plus_id=vocab.stoi["+"],
        digit0_id=vocab.stoi["0"],
    )


def encode_batch(texts):
    vocab = default_vocab()
    batch = [(torch.tensor(vocab.encode(t, add_sos=True, add_eos=True)), torch.tensor([0])) for t in texts]
    src, _, _, _ = collate_fn(batch, pad_id=vocab.pad_id)
    return src


def test_parse_numbers_round_trip():
    model = make_model()
    src = encode_batch(["654+114", "25+759", "9+99", "0+0"])
    a, b = model._parse_numbers(src)
    assert a.tolist() == [654, 25, 9, 0]
    assert b.tolist() == [114, 759, 99, 0]


def test_forward_shape_both_input_modes():
    vocab = default_vocab()
    src = encode_batch(["654+114", "25+759"])
    tgt = torch.tensor([vocab.encode("768", add_sos=True, add_eos=True), vocab.encode("784", add_sos=True, add_eos=True)])
    for mode in ("onehot", "scalar"):
        model = make_model(input_mode=mode)
        logits = model(src, torch.tensor([9, 8]), tgt)
        assert logits.shape == (2, tgt.size(1) - 1, vocab.size)
