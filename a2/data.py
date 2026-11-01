from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset

from .vocab import CharVocab, default_vocab


@dataclass
class AdditionExample:
    src_text: str
    tgt_text: str


def generate_addition_pair(max_digits: int, carry: bool | None = None, rng: random.Random | None = None) -> AdditionExample:
    """Generate one addition problem like "12+7" -> "19".

    If carry is not None, constrain whether the example includes a carry.
    ``rng`` defaults to the global ``random`` module.
    """
    rng = rng or random
    d1 = rng.randint(0, 10**max_digits - 1)
    d2 = rng.randint(0, 10**max_digits - 1)
    if carry is not None:
        # regenerate until it matches the carry condition
        def has_carry(a: int, b: int) -> bool:
            return any(((a % 10) + (b % 10)) >= 10 for a, b in zip(map(int, str(a)[::-1]), map(int, str(b)[::-1])))

        while has_carry(d1, d2) != carry:
            d1 = rng.randint(0, 10**max_digits - 1)
            d2 = rng.randint(0, 10**max_digits - 1)

    src = f"{d1}+{d2}"
    tgt = f"{d1 + d2}"
    return AdditionExample(src, tgt)


def generate_dataset(n: int, max_digits: int, carry: bool | None = None, seed: int | None = None) -> List[AdditionExample]:
    """Generate n examples. With ``seed`` given, the dataset is a fixed function of
    (n, max_digits, carry, seed), independent of the global random state."""
    rng = random.Random(seed) if seed is not None else None
    return [generate_addition_pair(max_digits, carry, rng) for _ in range(n)]


def permute_targets(examples: List[AdditionExample], seed: int) -> List[AdditionExample]:
    """Randomly reassign targets across examples (fixed seed), keeping inputs unchanged.

    This is the "random labels" experiment of Zhang et al. (2017): the resulting data
    has no input/target relationship, so anything the model gets right on it is pure
    memorization.
    """
    tgts = [ex.tgt_text for ex in examples]
    random.Random(seed).shuffle(tgts)
    return [AdditionExample(ex.src_text, t) for ex, t in zip(examples, tgts)]


class AdditionDataset(Dataset):
    def __init__(
        self,
        examples: Iterable[AdditionExample],
        vocab: CharVocab | None = None,
    ) -> None:
        self.examples = list(examples)
        self.vocab = vocab or default_vocab()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        src_ids = self.vocab.encode(ex.src_text, add_sos=True, add_eos=True)
        tgt_ids = self.vocab.encode(ex.tgt_text, add_sos=True, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = torch.tensor([len(x) for x in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgt_seqs], dtype=torch.long)
    max_src = max(len(x) for x in src_seqs)
    max_tgt = max(len(x) for x in tgt_seqs)
    src_batch = torch.full((len(batch), max_src), pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_batch[i, : len(s)] = s
        tgt_batch[i, : len(t)] = t
    return src_batch, src_lens, tgt_batch, tgt_lens


