from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    sos: str = "<sos>"
    eos: str = "<eos>"


class CharVocab:
    """Character-level vocabulary with specials.

    Maps characters to indices and back. Includes PAD/SOS/EOS.
    """

    def __init__(self, symbols: List[str], specials: SpecialTokens | None = None) -> None:
        self.specials = specials or SpecialTokens()
        unique_symbols = []
        for s in symbols:
            if s not in unique_symbols:
                unique_symbols.append(s)
        # prepend specials
        self.itos: List[str] = [self.specials.pad, self.specials.sos, self.specials.eos] + unique_symbols
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

        self.pad_id = self.stoi[self.specials.pad]
        self.sos_id = self.stoi[self.specials.sos]
        self.eos_id = self.stoi[self.specials.eos]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = True) -> List[int]:
        ids: List[int] = []
        if add_sos:
            ids.append(self.sos_id)
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Unknown character: {ch}")
            ids.append(self.stoi[ch])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], remove_specials: bool = True) -> str:
        chars: List[str] = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            s = self.itos[i]
            if remove_specials and s in {self.specials.pad, self.specials.sos, self.specials.eos}:
                continue
            chars.append(s)
        return "".join(chars)


def default_vocab() -> CharVocab:
    symbols = list("0123456789+")
    return CharVocab(symbols)






