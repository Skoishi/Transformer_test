from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from a2.data import AdditionDataset, collate_fn, generate_dataset, permute_targets
from a2.models.transformer_seq2seq import TransformerConfig, TransformerSeq2Seq
from a2.utils import sequence_exact_match, set_seed
from a2.vocab import default_vocab


def train_one_epoch(model: TransformerSeq2Seq, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int | None = None) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0
    bar = tqdm(loader, desc=f"Train epoch {epoch}" if epoch is not None else "Train", unit="batch", leave=False, total=len(loader))
    for src, src_lens, tgt, _ in bar:
        src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, src_lens, tgt)
        gold = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = sequence_exact_match(pred, gold, pad_id=model.pad_id)
        total_loss += loss.item()
        total_acc += acc
        seen += 1
        bar.set_postfix({"loss": f"{total_loss/seen:.4f}", "acc": f"{total_acc/seen:.4f}"})
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model: TransformerSeq2Seq, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0
    bar = tqdm(loader, desc="Eval", unit="batch", leave=False, total=len(loader))
    for src, src_lens, tgt, _ in bar:
        src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
        logits = model(src, src_lens, tgt)
        gold = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        pred = logits.argmax(dim=-1)
        acc = sequence_exact_match(pred, gold, pad_id=model.pad_id)
        total_loss += loss.item()
        total_acc += acc
        seen += 1
        bar.set_postfix({"loss": f"{total_loss/seen:.4f}", "acc": f"{total_acc/seen:.4f}"})
    n = len(loader)
    return total_loss / n, total_acc / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)  # ensure d_model % nhead == 0
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Ignored by simplified model (no dropout implemented); kept for CLI compatibility",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-digits", type=int, default=3)
    parser.add_argument("--test-max-digits", type=int, default=None, help="Max digits for the test set (default: --max-digits); set larger to test generalization to longer numbers")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Seed for model init / batch order")
    parser.add_argument("--data-seed", type=int, default=0, help="Seed for the train/val/test data (and the label permutation)")
    parser.add_argument("--random-labels", action="store_true", help="Shuffle targets across the training set (Zhang et al. 2017)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = default_vocab()

    train_ex = generate_dataset(args.train_size, args.max_digits, seed=args.data_seed)
    val_ex = generate_dataset(args.val_size, args.max_digits, seed=args.data_seed + 1)
    test_max_digits = args.test_max_digits or args.max_digits
    test_ex = generate_dataset(args.test_size, test_max_digits, seed=args.data_seed + 2)
    if args.random_labels:
        train_ex = permute_targets(train_ex, seed=args.data_seed)
        print("Training targets randomly permuted (--random-labels); val/test are unchanged.")

    train_ds = AdditionDataset(train_ex, vocab)
    val_ds = AdditionDataset(val_ex, vocab)
    test_ds = AdditionDataset(test_ex, vocab)

    collate = lambda b: collate_fn(b, pad_id=vocab.pad_id)  # noqa: E731
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"Datasets: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"Batches: train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}")

    model = TransformerSeq2Seq(
        TransformerConfig(
            vocab_size=vocab.size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.ffn_dim,
            dropout=args.dropout,
            max_len=128,
        ),
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        # train_acc above is a running average over minibatches *during* the epoch; re-evaluate
        # the final weights on the whole training set for a clean train exact-match number.
        _, train_em = evaluate(model, train_eval_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} (full-set EM {train_em:.4f}) | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test  | loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    main()



