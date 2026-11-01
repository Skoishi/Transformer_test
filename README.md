## CSE6521 - Assignment 2: Learning Addition with a Transformer (vs. an MLP Baseline)

In this assignment, you will implement self-attention and a decoder-only transformer in PyTorch.

You will then train this model on addition with a small number of digits. Inputs look like `12+7` and targets like `19`.

For comparison, we have provided a baseline MLP that predicts all digits of the answer at once.

### Goals

1) Implement the Transformer following the provided code outline and tests.
   - Fill in the `TODO`s in `a2/models/transformer_seq2seq.py`; the module docstring walks through the architecture.
   - Embeddings, positional encodings, and the attention masks are provided; you implement multi-head self-attention, the decoder block, and the forward pass.
2) Compare the Transformer to the baselines.
   - Train models with small, reasonable hyperparameters.
   - Evaluate accuracy and discuss training dynamics and generalization.

### Environment setup

1) Use Python 3.11 (recommended). 3.10/3.12 also work.
2) Create and activate a virtual environment, then install requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run tests (quick sanity checks)

```bash
python -m pytest -q
```

The mask tests check provided code and should pass immediately. `tests/test_transformer.py` fails until you
complete the TODOs. The tests only check shapes, so passing them does not mean your implementation is
correct; the training run is the real check.

### Quick start (training)

Transformer (decoder-only, prefix-LM):

```bash
python -m a2.scripts.train_transformer --epochs 20 --max-digits 3 \
  --train-size 20000 --val-size 2000 --test-size 2000
```

With a correct implementation and the default hyperparameters (2 layers, lr 1e-3), exact-match accuracy
stays near 0 for the first ~5 epochs and then rises quickly, reaching roughly 90%+ by epoch 20.
A long plateau before a sudden jump is normal for this task, so don't stop early. If accuracy is still
near 0 after 20 epochs, check your implementation.

MLP baseline (non-seq, weak reference):

```bash
python -m a2.scripts.train_mlp --epochs 10 --max-digits 3 \
  --train-size 20000 --val-size 2000 --test-size 2000 \
  --hidden-dim 256 --num-layers 2
```

Notes on the MLP:
- Input: the two operands as one-hot digits (`--input-mode onehot`, default: 2 x
  `max-digits` x 10 binary inputs, no hand-crafted structure). `--input-mode scalar`
  instead uses a few numeric features (log/normalized magnitudes, per-digit counts;
  `--no-digit-feat`, `--normalize`). Note the scalar features include log10(a+b+1),
  which leaks the answer's magnitude.
- Output: all target positions predicted at once from one hidden vector (one output
  weight matrix per position; not autoregressive).
- Increase `--hidden-dim` / `--num-layers` to adjust capacity.

Memorization vs. generalization (Zhang et al. 2017, *Understanding deep learning
requires rethinking generalization*): a wide-enough MLP can fit the training set even
when the targets carry no information about the inputs. Try it:

```bash
# true labels
python -m a2.scripts.train_mlp --epochs 100 --hidden-dim 512 --num-layers 3 \
  --lr 1e-3 --weight-decay 0
# same model, training targets randomly permuted across examples
python -m a2.scripts.train_mlp --epochs 100 --hidden-dim 512 --num-layers 3 \
  --lr 1e-3 --weight-decay 0 --random-labels
```

Each epoch prints the running minibatch accuracy and, in parentheses, the exact-match
accuracy of the epoch's final weights on the whole training set (`full-set EM`); compare
that with validation accuracy in the two runs. `--data-seed` fixes the train/val/test
split (and the label permutation) independently of `--seed`, which controls model
initialization and batch order. `--random-labels` and `--data-seed` also work for
`train_transformer.py`.

Notes:
- Data defaults (sizes, digits, seeds), batch size, and weight decay are shared by `train_mlp.py` and
  `train_transformer.py`. The learning rate differs: the Transformer defaults to 1e-3 and the MLP to 3e-4.
- You can override any hyperparameter via CLI flags; use `--help` on either script to see all options.

### What to compare (minimum)

- Final validation and test exact-match accuracy for both models.
- Training speed/efficiency: epochs to reach X% accuracy or wall-clock time (roughly is fine if on CPU).
- Generalization to harder inputs: e.g., train on 2-digit operands and test on 3-digit ones with
  `--max-digits 2 --test-max-digits 3` (works for both scripts; validation stays at `--max-digits`).

### Suggested experiments (optional but helpful)

- Transformer: vary number of heads/layers; change feedforward dim.
- Data: vary `--train-size` and `--max-digits` to see sample efficiency.

### Deliverables

- Short report that includes:
  - Setup: key hyperparameters and any deviations from defaults.
  - Figures: Learning curves (loss/accuracy) or summary tables over epochs.
  - Final validation/test accuracy for both models.
  - A brief analysis: which learns faster, which generalizes better, and why you think so.
  - A couple of decoded failure cases per model, with a short discussion.
  
### Submission checklist (be precise and reproducible)

Include all of the following:

- [ ] Code: your implemented decoder-only Transformer (and any minor MLP tweaks you used)
- [ ] Report: setup, methodology, results (tables/curves), analysis, failure cases
- [ ] Baseline: brief paragraph describing the MLP baseline (limitations + observed performance)
- [ ] Reproducibility: `runs.md` with exact commands for BOTH (Transformer + MLP baseline best run)
- [ ] Seeds & environment: seeds, OS, Python, PyTorch, CUDA (if used)
- [ ] Failure cases: at least 2 illustrative incorrect predictions per model with commentary
- [ ] Generalization experiment: documented command & results

### Repository layout

```
a2/
  vocab.py            # character vocabulary utilities
  data.py             # synthetic data generator and Dataset
  utils.py            # training utilities (metrics, seeding)
  models/
    transformer_seq2seq.py  # decoder-only Transformer (prefix-LM)
    mlp_addition.py    # simple non-seq MLP baseline
  scripts/
    train_transformer.py
    train_mlp.py
tests/
  test_transformer.py     # forward-shape tests
  test_masks.py       # prefix+causal mask tests
  test_mlp.py         # MLP forward-shape test
```

### Tips

- Tokenization and masking matter. Review how SOS/EOS/PAD are handled.
- Keep runs short first; verify metrics move in the right direction before scaling up.
- Seeding is already implemented (`set_seed` in `a2/utils.py`; `--seed` / `--data-seed` flags) for reproducibility; try a couple of seeds for stability.
- You may use matplotlib (or any other plotting library) to create learning curves and visualizations for your report.

### Grading (indicative)

| Component | Points (approx) |
|-----------|-----------------|
| Correct decoder-only Transformer implementation (passes tests; masking/shape correctness) | 25 |
| MLP baseline usage & description (clear explanation of limitations; reported results) | 10 |
| Experimental methodology & reproducibility (runs.md, seeds, clear commands) | 10 |
| Core results: accuracy tables / curves (train/val/test for both models) | 15 |
| Generalization experiment (e.g., train on fewer digits / smaller train-size then evaluate harder) | 15 |
| Comparative analysis (which model is better and interpret failure cases) | 15 |
| Writing quality & clarity (concise, well-structured, figures labeled) | 10 |
| Total | 100 |

Extra credit (+5): Insightful ablation (e.g., heads/layers sweep, effect of removing positional embeddings, or feature ablation on the MLP) or a diagnostic analyzing specific error patterns.

### Submission

Submit as a zip file on Carmen.



