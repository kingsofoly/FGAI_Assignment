"""Step 1: prepare the dataset and train SimpleRNN, BiRNN, LSTM and GRU under identical settings.

Usage:  python train.py                 # all four models
        python train.py --models LSTM   # a subset
"""
import argparse
import json
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src import config as C
from src import plots
from src.data import describe, is_word, load_books, load_corpus, make_windows
from src.models import build_model, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_epoch(model, X, y, optimizer=None, generator=None):
    """One pass over (X, y). Trains when an optimizer is given, otherwise evaluates.

    During training, a share (SHORT_CONTEXT_PROB) of the batches keeps only the last L words of each window,
    with L drawn uniformly from MIN_CONTEXT..SEQ_LEN-1, so that the models also learn to predict from the short
    contexts they see at the start of generation (seeds are 7-15 words long). Evaluation always uses 30 words.
    """
    training = optimizer is not None
    model.train(training)
    n = len(X)
    order = torch.randperm(n, generator=generator).to(X.device) if training else torch.arange(n, device=X.device)
    loss_sum = torch.zeros((), device=X.device)
    correct = torch.zeros((), device=X.device)
    with torch.set_grad_enabled(training):
        for i in range(0, n, C.BATCH_SIZE):
            idx = order[i:i + C.BATCH_SIZE]
            xb, yb = X[idx], y[idx]
            if training and torch.rand(1, generator=generator).item() < C.SHORT_CONTEXT_PROB:
                L = int(torch.randint(C.MIN_CONTEXT, C.SEQ_LEN, (1,), generator=generator))
                xb = xb[:, -L:]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), C.CLIP)
                optimizer.step()
            loss_sum += loss.detach() * len(idx)
            correct += (logits.argmax(1) == yb).sum()
    return loss_sum.item() / n, correct.item() / n


def train_model(name, corpus, data, device, seed=C.SEED, save=True, verbose=True):
    """Train one architecture; returns (summary dict, model holding the best-validation weights)."""
    (Xtr, ytr), (Xva, yva) = data
    set_seed(seed)
    model = build_model(name, corpus.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=C.LR_FACTOR, patience=C.LR_PATIENCE)
    gen = torch.Generator().manual_seed(seed)
    n_params = count_params(model)
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n=== {name}  ({n_params:,} parameters) ===")

    records, best_loss, best_state, best_epoch, bad, train_time = [], float("inf"), None, 0, 0, 0.0
    for epoch in range(1, C.MAX_EPOCHS + 1):
        sync(device)
        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, Xtr, ytr, optimizer, gen)
        sync(device)
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time
        va_loss, va_acc = run_epoch(model, Xva, yva)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_loss)
        records.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_ppl=float(np.exp(tr_loss)),
                            val_loss=va_loss, val_acc=va_acc, val_ppl=float(np.exp(va_loss)),
                            lr=lr, epoch_time_s=epoch_time))
        log(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} "
            f"ppl {np.exp(va_loss):7.1f} acc {va_acc:.3f} | lr {lr:.0e} | {epoch_time:.1f}s")
        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad, time_to_best = va_loss, epoch, 0, train_time
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                log(f"early stopping: no val improvement for {C.PATIENCE} epochs (best epoch {best_epoch})")
                break

    model.load_state_dict(best_state)
    if save:
        torch.save(dict(name=name, state_dict=best_state, vocab_size=corpus.vocab_size, best_epoch=best_epoch,
                        best_val_loss=best_loss, seq_len=C.SEQ_LEN), C.MODELS_DIR / f"{name}.pt")
    summary = dict(params=n_params, epochs=records, best_epoch=best_epoch, best_val_loss=best_loss,
                   best_val_ppl=float(np.exp(best_loss)), epochs_run=len(records), train_time_s=train_time,
                   time_to_best_s=time_to_best, mean_epoch_time_s=train_time / len(records))
    return summary, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=C.MODEL_NAMES, choices=C.MODEL_NAMES)
    args = parser.parse_args()
    for d in (C.MODELS_DIR, C.RESULTS_DIR, C.PLOTS_DIR):
        d.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---- dataset preparation & characteristics ----
    books = load_books()
    corpus = load_corpus()
    stats = describe(corpus, books)
    (C.RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    (C.MODELS_DIR / "vocab.json").write_text(json.dumps(corpus.itos))
    word_counts = sorted(Counter(t for t in corpus.tokens if is_word(t)).values(), reverse=True)
    plots.dataset_overview(stats, word_counts, C.PLOTS_DIR / "dataset_overview.png")
    print(f"tokens {stats['total_tokens']:,} | vocab {corpus.vocab_size:,} | windows {stats['windows']}")

    # ---- input -> target sequences (sliding window of SEQ_LEN words -> next word) ----
    data = [tuple(t.to(device) for t in make_windows(corpus.ids(s))) for s in ("train", "val")]
    X, y = data[0]
    print("example window:", " ".join(corpus.itos[i] for i in X[100].tolist()), "->", corpus.itos[y[100].item()])

    hist_path = C.RESULTS_DIR / "training_history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    for name in args.models:
        history[name], _ = train_model(name, corpus, data, device)
    history = {n: history[n] for n in C.MODEL_NAMES if n in history}
    hist_path.write_text(json.dumps(history, indent=2))
    plots.training_curves(history, C.PLOTS_DIR / "training_curves.png")

    print("\nsummary")
    for n, h in history.items():
        print(f"{n:10s} params {h['params']:>10,}  best epoch {h['best_epoch']:2d}/{h['epochs_run']:2d}  "
              f"val ppl {h['best_val_ppl']:7.2f}  train time {h['train_time_s']:6.1f}s")


if __name__ == "__main__":
    main()
