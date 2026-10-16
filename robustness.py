"""Optional check: re-train every architecture with several random seeds (checkpoints are not saved)
and report the mean +/- standard deviation of the test metrics, to confirm that the model ranking
is not the result of one lucky initialisation.

Usage:  python robustness.py
"""
import json

import numpy as np
import torch

from evaluate import md_table, test_metrics
from src import config as C
from src.data import is_word, load_corpus, make_windows
from train import train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus()
    data = [tuple(t.to(device) for t in make_windows(corpus.ids(s))) for s in ("train", "val")]
    Xte, yte = (t.to(device) for t in make_windows(corpus.ids("test")))
    word_mask = torch.tensor([is_word(w) for w in corpus.itos], device=device)
    word_mask[0] = False

    runs = {n: [] for n in C.MODEL_NAMES}
    for seed in C.ROBUSTNESS_SEEDS:
        for name in C.MODEL_NAMES:
            summary, model = train_model(name, corpus, data, device, seed=seed, save=False, verbose=False)
            m = test_metrics(model, Xte, yte, word_mask)
            m.update(train_time_s=summary["train_time_s"], best_epoch=summary["best_epoch"])
            runs[name].append(m)
            print(f"seed {seed} {name:10s} test ppl {m['test_ppl']:7.2f}  word ppl {m['word_ppl']:7.2f}  "
                  f"word top1 {m['word_top1_acc']:.3f}  time {m['train_time_s']:.0f}s")

    keys = ["test_ppl", "word_ppl", "top1_acc", "word_top1_acc", "train_time_s"]
    # sample standard deviation (ddof = 1) over the extra runs; descriptive only, no significance test
    agg = {n: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))) for k in keys}
           for n, rs in runs.items()}
    (C.RESULTS_DIR / "robustness.json").write_text(json.dumps({"seeds": C.ROBUSTNESS_SEEDS, "runs": runs,
                                                               "mean_std": agg}, indent=2))
    rows = [[n, ", ".join(f"{r['test_ppl']:.1f}" for r in runs[n]),
             f"{a['test_ppl'][0]:.1f} ± {a['test_ppl'][1]:.1f}", f"{a['word_ppl'][0]:.1f} ± {a['word_ppl'][1]:.1f}",
             f"{a['top1_acc'][0]:.1%} ± {a['top1_acc'][1]:.1%}", f"{a['word_top1_acc'][0]:.1%} ± {a['word_top1_acc'][1]:.1%}",
             f"{a['train_time_s'][0]:.0f} ± {a['train_time_s'][1]:.0f}"]
            for n, a in agg.items()]
    table = md_table(["Model", "Test PPL per run", "Test PPL (all)", "Test PPL (words)", "Top-1 (all)",
                      "Top-1 (words)", "Train time (s)"], rows)
    (C.RESULTS_DIR / "robustness.md").write_text(
        f"Mean ± sample std (ddof = 1) over {len(C.ROBUSTNESS_SEEDS)} extra training runs (seeds {C.ROBUSTNESS_SEEDS}). "
        "Descriptive only: no significance test.\n\n"
        + table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
