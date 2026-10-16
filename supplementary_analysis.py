"""Supplementary evaluation of the saved checkpoints (no training, no change to existing results).

1. Splits the test metrics by target type (real word / punctuation / <unk>), because the overall top-1
   accuracy and perplexity in results/test_metrics.json include punctuation and <unk> targets.
2. Measures test perplexity when the model sees only the last L context words (L = 30, 12, 5), because
   generation starts from 9-13-word seeds while training always used 30-word windows.

Usage:  python supplementary_analysis.py   (after train.py; writes only results/supplementary_analysis.{json,md})
"""
import json
import math

import torch
import torch.nn.functional as F

from evaluate import md_table
from src import config as C
from src.data import is_word, load_corpus, make_windows
from src.models import load_model

CONTEXT_LENGTHS = [30, 12, 5]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus()
    X, y = (t.to(device) for t in make_windows(corpus.ids("test")))
    is_unk = y == 0
    is_punct = torch.tensor([not is_word(corpus.itos[i]) and i != 0 for i in y.tolist()], device=device)
    is_wordt = ~is_unk & ~is_punct
    types = {"word": is_wordt, "punctuation": is_punct, "unk": is_unk}
    out = {"test_windows": len(y), "share_of_targets": {k: m.float().mean().item() for k, m in types.items()},
           "models": {}}

    for n in C.MODEL_NAMES:
        model, _ = load_model(n, device)
        with torch.no_grad():
            logits = torch.cat([model(X[i:i + 1024]) for i in range(0, len(X), 1024)])
            nll = F.cross_entropy(logits, y, reduction="none")
            hit = logits.argmax(1) == y
            ctx_ppl = {L: math.exp(torch.cat([F.cross_entropy(model(X[i:i + 1024, -L:]), y[i:i + 1024], reduction="none")
                                              for i in range(0, len(X), 1024)]).mean().item())
                       for L in CONTEXT_LENGTHS}
        out["models"][n] = {
            "ppl_all_targets": math.exp(nll.mean().item()),
            "ppl_excluding_unk_targets": math.exp(nll[~is_unk].mean().item()),
            "ppl_word_targets_only": math.exp(nll[is_wordt].mean().item()),
            "top1_all_targets": hit.float().mean().item(),
            **{f"top1_{k}_targets": hit[m].float().mean().item() for k, m in types.items()},
            "test_ppl_by_context_length": {str(L): v for L, v in ctx_ppl.items()},
        }

    (C.RESULTS_DIR / "supplementary_analysis.json").write_text(json.dumps(out, indent=2))
    s = out["share_of_targets"]
    rows = [[n, f"{m['ppl_all_targets']:.1f}", f"{m['ppl_word_targets_only']:.1f}", f"{m['top1_all_targets']:.2%}",
             f"{m['top1_word_targets']:.2%}", f"{m['top1_punctuation_targets']:.2%}", f"{m['top1_unk_targets']:.2%}"]
            for n, m in out["models"].items()]
    rows_ctx = [[n] + [f"{m['test_ppl_by_context_length'][str(L)]:.2f}" for L in CONTEXT_LENGTHS]
                for n, m in out["models"].items()]
    text = (f"# Supplementary analysis (saved checkpoints, test split, {out['test_windows']:,} windows)\n\n"
            f"Test targets: real words {s['word']:.1%}, punctuation {s['punctuation']:.1%}, `<unk>` {s['unk']:.1%}.\n\n"
            "## Metrics by target type\n\n"
            + md_table(["Model", "PPL (all targets)", "PPL (real-word targets)", "Top-1 (all)", "Top-1 (real words)",
                        "Top-1 (punctuation)", "Top-1 (`<unk>`)"], rows)
            + "\n\n## Test perplexity when only the last L context words are given\n\n"
            + md_table(["Model"] + [f"L = {L}" for L in CONTEXT_LENGTHS], rows_ctx) + "\n")
    (C.RESULTS_DIR / "supplementary_analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
