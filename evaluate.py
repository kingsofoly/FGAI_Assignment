"""Step 2: quantitative test-set evaluation, text generation at several temperatures,
and automatic + qualitative analysis of the generated text.

Usage:  python evaluate.py      (after python train.py)
"""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from src import config as C
from src import plots
from src.data import SENT_END, detokenize, is_word, load_corpus, make_windows, tokenize
from src.generation import generate
from src.models import load_model
from src.text_metrics import ContextScorer, TrigramJudge, copy_rate, distinct_n, ngrams, rep_n

N_REFERENCE_GROUPS = 8   # 8 groups x SAMPLES_PER_SEED real test-set passages form the "real Nesbit text" reference


@torch.no_grad()
def test_metrics(model, X, y, word_mask):
    """Overall metrics count every target (incl. punctuation and <unk>); word_* metrics count real-word targets only."""
    nll, hit1, hit5 = [], [], []
    for i in range(0, len(X), 1024):
        xb, yb = X[i:i + 1024], y[i:i + 1024]
        logits = model(xb)
        nll.append(F.cross_entropy(logits, yb, reduction="none"))
        top = logits.topk(5, dim=1).indices
        hit1.append(top[:, 0] == yb)
        hit5.append((top == yb[:, None]).any(1))
    nll, hit1, hit5 = torch.cat(nll).double(), torch.cat(hit1).double(), torch.cat(hit5).double()
    m = word_mask[y]
    return dict(test_loss=nll.mean().item(), test_ppl=math.exp(nll.mean().item()),
                top1_acc=hit1.mean().item(), top5_acc=hit5.mean().item(),
                word_loss=nll[m].mean().item(), word_ppl=math.exp(nll[m].mean().item()),
                word_top1_acc=hit1[m].mean().item(), word_top5_acc=hit5[m].mean().item())


def text_scores(groups, judge, scorer, train4):
    """groups: list of groups, each a list of (seed_tokens, continuation_tokens) -> averaged metric dict.

    Diversity (distinct-n) is measured across the samples of a group; the other measures per sample.
    """
    per_group = []
    for pairs in groups:
        conts = [c for _, c in pairs]
        per_group.append(dict(
            distinct_1=distinct_n(conts, 1),
            distinct_2=distinct_n(conts, 2),
            rep_3=float(np.mean([rep_n(c, 3) for c in conts])),
            judge_ppl=float(np.exp(np.mean([math.log(judge.perplexity(s, c)) for s, c in pairs]))),
            context_sim=float(np.nanmean([scorer.score(s, c) for s, c in pairs])),
            copy_4=float(np.mean([copy_rate(c, train4) for c in conts])),
            mean_sentence_len=float(np.mean([len(c) / max(sum(t in SENT_END for t in c), 1) for c in conts])),
        ))
    return {k: float(np.mean([g[k] for g in per_group])) for k in per_group[0]}


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = load_corpus()
    history = json.loads((C.RESULTS_DIR / "training_history.json").read_text())
    names = [n for n in C.MODEL_NAMES if n in history]
    models = {n: load_model(n, device)[0] for n in names}

    # ------------------------------------------------------------------ 1. quantitative (test split)
    Xte, yte = (t.to(device) for t in make_windows(corpus.ids("test")))
    word_mask = torch.tensor([is_word(w) for w in corpus.itos], device=device)
    word_mask[0] = False                                   # <unk> is not counted as a correct word
    metrics = {}
    for n in names:
        h = history[n]
        metrics[n] = dict(params=h["params"], **test_metrics(models[n], Xte, yte, word_mask),
                          best_val_ppl=h["best_val_ppl"], best_epoch=h["best_epoch"], epochs_run=h["epochs_run"],
                          train_time_s=h["train_time_s"], time_to_best_s=h["time_to_best_s"],
                          mean_epoch_time_s=h["mean_epoch_time_s"])
        print(f"{n:10s} test ppl {metrics[n]['test_ppl']:7.2f}  word ppl {metrics[n]['word_ppl']:7.2f}  "
              f"top1 {metrics[n]['top1_acc']:.3f}  word-top1 {metrics[n]['word_top1_acc']:.3f}")
    (C.RESULTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    rows = [[n, f"{m['params']:,}", f"{m['test_loss']:.3f}", f"{m['test_ppl']:.1f}", f"{m['word_ppl']:.1f}",
             f"{m['top1_acc']:.1%}", f"{m['word_top1_acc']:.1%}", f"{m['top5_acc']:.1%}", f"{m['word_top5_acc']:.1%}",
             f"{m['best_epoch']}/{m['epochs_run']}", f"{m['mean_epoch_time_s']:.1f}", f"{m['train_time_s']:.0f}"]
            for n, m in metrics.items()]
    (C.RESULTS_DIR / "test_metrics.md").write_text(md_table(
        ["Model", "Params", "Test loss", "Test PPL (all)", "Test PPL (words)", "Top-1 (all)", "Top-1 (words)",
         "Top-5 (all)", "Top-5 (words)", "Best/run epochs", "s/epoch", "Train time (s)"], rows) + "\n")
    plots.test_metrics(metrics, C.PLOTS_DIR / "test_metrics.png")

    # ------------------------------------------------------------------ 2. generation at several temperatures
    print("\nbuilding text-quality judges ...")
    train_tokens = corpus.splits["train"]
    judge = TrigramJudge(train_tokens)
    scorer = ContextScorer(train_tokens)
    train4 = set(ngrams(train_tokens, 4))

    seeds = [tokenize(s) for s in C.SEED_TEXTS]
    samples = {n: {} for n in names}
    for n in names:
        for ti, T in enumerate(C.TEMPERATURES):
            samples[n][T] = []
            for si, seed in enumerate(seeds):
                # the same random stream for every model, so differences come from the model alone
                out = generate(models[n], corpus.encode(seed), C.GEN_LEN, T, C.SAMPLES_PER_SEED,
                               rng_seed=C.SEED + 100 * si + ti)
                samples[n][T].append((seed, [[corpus.itos[i] for i in o] for o in out]))

    # reference: real continuations taken from the (unseen) test split
    test_tok = corpus.splits["test"]
    span = 12 + C.GEN_LEN
    starts = np.linspace(0, len(test_tok) - span - 1, N_REFERENCE_GROUPS * C.SAMPLES_PER_SEED).astype(int)
    passages = [(test_tok[s:s + 12], test_tok[s + 12:s + span]) for s in starts]
    ref_groups = [passages[g::N_REFERENCE_GROUPS] for g in range(N_REFERENCE_GROUPS)]
    human = text_scores(ref_groups, judge, scorer, train4)
    human["context_sim_random_pairing"] = float(np.nanmean(
        [scorer.score(passages[i][0], passages[(i + 7) % len(passages)][1]) for i in range(len(passages))]))

    table = {n: {T: text_scores([[(s, c) for c in cs] for s, cs in samples[n][T]], judge, scorer, train4)
                 for T in C.TEMPERATURES} for n in names}
    (C.RESULTS_DIR / "temperature_metrics.json").write_text(json.dumps(
        {"models": {n: {str(T): v for T, v in d.items()} for n, d in table.items()}, "real_text_reference": human},
        indent=2))
    cols = ["distinct_1", "distinct_2", "rep_3", "judge_ppl", "context_sim", "copy_4", "mean_sentence_len"]
    fmt = {"distinct_1": "{:.3f}", "distinct_2": "{:.3f}", "rep_3": "{:.1%}", "judge_ppl": "{:.1f}",
           "context_sim": "{:.3f}", "copy_4": "{:.1%}", "mean_sentence_len": "{:.1f}"}
    rows = [[n, T] + [fmt[c].format(table[n][T][c]) for c in cols] for n in names for T in C.TEMPERATURES]
    rows.append(["Real Nesbit text (test split)", "-"] + [fmt[c].format(human[c]) for c in cols])
    (C.RESULTS_DIR / "temperature_metrics.md").write_text(md_table(
        ["Model", "T", "Distinct-1", "Distinct-2", "Rep-3", "Judge PPL", "Context sim", "Copy-4", "Words/sentence"],
        rows) + f"\n\nContext similarity of a seed paired with an unrelated real passage: "
                f"{human['context_sim_random_pairing']:.3f}\n")
    plots.temperature_analysis(table, human, C.PLOTS_DIR / "temperature_analysis.png")

    # ------------------------------------------------------------------ 3. generated text samples
    js = {n: {str(T): [{"seed": " ".join(s), "samples": [detokenize(c, False) for c in cs]} for s, cs in samples[n][T]]
              for T in C.TEMPERATURES} for n in names}
    (C.RESULTS_DIR / "generated_samples.json").write_text(json.dumps(js, indent=2))
    lines = ["# Generated text samples", "",
             f"{C.GEN_LEN} words generated after each seed; first of {C.SAMPLES_PER_SEED} samples shown "
             f"(all samples are in `generated_samples.json`). The same random stream is used for every model.", ""]
    for si, seed in enumerate(seeds):
        lines += [f"## Seed {si + 1}: \"{detokenize(seed)}\"", ""]
        for T in C.TEMPERATURES:
            lines += [f"### Temperature {T}", ""]
            for n in names:
                cont = samples[n][T][si][1][0]
                lines.append(f"- **{n}**: *{detokenize(seed)}* {detokenize(cont, seed[-1] in SENT_END)}")
            lines.append("")
    (C.RESULTS_DIR / "generated_samples.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + (C.RESULTS_DIR / "test_metrics.md").read_text())
    print((C.RESULTS_DIR / "temperature_metrics.md").read_text())


if __name__ == "__main__":
    main()
