"""All figures. One fixed colour (and marker) per model so identity is consistent across charts."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
REFERENCE = "#8a8984"
COLORS = {"SimpleRNN": "#2a78d6", "BiRNN": "#eb6834", "LSTM": "#1baf7a", "GRU": "#eda100"}
MARKERS = {"SimpleRNN": "o", "BiRNN": "s", "LSTM": "^", "GRU": "D"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.size": 10, "legend.frameon": False, "legend.fontsize": 9, "text.color": TEXT,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _line(ax, x, y, name, **kw):
    ax.plot(x, y, color=COLORS[name], marker=MARKERS[name], markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1, label=name, **kw)


def _legend(fig, names=C.MODEL_NAMES, extra=None):
    handles = [plt.Line2D([], [], color=COLORS[n], marker=MARKERS[n], markersize=6,
                          markeredgecolor=SURFACE, label=n) for n in names]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))


def dataset_overview(stats: dict, word_counts: list[int], path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    words, freqs = zip(*stats["top_words"][:20])
    a1.barh(range(len(words))[::-1], freqs, height=0.6, color=COLORS["SimpleRNN"])
    a1.set_yticks(range(len(words))[::-1], words)
    a1.grid(axis="y", visible=False)
    a1.set_title("20 most frequent words")
    a1.set_xlabel("occurrences")
    ranks = range(1, len(word_counts) + 1)
    a2.loglog(ranks, word_counts, color=COLORS["SimpleRNN"])
    a2.set_title("Rank-frequency (Zipf) distribution")
    a2.set_xlabel("word rank (log)")
    a2.set_ylabel("frequency (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(history: dict, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("train_loss", "Training loss (cross-entropy)"), ("val_loss", "Validation loss (cross-entropy)"),
              ("val_ppl", "Validation perplexity"), ("val_acc", "Validation top-1 accuracy")]
    for ax, (key, title) in zip(axes.flat, panels):
        for name, h in history.items():
            ep = [r["epoch"] for r in h["epochs"]]
            _line(ax, ep, [r[key] for r in h["epochs"]], name)
            best = h["best_epoch"]
            ax.plot(best, h["epochs"][best - 1][key], marker=MARKERS[name], markersize=11,
                    color=COLORS[name], markeredgecolor=TEXT, markeredgewidth=1.2, zorder=5)
        ax.set_title(title)
        ax.set_xlabel("epoch")
    axes[1, 1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _legend(fig, list(history), extra=[plt.Line2D([], [], color=TEXT_2, marker="o", markersize=9, linestyle="",
                                                 markerfacecolor="none", markeredgecolor=TEXT,
                                                 label="best epoch (checkpoint kept)")])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_metrics(metrics: dict, path):
    names = list(metrics)
    panels = [("test_ppl", "Test perplexity, all targets", "{:.1f}"),
              ("word_ppl", "Test perplexity, real words", "{:.1f}"),
              ("word_top1_acc", "Top-1 accuracy, real words", "{:.1%}"),
              ("mean_epoch_time_s", "Training time per epoch (s)", "{:.1f}")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (key, title, fmt) in zip(axes, panels):
        vals = [metrics[n][key] for n in names]
        ax.bar(names, vals, width=0.55, color=[COLORS[n] for n in names])
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=TEXT)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=0)
        if "acc" in key:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def temperature_analysis(table: dict, human: dict, path):
    """table[model][temperature] -> metric dict; human -> the same metrics on real test-set text."""
    panels = [("distinct_2", "Diversity: distinct-2 (higher = more varied)"),
              ("rep_3", "Repetition: rep-3 (lower = less looping)"),
              ("judge_ppl", "Fluency proxy: trigram-judge PPL (lower = more typical)"),
              ("context_sim", "Context proxy: seed/continuation similarity"),
              ("copy_4", "Memorisation: 4-grams copied from training text")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (key, title) in zip(axes.flat, panels):
        for name, by_t in table.items():
            temps = sorted(by_t)
            _line(ax, temps, [by_t[t][key] for t in temps], name)
        if human and not math.isnan(human.get(key, float("nan"))):
            ax.axhline(human[key], color=REFERENCE, linewidth=1.2)
            ax.annotate("real Nesbit text", xy=(0, human[key]), xycoords=("axes fraction", "data"),
                        xytext=(4, 3), textcoords="offset points", ha="left", va="bottom",
                        fontsize=8, color=TEXT_2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("sampling temperature")
        ax.set_xticks(C.TEMPERATURES)
        if key in ("rep_3", "copy_4"):
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(steps=[1, 2, 5, 10]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if key == "judge_ppl":
            ax.set_yscale("log")
    axes.flat[-1].axis("off")
    _legend(fig, list(table))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
