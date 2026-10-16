"""Automatic measures of the qualitative properties of generated text.

  diversity            distinct-n: unique n-grams / total n-grams over all samples of a setting
  repetition           rep-3: share of trigrams in a sample that repeat an earlier trigram of that sample
  coherence (fluency)  perplexity of the continuation under an independent interpolated trigram LM
                       fitted on the training split (lower = locally more like the training text)
  contextual           cosine similarity between the seed and the continuation in an LSA
  consistency          (PPMI + SVD) word space built from the training split
  memorisation         copy-4: share of generated 4-grams that occur verbatim in the training text
"""
import math
from collections import Counter

import numpy as np

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under until up upon very was we
were what when where which while who whom why will with would you your yours yourself yourselves said say
one shall may might must us it's i'm i've don't didn't can't won't there's that's he's mr mrs miss s
""".split())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(samples: list[list[str]], n: int) -> float:
    grams = [g for s in samples for g in ngrams(s, n)]
    return len(set(grams)) / max(len(grams), 1)


def rep_n(tokens: list[str], n: int = 3) -> float:
    grams = ngrams(tokens, n)
    return 1 - len(set(grams)) / max(len(grams), 1)


class TrigramJudge:
    """Jelinek-Mercer interpolated trigram LM used as a neutral fluency judge."""

    def __init__(self, train_tokens, lambdas=(0.6, 0.3, 0.1)):
        self.l3, self.l2, self.l1 = lambdas
        self.uni = Counter(train_tokens)
        self.bi = Counter(ngrams(train_tokens, 2))
        self.tri = Counter(ngrams(train_tokens, 3))
        self.total = len(train_tokens)
        self.V = len(self.uni) + 1

    def prob(self, u, v, w):
        p1 = (self.uni[w] + 1) / (self.total + self.V)
        p2 = self.bi[(v, w)] / self.uni[v] if self.uni[v] else 0.0
        p3 = self.tri[(u, v, w)] / self.bi[(u, v)] if self.bi[(u, v)] else 0.0
        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1

    def perplexity(self, context: list[str], continuation: list[str]) -> float:
        seq = context[-2:] + continuation
        off = len(seq) - len(continuation)
        nll = [-math.log(self.prob(seq[i - 2], seq[i - 1], seq[i])) for i in range(off, len(seq))]
        return math.exp(sum(nll) / len(nll))


def is_content(token: str) -> bool:
    return token.isalpha() and len(token) > 2 and token not in STOPWORDS


class ContextScorer:
    """LSA word space (PPMI of +/-10-word co-occurrence, truncated SVD) for seed/continuation relatedness."""

    def __init__(self, train_tokens, window=10, min_count=5, dim=100):
        counts = Counter(t for t in train_tokens if is_content(t))
        words = sorted(w for w, c in counts.items() if c >= min_count)
        self.index = {w: i for i, w in enumerate(words)}
        pos = [(i, self.index[t]) for i, t in enumerate(train_tokens) if t in self.index]
        M = np.zeros((len(words), len(words)), dtype=np.float64)
        for a, (i, wi) in enumerate(pos):
            b = a + 1
            while b < len(pos) and pos[b][0] - i <= window:
                wj = pos[b][1]
                M[wi, wj] += 1
                M[wj, wi] += 1
                b += 1
        total, row = M.sum(), M.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(M * total / (row @ row.T))
        ppmi = np.nan_to_num(np.maximum(pmi, 0), nan=0.0, posinf=0.0, neginf=0.0)
        U, S, _ = np.linalg.svd(ppmi, hermitian=True)
        vecs = U[:, :dim] * np.sqrt(S[:dim])
        self.vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    def _mean(self, tokens):
        rows = [self.index[t] for t in tokens if t in self.index]
        return self.vecs[rows].mean(0) if rows else None

    def score(self, seed: list[str], continuation: list[str]) -> float:
        a, b = self._mean(seed), self._mean(continuation)
        if a is None or b is None:
            return float("nan")
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def copy_rate(tokens: list[str], train_4grams: set, n: int = 4) -> float:
    grams = ngrams(tokens, n)
    return sum(g in train_4grams for g in grams) / max(len(grams), 1)
