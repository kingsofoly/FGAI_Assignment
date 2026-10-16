"""Loading, cleaning, tokenising and windowing the E. Nesbit corpus (three Project Gutenberg books)."""
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import torch

from . import config as C

UNK = "<unk>"
PUNCT = set(".,!?;:")
SENT_END = set(".!?")
# words (with internal apostrophes such as "don't" / "anthea's"), numbers, and sentence punctuation
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|\d+|[.,!?;:]")
# "CHAPTER 1" / "CHAPTER 1. THE EGG" / "CHAPTER I." at the start of a line, plus an all-capitals title line after it
CHAPTER_RE = re.compile(r"^CHAPTER\b[^\n]*\n(?:[ \t]*\n)*(?:[^a-z\n]+\n)?", re.M)
FIRST_CHAPTER_RE = re.compile(r"^CHAPTER (?:1|I)\b", re.M)


def load_book(path) -> str:
    """Return one book's story text without the Gutenberg licence, title page, dedication and contents."""
    text = path.read_text(encoding="utf-8")
    start = text.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = text.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = text[text.index("\n", start):end]
    body = body[FIRST_CHAPTER_RE.search(body).start():]          # first real chapter heading (column 0)
    body = re.sub(r"^\*Project Gutenberg Etext.*$|^\*\*\* x-header \*\*\*$", "", body, flags=re.M)
    return body


def load_books() -> dict[str, str]:
    return {title: load_book(C.DATA_DIR / fname) for _, fname, title in C.BOOKS}


def clean_text(raw: str) -> str:
    text = CHAPTER_RE.sub(" ", raw)
    text = text.replace("[Illustration]", " ")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[—–-]", " ", text)                     # dashes separate words / clauses
    for src, dst in (("æ", "ae"), ("œ", "oe")):
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()  # é -> e
    text = text.replace("_", "")                            # _italic_ markers
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def is_word(token: str) -> bool:
    return token[0].isalnum()


def detokenize(tokens: list[str], cap_first: bool = True) -> str:
    """Join tokens into readable text: attach punctuation, capitalise sentence starts and 'I'."""
    out, cap = [], cap_first
    for t in tokens:
        if t in PUNCT:
            if out:
                out[-1] += t
            else:
                out.append(t)
            cap = t in SENT_END
            continue
        w = "I" + t[1:] if t == "i" or t.startswith("i'") else t
        out.append(w[0].upper() + w[1:] if cap else w)
        cap = False
    return " ".join(out)


@dataclass
class Corpus:
    book_tokens: dict[str, list[str]]  # cleaned token stream of each book
    splits: dict[str, list[str]]       # train / val / test tokens (each book split 80/10/10, then joined)
    itos: list[str]                    # id -> token (index 0 is <unk>)
    stoi: dict[str, int]

    @property
    def tokens(self) -> list[str]:
        return [t for toks in self.book_tokens.values() for t in toks]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, 0) for t in tokens]

    def ids(self, split: str) -> torch.Tensor:
        return torch.tensor(self.encode(self.splits[split]), dtype=torch.long)


def build_vocab(tokens: list[str], min_freq: int) -> list[str]:
    counts = Counter(tokens)
    kept = [w for w, c in counts.items() if c >= min_freq]
    return [UNK] + sorted(kept, key=lambda w: (-counts[w], w))


def load_corpus() -> Corpus:
    book_tokens = {title: tokenize(clean_text(body)) for title, body in load_books().items()}
    splits = {"train": [], "val": [], "test": []}
    for toks in book_tokens.values():
        # contiguous split inside every book, so each split contains all three books
        n = len(toks)
        n_train, n_val = int(C.SPLIT[0] * n), int(C.SPLIT[1] * n)
        splits["train"] += toks[:n_train]
        splits["val"] += toks[n_train:n_train + n_val]
        splits["test"] += toks[n_train + n_val:]
    # the vocabulary is built from the training split only, so val/test contain genuine unseen words
    itos = build_vocab(splits["train"], C.MIN_FREQ)
    return Corpus(book_tokens, splits, itos, {w: i for i, w in enumerate(itos)})


def make_windows(ids: torch.Tensor, seq_len: int = C.SEQ_LEN):
    """Sliding window (stride 1): `seq_len` context words -> the next word."""
    w = ids.unfold(0, seq_len + 1, 1)
    return w[:, :-1].contiguous(), w[:, -1].contiguous()


def describe(corpus: Corpus, books: dict[str, str]) -> dict:
    """Dataset characteristics reported in the README."""
    toks = corpus.tokens
    words = [t for t in toks if is_word(t)]
    counts = Counter(words)
    n_sent = sum(t in SENT_END for t in toks)
    train_counts = Counter(corpus.splits["train"])
    return {
        "name": C.DATASET_NAME,
        "books": [{"title": title, "gutenberg_id": gid, "url": C.BOOK_URL.format(gid),
                   "characters_of_story_text": len(books[title]), "tokens": len(corpus.book_tokens[title])}
                  for gid, _, title in C.BOOKS],
        "total_tokens": len(toks),
        "word_tokens": len(words),
        "punctuation_tokens": len(toks) - len(words),
        "unique_words": len(counts),
        "hapax_legomena": sum(c == 1 for c in counts.values()),
        "sentences": n_sent,
        "mean_sentence_length_tokens": round(len(toks) / max(n_sent, 1), 2),
        "type_token_ratio": round(len(counts) / len(words), 4),
        "vocab_size_incl_unk": corpus.vocab_size,
        "min_freq": C.MIN_FREQ,
        "train_vocab_coverage": round(sum(c for w, c in train_counts.items() if w in corpus.stoi) / len(corpus.splits["train"]), 4),
        "split_tokens": {k: len(v) for k, v in corpus.splits.items()},
        "unk_rate": {k: round(sum(t not in corpus.stoi for t in v) / len(v), 4) for k, v in corpus.splits.items()},
        "windows": {k: max(len(v) - C.SEQ_LEN, 0) for k, v in corpus.splits.items()},
        "seq_len": C.SEQ_LEN,
        "top_words": counts.most_common(25),
    }
