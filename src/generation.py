"""Autoregressive temperature sampling shared by evaluate.py and generate.py."""
import torch

from . import config as C


@torch.no_grad()
def generate(model, seed_ids: list[int], n_words: int, temperature: float, n_samples: int = 1,
             rng_seed: int = C.SEED) -> list[list[int]]:
    """Feed the last SEQ_LEN words, sample the next word from softmax(logits / T), append it, repeat.

    Returns only the generated continuation ids for each of the `n_samples` parallel samples.
    """
    device = next(model.parameters()).device
    model.eval()
    ctx = torch.tensor([seed_ids] * n_samples, dtype=torch.long, device=device)
    gen = torch.Generator().manual_seed(rng_seed)
    for _ in range(n_words):
        logits = model(ctx[:, -C.SEQ_LEN:]).float()
        logits[:, 0] = float("-inf")                       # never emit <unk>
        probs = torch.softmax(logits / temperature, dim=-1).cpu()
        nxt = torch.multinomial(probs, 1, generator=gen).to(device)
        ctx = torch.cat([ctx, nxt], dim=1)
    return ctx[:, len(seed_ids):].tolist()
