"""Step 3 (interactive): generate text from your own seed with any trained model.

Examples:
  python generate.py --seed "the psammead looked at the children and" --model LSTM --temperature 0.5
  python generate.py --seed "the door opened and" --temperature 0.2 0.5 1.0 --samples 2
"""
import argparse
import json

import torch

from src import config as C
from src.data import SENT_END, detokenize, tokenize
from src.generation import generate
from src.models import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", required=True, help="seed text")
    p.add_argument("--model", nargs="+", default=C.MODEL_NAMES, choices=C.MODEL_NAMES)
    p.add_argument("--temperature", nargs="+", type=float, default=[0.5])
    p.add_argument("--length", type=int, default=C.GEN_LEN)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--rng", type=int, default=C.SEED)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    itos = json.loads((C.MODELS_DIR / "vocab.json").read_text())
    stoi = {w: i for i, w in enumerate(itos)}
    seed = tokenize(args.seed)
    unknown = [w for w in seed if w not in stoi]
    if unknown:
        print(f"note: not in vocabulary (read as <unk>): {unknown}")
    seed_ids = [stoi.get(w, 0) for w in seed]

    for name in args.model:
        model, _ = load_model(name, device)
        for T in args.temperature:
            for k, out in enumerate(generate(model, seed_ids, args.length, T, args.samples, args.rng)):
                text = detokenize([itos[i] for i in out], cap_first=bool(seed) and seed[-1] in SENT_END)
                print(f"[{name} | T={T} | sample {k + 1}] {detokenize(seed)} >> {text}\n")


if __name__ == "__main__":
    main()
