"""The four next-word prediction models.

All share the same skeleton  Embedding -> 2-layer recurrent encoder -> Dropout -> Linear(vocab)
and differ only in the recurrent cell / direction:

  SimpleRNN : vanilla tanh RNN, left-to-right
  BiRNN     : vanilla tanh RNN run left-to-right AND right-to-left over the context window
  LSTM      : long short-term memory (input/forget/output gates + cell state)
  GRU       : gated recurrent unit (update/reset gates)

The bidirectional model only ever reads the *context window* (words t-L .. t-1) in both
directions and predicts word t, so no future information leaks into the prediction and
it remains a valid autoregressive generator.
"""
import torch
from torch import nn

from . import config as C

MODEL_SPECS = {
    "SimpleRNN": dict(cell="rnn", bidirectional=False),
    "BiRNN": dict(cell="rnn", bidirectional=True),
    "LSTM": dict(cell="lstm", bidirectional=False),
    "GRU": dict(cell="gru", bidirectional=False),
}
CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class NextWordModel(nn.Module):
    def __init__(self, vocab_size, cell, bidirectional, embed_dim=C.EMBED_DIM,
                 hidden_dim=C.HIDDEN_DIM, num_layers=C.NUM_LAYERS, dropout=C.DROPOUT):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = CELLS[cell](embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)

    def forward(self, x):                       # x: (batch, seq_len) word ids
        _, h = self.rnn(self.dropout(self.embedding(x)))
        if isinstance(h, tuple):                # LSTM returns (h_n, c_n)
            h = h[0]
        # h: (num_layers * num_directions, batch, hidden) -> final state(s) of the top layer
        feat = torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(feat))      # logits over the vocabulary for the next word


def build_model(name: str, vocab_size: int) -> NextWordModel:
    return NextWordModel(vocab_size, **MODEL_SPECS[name])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model(name: str, device) -> tuple[NextWordModel, dict]:
    ckpt = torch.load(C.MODELS_DIR / f"{name}.pt", map_location=device)
    model = build_model(name, ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
