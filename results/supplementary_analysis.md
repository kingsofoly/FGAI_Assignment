# Supplementary analysis (saved checkpoints, test split, 21,331 windows)

Test targets: real words 81.2%, punctuation 14.0%, `<unk>` 4.9%.

## Metrics by target type

| Model | PPL (all targets) | PPL (real-word targets) | Top-1 (all) | Top-1 (real words) | Top-1 (punctuation) | Top-1 (`<unk>`) |
|---|---|---|---|---|---|---|
| SimpleRNN | 104.6 | 174.0 | 18.95% | 15.29% | 40.51% | 18.05% |
| BiRNN | 109.5 | 185.9 | 19.07% | 15.08% | 41.95% | 19.88% |
| LSTM | 91.0 | 148.7 | 19.81% | 15.04% | 44.00% | 29.83% |
| GRU | 96.4 | 158.0 | 19.38% | 14.93% | 42.76% | 26.54% |

## Test perplexity when only the last L context words are given

| Model | L = 30 | L = 12 | L = 5 |
|---|---|---|---|
| SimpleRNN | 104.60 | 104.78 | 107.25 |
| BiRNN | 109.52 | 109.65 | 112.90 |
| LSTM | 91.00 | 91.42 | 93.96 |
| GRU | 96.39 | 94.31 | 98.17 |
