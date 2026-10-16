Mean ± sample std (ddof = 1) over 3 extra training runs (seeds [1, 2, 3]). Descriptive only: no significance test.

| Model | Test PPL per run | Test PPL (all) | Test PPL (words) | Top-1 (all) | Top-1 (words) | Train time (s) |
|---|---|---|---|---|---|---|
| SimpleRNN | 105.5, 107.2, 105.5 | 106.1 ± 1.0 | 178.3 ± 1.7 | 19.0% ± 0.3% | 15.1% ± 0.2% | 87 ± 3 |
| BiRNN | 114.3, 113.4, 116.5 | 114.8 ± 1.6 | 196.7 ± 3.1 | 18.8% ± 0.2% | 15.0% ± 0.1% | 187 ± 4 |
| LSTM | 93.0, 92.8, 97.3 | 94.4 ± 2.5 | 155.5 ± 4.0 | 19.3% ± 0.2% | 14.7% ± 0.2% | 48 ± 0 |
| GRU | 96.8, 97.6, 96.9 | 97.1 ± 0.5 | 159.6 ± 1.5 | 18.9% ± 0.6% | 14.6% ± 0.5% | 50 ± 16 |
