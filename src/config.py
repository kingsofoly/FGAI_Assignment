"""Central configuration: paths, dataset sources and the shared hyper-parameters.

Every model is trained with exactly these settings so that the comparison between
Simple RNN, Bidirectional RNN, LSTM and GRU uses the same training protocol.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

# Dataset: E. Nesbit's "Psammead" trilogy (Project Gutenberg, public domain)
DATASET_NAME = "E. Nesbit's Psammead trilogy (Five Children and It; The Phoenix and the Carpet; The Story of the Amulet)"
BOOKS = [  # (Gutenberg id, local file, title)
    (778, "five_children_and_it.txt", "Five Children and It (1902)"),
    (836, "phoenix_and_the_carpet.txt", "The Phoenix and the Carpet (1904)"),
    (837, "story_of_the_amulet.txt", "The Story of the Amulet (1906)"),
]
BOOK_URL = "https://www.gutenberg.org/ebooks/{}"
BOOK_RAW_URL = "https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt"

# ---------------- data ----------------
SEED = 42
SEQ_LEN = 30            # maximum number of context words fed to the model to predict the next word
MIN_CONTEXT = 5         # shortest context used during training (see SHORT_CONTEXT_PROB)
SHORT_CONTEXT_PROB = 0.5  # share of training batches that use a random shorter context (MIN_CONTEXT..SEQ_LEN-1)
MIN_FREQ = 2            # words seen fewer times in the training split are mapped to <unk>
SPLIT = (0.8, 0.1, 0.1) # contiguous train / validation / test split, applied to each book separately

# ---------------- model (identical for all four architectures) ----------------
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# ---------------- training (identical for all four architectures) ----------------
BATCH_SIZE = 256
LR = 2e-3
MAX_EPOCHS = 30
PATIENCE = 3            # early stopping on validation loss
LR_FACTOR = 0.5         # ReduceLROnPlateau factor
LR_PATIENCE = 1
CLIP = 1.0              # gradient-norm clipping (essential for the vanilla RNNs)

MODEL_NAMES = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]

ROBUSTNESS_SEEDS = [1, 2, 3]  # extra independent training runs used by robustness.py

# ---------------- generation ----------------
TEMPERATURES = [0.2, 0.5, 0.7, 1.0, 1.5]
GEN_LEN = 40            # words generated after each seed
SAMPLES_PER_SEED = 10   # independent samples per (model, seed, temperature) for the text metrics
SEED_TEXTS = [
    "the children ran down to the gravel pit to look for the psammead , and",
    "the phoenix spread its golden wings and said",
    "when they spread out the magic carpet on the nursery floor ,",
    "i wish , said anthea , that",
]
