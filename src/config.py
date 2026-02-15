"""Paths, constants, and hyperparameter defaults."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
STYLES_CSV = DATA_DIR / "styles.csv"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Dataset constants ──────────────────────────────────────────────────
IMAGE_SIZE = 224
MIN_CATEGORY_COUNT = 500

# ── Train / val / test fractions ───────────────────────────────────────
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ── ImageNet normalization ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Default hyperparameters ────────────────────────────────────────────
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_DROPOUT = 0.3
DEFAULT_NUM_WORKERS = 2

# ── W&B ────────────────────────────────────────────────────────────────
WANDB_PROJECT = "fashion-image-classifier"
WANDB_ENTITY = None  # uses default entity from wandb login

# ── Random seed ────────────────────────────────────────────────────────
SEED = 42
