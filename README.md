# Fashion Image Classifier with W&B Experiment Tracking

**Research question:** How much does transfer learning improve over a from-scratch CNN for fashion product classification, and which pretrained backbone performs best?

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) from Kaggle. ~44,000 product images with metadata. We filter to the top categories (>= 500 images each) and classify by `articleType` (Tshirts, Shirts, Casual Shoes, Watches, Tops, Handbags, Heels, etc.).

## Models

| Model | Type | Parameters | Pretrained? |
|-------|------|-----------|-------------|
| SimpleCNN | 3-block CNN | ~2M | No (from scratch) |
| ResNet-50 | Deep residual | ~25.6M | ImageNet |
| EfficientNet-B0 | Compound scaling | ~5.3M | ImageNet |

## Results

*Results will be populated after training runs complete.*

| Model | Val Accuracy | Top-3 Accuracy | Best LR | Notes |
|-------|-------------|---------------|---------|-------|
| SimpleCNN | -- | -- | -- | Baseline |
| ResNet-50 | -- | -- | -- | Transfer learning |
| EfficientNet-B0 | -- | -- | -- | Transfer learning |

## W&B Dashboard

*Link to public W&B project will be added after runs complete.*

## Project Structure

```
fashion-image-classifier/
├── .github/workflows/ci.yml          # Lint + test CI
├── .env.example                       # WANDB_API_KEY template
├── pyproject.toml                     # pytest + ruff config
├── requirements.txt
├── sweep.yaml                         # W&B Bayesian sweep config
├── data/raw/
│   └── README.md                      # Kaggle download instructions
├── src/
│   ├── config.py                      # Paths, constants, hyperparameters
│   ├── data_loader.py                 # CSV parsing, filtering, DataLoaders
│   ├── transforms.py                  # Augmentation + normalization
│   ├── models.py                      # SimpleCNN, ResNet50, EfficientNetB0
│   ├── training.py                    # Train/eval loops + W&B logging
│   ├── evaluation.py                  # Metrics (pure functions)
│   └── wandb_utils.py                # W&B helpers
├── tests/                             # ~34 CI-safe tests
├── notebooks/
│   └── experiment.ipynb               # Full experiment notebook
└── results/figures/
```

## Setup

```bash
# Clone and install
git clone https://github.com/mtsdrury/fashion-image-classifier.git
cd fashion-image-classifier
pip install -r requirements.txt

# Download dataset (requires Kaggle account)
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip -d data/raw/

# Set up W&B
cp .env.example .env
# Edit .env with your WANDB_API_KEY
wandb login

# Run tests
pytest tests/ -v

# Run a sweep
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

## Key Design Decisions

- **W&B isolated in `wandb_utils.py`**: training and evaluation are testable without a W&B account.
- **`use_wandb` flag**: training loop works with or without W&B, so local development needs no API key.
- **CPU-only PyTorch in CI**: avoids 2GB CUDA download; all tests use small random tensors.
- **Mocked pretrained weights in tests**: ResNet/EfficientNet tests mock backbones to avoid downloading weights.
- **Stratified splits**: proportional class representation in train/val/test sets.

## Tech Stack

Python, PyTorch, torchvision, scikit-learn, Weights & Biases, Pandas, Pillow, GitHub Actions
