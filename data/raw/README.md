# Dataset: Fashion Product Images (Small)

This project uses the **Fashion Product Images (Small)** dataset from Kaggle.

## Download Instructions

1. Go to: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
2. Download the dataset (requires a Kaggle account)
3. Extract the archive into this directory (`data/raw/`)

After extraction, you should have:
```
data/raw/
  images/       # ~44,000 product images
  styles.csv    # Metadata with articleType labels
```

## Alternative: Kaggle CLI

```bash
pip install kaggle
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip fashion-product-images-small.zip -d data/raw/
```

The dataset files are gitignored due to their size (~600MB).
