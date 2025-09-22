# Elliptic Illicit Transaction Detection

End-to-end ML pipeline to detect illicit Bitcoin transactions (Elliptic dataset).
Includes clean train/eval CLIs, reproducible pipelines, and model registry.

## Data

This project uses the Elliptic Bitcoin dataset.

- Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Place the three CSVs in `data/raw/` as:
  - `elliptic_txs_features.csv`
  - `elliptic_txs_classes.csv`
  - `elliptic_txs_edgelist.csv`

> Note: raw data is not committed to the repo.

## Quickstart

```bash
conda create -n elliptic python=3.10 -y
conda activate elliptic
pip install -r requirements.txt

# Put the three CSVs into data/raw/
#  - elliptic_txs_features.csv
#  - elliptic_txs_classes.csv
#  - elliptic_txs_edgelist.csv

# Train + evaluate
python -m src.elliptic.train --config config/train.yaml

# Evaluate a saved model on test set
python -m src.elliptic.evaluate --model_path models/registry/best_pipeline.joblib --config config/train.yaml