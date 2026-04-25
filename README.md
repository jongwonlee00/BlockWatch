# BlockWatch

## Overview
**BlockWatch** is a simple machine learning project focused on detecting crypto fraud using a **3-layer GraphSAGE network**, a graph neural network architecture from the broader family of Graph Convolutional Networks.

The model analyzes blockchain transaction data by representing wallets and transactions as a graph, enabling detection of suspicious behavior patterns that are difficult to identify with traditional methods.

---

BlockWatch leverages graph-based features such as:

- **Wallet-to-wallet transaction patterns**
- **Wallet activity features**
- **Connectivity behavior across transaction graphs**

---

## Goal
Classify crypto wallets into:

- `0` → Normal  
- `1` → Suspicious / Fraudulent  

---

## Fraud Detection Logic
- Structuring (smurfing)
  Many small transactions instead of one big one
  Example: $10,000 split into 50 transfers of $200
- Fan-in / Fan-out patterns
  Fan-in: many wallets → one wallet (aggregation)
  Fan-out: one wallet → many wallets (distribution)
- Circular transactions
  Money loops back to original wallet (laundering)
- Behavioral anomalies
  Wallet suddenly changes behavior:
  normally sends small → suddenly sends large
  low frequency → high burst activity

## Model
- **3-layer GraphSAGE** (`SAGEConv` × 3) graph neural network
- **Hidden dimension:** 256
- **Output dimension:** 2 (binary classification: licit vs. illicit)
- **Dropout:** 0.5 between graph convolution layers
- **Loss:** class-weighted cross-entropy (handles ~9% illicit class imbalance)
- **Threshold tuning:** decision threshold selected from the validation precision–recall curve to maximize F1, rather than using the default 0.5
- Learns relationships between wallets via transaction edges through three rounds of message passing
- Captures both **local** and **global** graph structure

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
``` 

## Dataset Setup

BlockWatch is trained and evaluated on the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
The dataset is **not** committed to this repository for size limitations; download it separately before testing the model.
Otherwise, use your own real dataset to train and evaluate the model. 

### Recommended Option `kagglehub` 

```bash
pip install kagglehub
```

```python
import kagglehub, shutil, os

path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
print("Downloaded to:", path)

# Move the three CSVs into data/raw/
os.makedirs("data/raw", exist_ok=True)
for fname in ("elliptic_txs_classes.csv",
              "elliptic_txs_edgelist.csv",
              "elliptic_txs_features.csv"):
    shutil.copy(os.path.join(path, fname), f"data/raw/{fname}")
```

### Verifying the dataset is in place

```bash
ls data/raw/elliptic_txs_*.csv
```

You should see all three files listed.

### Running the trained model

The repository ships with a pre-trained checkpoint at `model/best_model.pt` and the saved graph object at `model/graph_data.pt`. 
With the Elliptic CSVs in place you can run inference directly without retraining:

```bash
python scripts/predict.py
```

This will load the checkpoint and print illicit-probability predictions for a handful of example transaction IDs.

### Retraining from scratch

To reproduce training end-to-end (≈10–20 minutes on a modern CPU, faster on GPU):

```bash
python scripts/model_train.py
```

This will overwrite `model/best_model.pt` and `model/graph_data.pt`.

### .gitignore

The Elliptic CSVs are large enough to violate GitHub's 100 MB per-file limit, so they should never be committed. Add the following to `.gitignore`:

```
data/raw/elliptic_txs_*.csv
```

## Requirements

- torch
- torch-geometric
- pandas
- numpy
- scikit-learn
- networkx
- matplotlib

