# BlockWatch

## Overview
**BlockWatch** is a simple machine learning project focused on detecting crypto fraud using a **Graph Convolutional Network (GCN)**.

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
- Graph Convolutional Network (GCN)
- Learns relationships between wallets via transaction edges
- Captures both **local** and **global** graph structure

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
``` 

## Requirements

- torch
- torch-geometric
- pandas
- numpy
- scikit-learn
- networkx
- matplotlib

