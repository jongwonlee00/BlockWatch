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

## Model
- Graph Convolutional Network (GCN)
- Learns relationships between wallets via transaction edges
- Captures both **local** and **global** graph structure
