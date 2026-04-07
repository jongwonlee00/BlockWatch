# Crypto Fraud GCN Project

## 1. Design

Input: User inputs wallet address.
Process: Model fetches past transactions of the wallet and determines legitmacy through learned features.
Output: Probability from 0 to 1 that assesses whether the wallet is legitimate or not.

## 2. Model Development Flow

- Develop a logistic regression based on transaction-level data as a baseline model. (Optional)
- Begin graph convolutional network architecture.
    - Define hidden layers
    - Decide on the activation functions (ReLU for each hidden layer but softmax for final output layer)
    - Choose loss function (probably Cross Entropy Loss)
- Split dataset (70/15/15)
- Training GCN (architecture)
    - Implement forward pass
    - Compute losses
    - Implement backpropagation
    - Evaluate validation and tune hyperparameters
- Model Evaluation across metrics like F1, precision, and accuracy