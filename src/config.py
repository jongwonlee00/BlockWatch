from dataclasses import dataclass

@dataclass
class Config:
    raw_data_path: str = "data/raw/transactions.csv"
    hidden_dim: int = 16
    output_dim: int = 2
    learning_rate: float = 1e-2
    epochs: int = 30
    test_size: float = 0.4
    random_state: int = 42
