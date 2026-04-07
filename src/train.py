import torch
from sklearn.model_selection import train_test_split

from src.config import Config
from src.data_loader import build_graph, load_transactions
from src.evaluate import evaluate
from src.model import FraudGCN


config = Config()


def main():
    df = load_transactions(config.raw_data_path)
    data = build_graph(df)

    num_nodes = data.x.size(0)
    indices = list(range(num_nodes))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    model = FraudGCN(
        in_channels=data.x.size(1),
        hidden_channels=config.hidden_dim,
        out_channels=config.output_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])

        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)
        train_acc = (preds[train_mask] == data.y[train_mask]).float().mean().item()

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

    print("\nTest Results:")
    evaluate(model, data, test_mask)


if __name__ == "__main__":
    main()
