import torch
from sklearn.metrics import classification_report


def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)

    print(classification_report(data.y[mask].cpu(), preds[mask].cpu(), zero_division=0))