import torch

def save_model(model, path: str = "model.pt"):
    torch.save(model.state_dict(), path)


def load_model(model, path: str = "model.pt"):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
