import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Load saved files
print("Beginning of data load\n")
graph = torch.load(r"model/graph_data.pt", weights_only=False)
print("`graph.pt` loaded\n")

data = graph['data']
print("Data loaded\n")

tx_to_index = graph['tx_to_index']
print("Transactions to index loaded\n")

best_thresh = graph['best_thresh']
print("Best threshold loaded\n")

scaler = graph['scaler']
print("Scaler loaded\n")

# Load in model
print("Loading model...\n")
class GraphSage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)

model = GraphSage(in_channels=165, hidden_channels=256, out_channels=2, dropout=0.5)
model.load_state_dict(torch.load(r"model/best_model.pt", weights_only=True))
model.eval()
print("Model loaded\n")

# Prediction
@torch.no_grad()
def predict(tx_id: int) -> dict:
    """
    Given a txId from the Elliptic dataset, returns its illicit probability.
    
    Args:
        tx_id: integer transaction ID (must exist in the graph)
    
    Returns:
        dict with txId, p_illicit, prediction, and confidence
    """
    # Validate
    if tx_id not in tx_to_index:
        raise ValueError(f"txId {tx_id} not found in graph. Only transactions from the Elliptic dataset are supported.")
    
    model.eval()
    
    # Run full graph through model
    out   = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)          # [n_nodes, 2]
    
    # Extract this specific node
    idx       = tx_to_index[tx_id]
    p_licit   = probs[idx, 0].item()
    p_illicit = probs[idx, 1].item()
    
    prediction = "illicit" if p_illicit >= best_thresh else "licit"
    
    # Confidence = how far from the threshold
    confidence = abs(p_illicit - best_thresh) / max(best_thresh, 1 - best_thresh)
    
    return {
        "txId": tx_id,
        "p_licit": round(p_licit, 4),
        "p_illicit": round(p_illicit, 4),
        "threshold": round(float(best_thresh), 4),
        "prediction": prediction,
        "confidence": round(float(confidence), 4)
    }

# Example usage
print("Predicting 1...\n")
result = predict(94336035) # Should predict illicit
print(result)

print("\nPredicting 2...\n")
result2 = predict(94580503) # Should predict licit
print(result2)

print("\nPredicting 3...\n")
result3 = predict(94153268) # Should predict illicit
print(result3)

print("\nPredicting 4...\n")
result4 = predict(91881937) # Should predict licit
print(result4)