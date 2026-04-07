import pandas as pd
import torch
from torch_geometric.data import Data

def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df = df.dropna(subset=["source", "target", "amount"])
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["amount", "timestamp"])

def build_graph(df: pd.DataFrame) -> Data:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    wallets = sorted(set(df["source"]).union(set(df["target"])))
    wallet_to_idx = {wallet: i for i, wallet in enumerate(wallets)}

    edge_index = torch.tensor(
        [
            [wallet_to_idx[src] for src in df["source"]],
            [wallet_to_idx[dst] for dst in df["target"]],
        ],
        dtype=torch.long,
    )

    incoming_count = {w: 0 for w in wallets}
    outgoing_count = {w: 0 for w in wallets}

    incoming_amount = {w: 0.0 for w in wallets}
    outgoing_amount = {w: 0.0 for w in wallets}

    max_in_amount = {w: 0.0 for w in wallets}
    max_out_amount = {w: 0.0 for w in wallets}

    incoming_neighbors = {w: set() for w in wallets}
    outgoing_neighbors = {w: set() for w in wallets}

    incoming_times = {w: [] for w in wallets}
    outgoing_times = {w: [] for w in wallets}

    # Aggregate wallet stats
    for _, row in df.iterrows():
        src = row["source"]
        dst = row["target"]
        amt = float(row["amount"])

        outgoing_count[src] += 1
        incoming_count[dst] += 1

        outgoing_amount[src] += amt
        incoming_amount[dst] += amt

        max_out_amount[src] = max(max_out_amount[src], amt)
        max_in_amount[dst] = max(max_in_amount[dst], amt)

        outgoing_neighbors[src].add(dst)
        incoming_neighbors[dst].add(src)

        if "timestamp" in df.columns:
            ts = row["timestamp"].timestamp()
            outgoing_times[src].append(ts)
            incoming_times[dst].append(ts)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def avg_time_gap(times):
        if len(times) < 2:
            return 0.0
        times = sorted(times)
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        return sum(gaps) / len(gaps)

    def tx_per_time_window(times):
        # Simple "burstiness" proxy:
        # number of tx / active time span in seconds
        if len(times) < 2:
            return 0.0
        span = max(times) - min(times)
        if span == 0:
            return float(len(times))
        return len(times) / span

    features = []
    for wallet in wallets:
        in_cnt = incoming_count[wallet]
        out_cnt = outgoing_count[wallet]
        in_amt = incoming_amount[wallet]
        out_amt = outgoing_amount[wallet]

        avg_in_amt = in_amt / in_cnt if in_cnt > 0 else 0.0
        avg_out_amt = out_amt / out_cnt if out_cnt > 0 else 0.0

        uniq_in = len(incoming_neighbors[wallet])
        uniq_out = len(outgoing_neighbors[wallet])

        in_gap = avg_time_gap(incoming_times[wallet]) if "timestamp" in df.columns else 0.0
        out_gap = avg_time_gap(outgoing_times[wallet]) if "timestamp" in df.columns else 0.0

        in_burst = tx_per_time_window(incoming_times[wallet]) if "timestamp" in df.columns else 0.0
        out_burst = tx_per_time_window(outgoing_times[wallet]) if "timestamp" in df.columns else 0.0

        amount_ratio = out_amt / (in_amt + 1e-6)
        count_ratio = out_cnt / (in_cnt + 1e-6)

        features.append(
            [
                in_cnt,                  # 0 incoming tx count
                out_cnt,                 # 1 outgoing tx count
                in_amt,                  # 2 total incoming amount
                out_amt,                 # 3 total outgoing amount
                avg_in_amt,              # 4 avg incoming amount
                avg_out_amt,             # 5 avg outgoing amount
                max_in_amount[wallet],   # 6 max incoming tx amount
                max_out_amount[wallet],  # 7 max outgoing tx amount
                uniq_in,                 # 8 unique incoming counterparties
                uniq_out,                # 9 unique outgoing counterparties
                in_gap,                  # 10 avg incoming time gap
                out_gap,                 # 11 avg outgoing time gap
                in_burst,                # 12 incoming burstiness
                out_burst,               # 13 outgoing burstiness
                amount_ratio,            # 14 outgoing/incoming amount ratio
                count_ratio,             # 15 outgoing/incoming count ratio
            ]
        )

    x = torch.tensor(features, dtype=torch.float)

    labels = torch.zeros(len(wallets), dtype=torch.long)
    if "label" in df.columns:
        suspicious_wallets = set(df.loc[df["label"] == 1, "source"]).union(
            set(df.loc[df["label"] == 1, "target"])
        )
        for wallet in suspicious_wallets:
            labels[wallet_to_idx[wallet]] = 1

    return Data(x=x, edge_index=edge_index, y=labels)