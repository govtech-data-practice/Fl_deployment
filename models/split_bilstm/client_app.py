"""Split Learning BiLSTM — Client (Bottom Model).

Each client holds:
  - Bottom model: BiLSTM encoder (STAYS LOCAL, never shared)
  - Top model: Classifier (SHARED via FedAvg — simulates server-side aggregation)

The split point is at the LSTM output (hidden state embedding).
In real split learning, the client would send activations to the server.
In this simulation, each client trains both parts end-to-end but only
the top model weights are aggregated — the LSTM stays private.

This demonstrates: raw data never leaves the client, and the feature
extractor (LSTM) is never shared with the server.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import flwr as fl
from flwr.client import Client
from flwr.clientapp import ClientApp
from flwr.common import Context

logger = logging.getLogger("split_bilstm.client")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SepsisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SplitBottomModel(nn.Module):
    """Bottom model (client-side): BiLSTM encoder. NEVER SHARED."""
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # Last timestep hidden state (embed_dim = hidden*2 = 128)


class SplitTopModel(nn.Module):
    """Top model (aggregated via FL): classifier on LSTM embedding."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


_data_cache = {"key": None, "train": None, "val": None}


def _load(pid, n_clients, input_dim=14):
    key = f"split_{pid}_{n_clients}"
    if _data_cache["key"] == key and _data_cache["train"]:
        return _data_cache["train"], _data_cache["val"]

    # Synthetic sepsis data
    n = int(os.environ.get("MAX_SAMPLES", 2000))
    rng = np.random.RandomState(42)
    X = rng.randn(n, 48, input_dim).astype(np.float32)
    y = (rng.rand(n) > 0.7).astype(np.float32)

    # Normalize
    flat = X.reshape(-1, input_dim)
    mean, std = flat.mean(0), flat.std(0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std

    # IID partition across clients
    indices = rng.permutation(n)
    per_client = n // n_clients
    start = pid * per_client
    end = n if pid == n_clients - 1 else start + per_client
    client_idx = indices[start:end]

    n_val = int(len(client_idx) * 0.1)
    train_idx, val_idx = client_idx[n_val:], client_idx[:n_val]

    train_dl = DataLoader(SepsisDataset(X[train_idx], y[train_idx]), batch_size=32, shuffle=True)
    val_dl = DataLoader(SepsisDataset(X[val_idx], y[val_idx]), batch_size=32)

    _data_cache.update(key=key, train=train_dl, val=val_dl)
    logger.info(f"Split client {pid}: train={len(train_idx)}, val={len(val_idx)}")
    return train_dl, val_dl


class SplitClient(fl.client.NumPyClient):
    def __init__(self, pid, n_clients, input_dim=14):
        self.pid = pid
        self.n_clients = n_clients
        # Bottom model stays LOCAL — never shared
        self.bottom = SplitBottomModel(input_dim).to(DEVICE)
        # Top model is SHARED via FL
        self.top = SplitTopModel(128).to(DEVICE)

    def get_parameters(self, config):
        # ONLY return top model parameters (classifier) — bottom stays private
        return [v.cpu().numpy() for v in self.top.state_dict().values()]

    def set_parameters(self, params):
        # ONLY update top model from server
        sd = dict(zip(self.top.state_dict().keys(), params))
        self.top.load_state_dict({k: torch.tensor(np.array(v), device=DEVICE) for k, v in sd.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_dl, _ = _load(self.pid, self.n_clients)

        # Train both bottom and top end-to-end
        all_params = list(self.bottom.parameters()) + list(self.top.parameters())
        opt = optim.Adam(all_params, lr=0.001)
        crit = nn.BCELoss()
        self.bottom.train()
        self.top.train()
        total_loss, nb = 0.0, 0

        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            # Forward: bottom → embedding → top → prediction
            embedding = self.bottom(X)
            pred = self.top(embedding)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1

        # Return ONLY top model weights
        return self.get_parameters({}), len(train_dl.dataset), {"loss": total_loss / max(nb, 1)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        _, val_dl = _load(self.pid, self.n_clients)
        self.bottom.eval()
        self.top.eval()
        crit = nn.BCELoss()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                embedding = self.bottom(X)
                pred = self.top(embedding)
                loss += crit(pred, y).item()
                correct += ((pred > 0.5).float() == y).sum().item()
                total += y.size(0)
        return loss / max(len(val_dl), 1), total, {"accuracy": correct / total if total else 0}


def client_fn(context: Context) -> Client:
    pid = int(context.node_config.get("partition-id", 0))
    n = int(context.node_config.get("num-clients", 0) or context.node_config.get("num-partitions", 0) or 5)
    idim = int(os.environ.get("INPUT_DIM", 14))
    return SplitClient(pid, n, idim).to_client()


app = ClientApp(client_fn=client_fn)
