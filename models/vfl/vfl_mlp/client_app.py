"""VFL Fraud Detection — Client (Bottom Model).

Each client holds a disjoint subset of features (vertical partition):
  Client 0: features 0-9   (e.g., transaction features from Bank A)
  Client 1: features 10-19 (e.g., account features from Bank B)
  Client 2: features 20-29 (e.g., merchant features from Bank C)

In simulation mode, each client trains a bottom model on its features plus
a local classifier for end-to-end gradient flow. The bottom model weights
are aggregated via FedAvg.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

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

logger = logging.getLogger("vfl_fraud.client")

FEATURES_PER_CLIENT = 10
TOTAL_FEATURES = 30


class VerticalDataset(Dataset):
    """Dataset that only exposes a subset of features."""
    def __init__(self, X, y, feature_start, feature_end):
        self.X = torch.from_numpy(X[:, feature_start:feature_end]).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class VFLBottomModel(nn.Module):
    def __init__(self, input_dim=10, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, embed_dim), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        embed = self.net(x)
        return self.classifier(embed).squeeze(-1)


_data_cache = {"key": None, "train": None, "val": None}


def _load(pid, n_clients):
    key = f"vfl_{pid}_{n_clients}"
    if _data_cache["key"] == key and _data_cache["train"]:
        return _data_cache["train"], _data_cache["val"]

    # Generate full fraud data then vertically partition
    rng = np.random.RandomState(42)
    n = int(os.environ.get("MAX_SAMPLES", 5000))
    X = rng.randn(n, TOTAL_FEATURES).astype(np.float32)
    y = (rng.rand(n) < 0.02).astype(np.float32)
    # Add signal to fraud samples
    fraud_mask = y == 1
    X[fraud_mask, :5] += 2.0

    # Normalize
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std

    # Vertical partition: each client gets FEATURES_PER_CLIENT features
    feat_start = pid * FEATURES_PER_CLIENT
    feat_end = feat_start + FEATURES_PER_CLIENT

    # Train/val split
    n_val = int(n * 0.1)
    n_train = n - n_val

    train_dl = DataLoader(
        VerticalDataset(X[:n_train], y[:n_train], feat_start, feat_end),
        batch_size=64, shuffle=True)
    val_dl = DataLoader(
        VerticalDataset(X[n_train:], y[n_train:], feat_start, feat_end),
        batch_size=64)

    _data_cache.update(key=key, train=train_dl, val=val_dl)
    logger.info(f"VFL Client {pid}: features [{feat_start}:{feat_end}], "
                f"train={n_train}, val={n_val}")
    return train_dl, val_dl


class VFLClient(fl.client.NumPyClient):
    def __init__(self, pid, n_clients):
        self.pid = pid
        self.n_clients = n_clients
        self.model = VFLBottomModel(FEATURES_PER_CLIENT)

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        sd = dict(zip(self.model.state_dict().keys(), params))
        clamped = {}
        for k, v in sd.items():
            t = torch.tensor(np.array(v))
            if not torch.isfinite(t).all():
                t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
            clamped[k] = t
        self.model.load_state_dict(clamped)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_dl, _ = _load(self.pid, self.n_clients)

        opt = optim.Adam(self.model.parameters(), lr=0.001)
        crit = nn.BCELoss()
        self.model.train()
        total_loss, nb = 0.0, 0

        epochs = int(config.get("local_epochs", 1))
        for _ in range(epochs):
            for X, y in train_dl:
                opt.zero_grad()
                loss = crit(self.model(X).clamp(1e-7, 1-1e-7), y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                nb += 1

        return self.get_parameters({}), len(train_dl.dataset), {"loss": total_loss / max(nb, 1)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        _, val_dl = _load(self.pid, self.n_clients)
        self.model.eval()
        crit = nn.BCELoss()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                pred = self.model(X).clamp(1e-7, 1-1e-7)
                loss += crit(pred, y).item()
                correct += ((pred > 0.5).float() == y).sum().item()
                total += y.size(0)
        return loss / max(len(val_dl), 1), total, {"accuracy": correct / total if total else 0}


def client_fn(context: Context) -> Client:
    pid = int(context.node_config.get("partition-id", 0))
    n = int(context.node_config.get("num-clients", 0) or context.node_config.get("num-partitions", 0) or 3)
    return VFLClient(pid, n).to_client()


app = ClientApp(client_fn=client_fn)
