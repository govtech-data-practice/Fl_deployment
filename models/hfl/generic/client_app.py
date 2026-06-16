"""
Generic ClientApp — auto-adapts to any tabular dataset via config.

No code changes needed for new datasets. Configure via:
  GENERIC_DATA_PATH=/path/to/data.csv
  GENERIC_TARGET_COL=label
  GENERIC_INPUT_DIM=30       # auto-detected from data if CSV/NPZ
  GENERIC_NUM_CLASSES=2      # auto-detected from target
  GENERIC_TASK_TYPE=binary   # binary/multiclass
  GENERIC_MODEL=mlp          # mlp/logreg/autoencoder
  GENERIC_HIDDEN=64          # hidden dim
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from flwr.client import Client
from flwr.client import ClientApp
from flwr.common import Context

from fl_common import build_trainable_to_state_map, clip_and_noise
from tasks.hfl.generic.data import prepare_federated_data

logger = logging.getLogger("generic.client")


# ── Auto-configurable models ────────────────────────────────────────

class GenericMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=64, task_type="binary"):
        super().__init__()
        self.task_type = task_type
        if task_type == "binary":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, 1), nn.Sigmoid(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if self.task_type == "binary" else out


class GenericLogReg(nn.Module):
    def __init__(self, input_dim, num_classes, task_type="binary", **kwargs):
        super().__init__()
        self.task_type = task_type
        if task_type == "binary":
            self.linear = nn.Linear(input_dim, 1)
        else:
            self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        if self.task_type == "binary":
            return torch.sigmoid(out).squeeze(-1)
        return out


MODEL_REGISTRY = {
    "mlp": GenericMLP,
    "logreg": GenericLogReg,
}


# ── Data cache ──────────────────────────────────────────────────────

_data_cache = {"key": None, "loaders": None, "metadata": None}


def _load(pid, n_clients, ptype, alpha):
    key = f"generic_{ptype}_{alpha}_{pid}_{n_clients}"
    if _data_cache["key"] == key and _data_cache["loaders"]:
        loaders = _data_cache["loaders"]
        if pid not in loaders:
            pid = next(iter(loaders))
        return loaders[pid]["train"], loaders[pid]["val"], _data_cache["metadata"]

    # Use custom data module if specified
    data_module = os.environ.get("GENERIC_DATA_MODULE", "")
    if data_module:
        import importlib
        data_mod = importlib.import_module(data_module)
        load_fn = data_mod.prepare_federated_data
    else:
        load_fn = prepare_federated_data

    loaders, metadata = load_fn(
        num_clients=n_clients, partition_type=ptype, alpha=alpha,
        max_samples=int(os.environ.get("MAX_SAMPLES", "0")),
        seed=42,
    )
    _data_cache.update(key=key, loaders=loaders, metadata=metadata)
    if pid not in loaders:
        pid = next(iter(loaders))
    return loaders[pid]["train"], loaders[pid]["val"], metadata


# ── Client ──────────────────────────────────────────────────────────

class GenericClient(fl.client.NumPyClient):
    def __init__(self, pid, n_clients):
        self.pid = pid
        self.n_clients = n_clients
        self.model = None  # lazy init after first data load
        self._tmap = None
        self._model_name = os.environ.get("GENERIC_MODEL", "mlp")
        self._hidden = int(os.environ.get("GENERIC_HIDDEN", "64"))

    def _ensure_model(self, metadata):
        if self.model is not None:
            return
        input_dim = metadata["input_dim"]
        num_classes = metadata["num_classes"]
        task_type = metadata["task_type"]
        model_cls = MODEL_REGISTRY.get(self._model_name, GenericMLP)
        self.model = model_cls(input_dim, num_classes, hidden=self._hidden, task_type=task_type)
        self._tmap = build_trainable_to_state_map(self.model)
        self._task_type = task_type
        self._num_classes = num_classes
        logger.info(f"Model: {self._model_name}, input={input_dim}, classes={num_classes}, "
                     f"type={task_type}, params={sum(p.numel() for p in self.model.parameters())}")

    def get_parameters(self, config):
        if self.model is None:
            # Load data to init model
            _, _, meta = _load(self.pid, self.n_clients, "iid", 100.0)
            self._ensure_model(meta)
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        if self.model is None:
            _, _, meta = _load(self.pid, self.n_clients, "iid", 100.0)
            self._ensure_model(meta)
        sd = dict(zip(self.model.state_dict().keys(), params))
        clamped = {}
        for k, v in sd.items():
            t = torch.tensor(np.array(v))
            if not torch.isfinite(t).all():
                t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
            clamped[k] = t
        self.model.load_state_dict(clamped)

    def fit(self, parameters, config):
        strategy = config.get("strategy", "fedavg")
        lr = float(config.get("learning_rate", 0.001))
        epochs = int(config.get("local_epochs", 1))
        ptype = config.get("partition_type", "iid")
        alpha = float(config.get("dirichlet_alpha", 100.0))

        train_dl, _, meta = _load(self.pid, self.n_clients, ptype, alpha)
        self._ensure_model(meta)
        self.set_parameters(parameters)

        if self._task_type == "binary":
            crit = nn.BCELoss()
        else:
            crit = nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            for X, y in train_dl:
                opt.zero_grad()
                pred = self.model(X)
                if self._task_type == "binary":
                    pred = pred.clamp(1e-7, 1-1e-7)
                loss = crit(pred, y)
                loss.backward()
                opt.step()

        new_params = [v.cpu().numpy() for v in self.model.state_dict().values()]

        # DP noise
        eps = config.get("dp_epsilon")
        clip_norm = float(config.get("dp_clip_norm", 1.0))
        if eps is not None:
            new_params = clip_and_noise(new_params, parameters, float(eps), clip_norm)

        return new_params, len(train_dl.dataset), {}

    def evaluate(self, parameters, config):
        ptype = config.get("partition_type", "iid")
        alpha = float(config.get("dirichlet_alpha", 100.0))
        _, val_dl, meta = _load(self.pid, self.n_clients, ptype, alpha)
        self._ensure_model(meta)
        self.set_parameters(parameters)

        if self._task_type == "binary":
            crit = nn.BCELoss()
        else:
            crit = nn.CrossEntropyLoss()

        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                pred = self.model(X)
                if self._task_type == "binary":
                    pred = pred.clamp(1e-7, 1-1e-7)
                loss += crit(pred, y).item()
                if self._task_type == "binary":
                    correct += ((pred > 0.5).float() == y).sum().item()
                else:
                    correct += (pred.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return loss / max(len(val_dl), 1), total, {"accuracy": correct / total if total else 0}


def client_fn(context: Context) -> Client:
    pid = int(context.node_config.get("partition-id", 0))
    n = int(context.node_config.get("num-clients", 0) or context.node_config.get("num-partitions", 0) or 5)
    return GenericClient(pid, n).to_client()


app = ClientApp(client_fn=client_fn)
