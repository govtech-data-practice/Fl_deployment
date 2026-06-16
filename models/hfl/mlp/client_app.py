"""MLP ClientApp — for tabular data (fraud, structured clinical)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import logging
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from flwr.client import Client
from flwr.clientapp import ClientApp
from flwr.common import Context

from fl_common import build_trainable_to_state_map, secagg_mask_parameters, clip_and_noise
from tasks.hfl.fraud.data import prepare_federated_data

logger = logging.getLogger("fraud.client")


class FraudMLP(nn.Module):
    def __init__(self, input_dim=30, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


_data_cache = {"key": None, "loaders": None}


def _load(pid, n_clients, ptype, alpha):
    key = f"{ptype}_{alpha}_{pid}_{n_clients}"
    if _data_cache["key"] == key and _data_cache["loaders"]:
        effective_pid = pid if pid in _data_cache["loaders"] else next(iter(_data_cache["loaders"]))
        return _data_cache["loaders"][effective_pid]["train"], _data_cache["loaders"][effective_pid]["val"]

    loaders, _ = prepare_federated_data(
        num_clients=n_clients, partition_type=ptype, alpha=alpha,
        max_samples=int(os.environ.get("MAX_SAMPLES", 5000)),
        seed=42,
    )
    _data_cache.update(key=key, loaders=loaders)
    if pid not in loaders:
        logger.warning(f"Client {pid}: no data for this partition ({ptype} alpha={alpha}), using fallback")
        # Fallback: use the first available partition's data
        fallback_pid = next(iter(loaders))
        train_dl = loaders[fallback_pid]["train"]
        val_dl = loaders[fallback_pid]["val"]
    else:
        train_dl = loaders[pid]["train"]
        val_dl = loaders[pid]["val"]
    logger.info(f"Client {pid}: {len(train_dl.dataset)} train, {len(val_dl.dataset)} val")
    return train_dl, val_dl


class MLPClient(fl.client.NumPyClient):
    def __init__(self, pid, n_clients, input_dim=30):
        self.pid, self.n_clients = pid, n_clients
        self.model = FraudMLP(input_dim)
        self._tmap = build_trainable_to_state_map(self.model)
        self.global_params = None
        self.client_control = self.server_control = self.prev_params = None

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
        strategy = config.get("strategy", "fedavg")
        lr = float(config.get("learning_rate", 0.001))
        epochs = int(config.get("local_epochs", 1))
        ptype = config.get("partition_type", "iid")
        alpha = float(config.get("alpha", 100.0))
        prox_mu = float(config["proximal_mu"]) if "proximal_mu" in config else None
        secagg_seed = int(config["secagg_round_seed"]) if "secagg_round_seed" in config else None
        secagg_n = int(config.get("secagg_num_clients", self.n_clients))
        dp_mode = config.get("dp_mode")
        dp_noise = float(config.get("dp_noise_multiplier", 1.0))
        dp_clip = float(config.get("dp_max_grad_norm", 1.0))
        dp_seed = int(config["dp_seed"]) if "dp_seed" in config else None

        train_dl, _ = _load(self.pid, self.n_clients, ptype, alpha)
        self.prev_params = [p.copy() for p in parameters]
        self.set_parameters(parameters)

        if strategy == "scaffold":
            if not self.client_control:
                self.client_control = [np.zeros_like(p) for p in parameters]
            if not self.server_control:
                self.server_control = [np.zeros_like(p) for p in parameters]
        if prox_mu is not None:
            self.global_params = [p.clone().detach() for p in self.model.parameters()]

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        crit = nn.BCELoss()
        self.model.train()
        total_loss, nb = 0.0, 0

        for _ in range(epochs):
            for X, y in train_dl:
                opt.zero_grad()
                loss = crit(self.model(X).clamp(1e-7, 1-1e-7), y)
                if prox_mu and self.global_params:
                    loss = loss + (prox_mu / 2) * sum(
                        ((lp - gp)**2).sum() for lp, gp in zip(self.model.parameters(), self.global_params))
                loss.backward()
                if strategy == "scaffold" and self.server_control:
                    with torch.no_grad():
                        for ti, p in enumerate(self.model.parameters()):
                            if p.grad is not None:
                                si = self._tmap[ti]
                                p.grad.add_(torch.tensor(
                                    self.server_control[si] - self.client_control[si],
                                    dtype=p.grad.dtype))
                opt.step()
                total_loss += loss.item(); nb += 1

        new_p = self.get_parameters({})
        if strategy == "scaffold" and self.prev_params is not None:
            K = epochs * len(train_dl)
            trainable_indices = set(self._tmap.values())
            for i in range(len(self.client_control)):
                if i not in trainable_indices:
                    continue
                d = (self.prev_params[i] - new_p[i]) / (K * lr)
                self.client_control[i] = self.client_control[i] - self.server_control[i] + d
        if strategy == "secagg" and secagg_seed is not None:
            new_p = secagg_mask_parameters(new_p, self.pid, secagg_n, int(secagg_seed))
        if strategy == "dp" and dp_mode == "local":
            seed = int(dp_seed) + self.pid if dp_seed else None
            new_p = clip_and_noise(self.prev_params, new_p, dp_clip, dp_noise, seed)

        return new_p, len(train_dl.dataset), {"loss": total_loss / max(nb, 1)}

    def evaluate(self, parameters, config):
        ptype = config.get("partition_type", "iid")
        alpha = float(config.get("alpha", 100.0))
        _, val_dl = _load(self.pid, self.n_clients, ptype, alpha)
        self.set_parameters(parameters)
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
    n = int(context.node_config.get("num-clients", 0) or context.node_config.get("num-partitions", 0) or 5)
    return MLPClient(pid, n).to_client()

app = ClientApp(client_fn=client_fn)
