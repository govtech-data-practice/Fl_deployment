"""BiLSTM ClientApp — task dispatch via TASK env/config (sepsis, ecg)."""
import sys, os, importlib, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np, torch, torch.nn as nn, torch.optim as optim
import flwr as fl
from flwr.client import Client
from flwr.clientapp import ClientApp
from flwr.common import Context
from fl_common import build_trainable_to_state_map, secagg_mask_parameters, clip_and_noise

logger = logging.getLogger("bilstm.client")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_MODULES = {"sepsis": "tasks.sepsis.data", "ecg": "tasks.ecg.data"}

class BiLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)

_cache = {"key": None, "train": None, "val": None}

def _load(pid, n_clients, ptype, alpha, data_path, max_samples, task):
    key = "%s_%s_%s_%s" % (task, ptype, alpha, pid)
    if _cache["key"] == key and _cache["train"]:
        return _cache["train"], _cache["val"]
    mod = importlib.import_module(TASK_MODULES.get(task, "tasks.sepsis.data"))
    loaders, _ = mod.prepare_federated_data(
        data_path=data_path, num_clients=n_clients,
        partition_type=ptype, alpha=alpha, max_samples=max_samples)
    if pid not in loaders:
        pid = sorted(loaders.keys())[pid % len(loaders)]
    _cache.update(key=key, train=loaders[pid]["train"], val=loaders[pid]["val"])
    return _cache["train"], _cache["val"]

class BiLSTMClient(fl.client.NumPyClient):
    def __init__(self, pid, n_clients, data_path, max_samples, task, input_dim):
        self.pid, self.n_clients = pid, n_clients
        self.data_path, self.max_samples, self.task = data_path, max_samples, task
        self.model = BiLSTM(input_dim=input_dim).to(DEVICE)
        self._tmap = build_trainable_to_state_map(self.model)
        self.global_params = self.client_control = self.server_control = self.prev_params = None

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]
    def set_parameters(self, params):
        sd = dict(zip(self.model.state_dict().keys(), params))
        self.model.load_state_dict({k: torch.tensor(v, device=DEVICE) for k, v in sd.items()})

    def fit(self, parameters, config):
        strategy = config.get("strategy", "fedavg")
        lr = float(config.get("learning_rate", 0.001))
        ptype, alpha = config.get("partition_type", "iid"), float(config.get("alpha", 100.0))
        prox_mu = float(config["proximal_mu"]) if "proximal_mu" in config else None
        secagg_seed = int(config["secagg_round_seed"]) if "secagg_round_seed" in config else None
        secagg_n = int(config.get("secagg_num_clients", self.n_clients))
        dp_mode = config.get("dp_mode")
        dp_noise, dp_clip = float(config.get("dp_noise_multiplier", 1.0)), float(config.get("dp_max_grad_norm", 5.0))
        dp_seed = int(config["dp_seed"]) if "dp_seed" in config else None

        train, _ = _load(self.pid, self.n_clients, ptype, alpha, self.data_path, self.max_samples, self.task)
        self.prev_params = [p.copy() for p in parameters]
        self.set_parameters(parameters)
        if strategy == "scaffold":
            if not self.client_control: self.client_control = [np.zeros_like(p) for p in parameters]
            if not self.server_control: self.server_control = [np.zeros_like(p) for p in parameters]
        if prox_mu is not None:
            self.global_params = [p.clone().detach() for p in self.model.parameters()]

        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        crit = nn.BCELoss(); self.model.train(); tl, nb = 0.0, 0
        for X, y in train:
            X, y = X.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            loss = crit(self.model(X), y)
            if prox_mu and self.global_params:
                loss = loss + (prox_mu/2)*sum(((a-b)**2).sum() for a,b in zip(self.model.parameters(), self.global_params))
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if strategy == "scaffold" and self.server_control:
                with torch.no_grad():
                    for ti, p in enumerate(self.model.parameters()):
                        if p.grad is not None:
                            si = self._tmap[ti]
                            p.grad.add_(torch.tensor(self.server_control[si]-self.client_control[si], device=DEVICE, dtype=p.grad.dtype))
            opt.step(); tl += loss.item(); nb += 1

        new_p = self.get_parameters({})
        if strategy == "scaffold" and self.prev_params is not None:
            ti_set = set(self._tmap.values())
            for i in range(len(self.client_control)):
                if i not in ti_set: continue
                self.client_control[i] = self.client_control[i] - self.server_control[i] + (self.prev_params[i]-new_p[i])/(len(train)*lr)
        if strategy == "secagg" and secagg_seed is not None:
            new_p = secagg_mask_parameters(new_p, self.pid, secagg_n, secagg_seed)
        if strategy == "dp" and dp_mode == "local":
            new_p = clip_and_noise(self.prev_params, new_p, dp_clip, dp_noise, (dp_seed+self.pid) if dp_seed else None)
        return new_p, len(train.dataset), {"loss": tl/max(nb,1)}

    def evaluate(self, parameters, config):
        ptype, alpha = config.get("partition_type", "iid"), float(config.get("alpha", 100.0))
        _, val = _load(self.pid, self.n_clients, ptype, alpha, self.data_path, self.max_samples, self.task)
        self.set_parameters(parameters); self.model.eval()
        crit = nn.BCELoss(); loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val:
                X, y = X.to(DEVICE), y.to(DEVICE); pred = self.model(X)
                loss += crit(pred, y).item(); correct += ((pred>0.5).float()==y).sum().item(); total += y.size(0)
        return loss/max(len(val),1), total, {"accuracy": correct/total if total else 0.0}

def client_fn(context: Context) -> Client:
    nc = context.node_config
    pid = int(nc.get("partition-id", 0))
    n = int(nc.get("num-clients",0) or nc.get("num-partitions",0) or os.environ.get("NUM_CLIENTS","2"))
    data = str(nc.get("data-path","") or os.environ.get("DATA_PATH","/data/flower_data"))
    ms = int(nc.get("max-samples",0) or os.environ.get("MAX_SAMPLES","0"))
    task = str(nc.get("task","") or os.environ.get("TASK","sepsis"))
    idim = int(nc.get("input-dim",0) or os.environ.get("INPUT_DIM","14"))
    return BiLSTMClient(pid, n, data, ms, task, idim).to_client()

app = ClientApp(client_fn=client_fn)
