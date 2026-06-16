"""Autoencoder ServerApp — for anomaly detection (reconstruction-based)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy


class Autoencoder(nn.Module):
    def __init__(self, input_dim=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_strategy(name, num_clients, input_dim=40):
    return build_strategy(name, num_clients, model_init_fn=lambda: Autoencoder(input_dim),
                          metric_name="auc", lr=0.001, patience=15)


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 5))
    return ServerAppComponents(strategy=make_strategy(name, n_clients),
                               config=ServerConfig(num_rounds=n_rounds))

app = ServerApp(server_fn=server_fn)
