"""MLP ServerApp — for tabular data (fraud, structured clinical, claims)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy

class FraudMLP(nn.Module):
    def __init__(self, input_dim=30, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def make_strategy(name, num_clients, input_dim=30):
    return build_strategy(name, num_clients, model_init_fn=lambda: FraudMLP(input_dim),
                          metric_name="accuracy", lr=0.001, patience=15)

def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 5))
    return ServerAppComponents(strategy=make_strategy(name, n_clients),
                               config=ServerConfig(num_rounds=n_rounds))

app = ServerApp(server_fn=server_fn)
