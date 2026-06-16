"""Logistic Regression ServerApp — simplest baseline for binary classification."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x)).squeeze(-1)


def make_strategy(name, num_clients, input_dim=20):
    return build_strategy(name, num_clients, model_init_fn=lambda: LogisticRegression(input_dim),
                          metric_name="accuracy", lr=0.01, patience=15)


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 5))
    return ServerAppComponents(strategy=make_strategy(name, n_clients),
                               config=ServerConfig(num_rounds=n_rounds))

app = ServerApp(server_fn=server_fn)
