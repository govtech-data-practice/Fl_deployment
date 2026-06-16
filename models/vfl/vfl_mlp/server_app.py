"""VFL Fraud Detection — Server (Top Model).

Vertical FL: each client holds a subset of features (e.g., Bank A has transaction
features, Bank B has account features). Clients send embeddings, not raw features.
Server aggregates embeddings and runs the classifier.

This uses Flower simulation to mimic VFL by:
1. Each client trains a bottom model on its feature subset
2. The server strategy aggregates the bottom model weights (standard FedAvg)
3. Each client also holds a local top model for end-to-end training

In real VFL, only activations (not weights) would be shared. This simulation
demonstrates the concept; production VFL would use a split-learning protocol.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import logging
import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy

logger = logging.getLogger("vfl_fraud.server")


class VFLBottomModel(nn.Module):
    """Bottom model that each client runs on its feature subset.
    All clients share the same architecture but see different features.
    """
    def __init__(self, input_dim=10, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, embed_dim), nn.ReLU(),
        )
        # Local top model for end-to-end training in simulation
        # In real VFL, this would be on the server only
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        embed = self.net(x)
        return self.classifier(embed).squeeze(-1)

    def get_embedding(self, x):
        return self.net(x)


def make_strategy(name, num_clients, input_dim=10):
    return build_strategy(
        name, num_clients,
        model_init_fn=lambda: VFLBottomModel(input_dim),
        metric_name="accuracy", lr=0.001, patience=15,
    )


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 3))
    return ServerAppComponents(
        strategy=make_strategy(name, n_clients),
        config=ServerConfig(num_rounds=n_rounds),
    )


app = ServerApp(server_fn=server_fn)
