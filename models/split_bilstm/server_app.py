"""Split Learning BiLSTM — Server (Top Model).

Split Learning: clients run the bottom layers (LSTM encoder), server runs
the top layers (classifier). Only activations are shared, not raw data.

In Flower simulation, we approximate this by having each client hold both
bottom + top layers, but only aggregating the top classifier weights.
The bottom LSTM stays local (never shared), demonstrating the split concept.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import logging
import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy

logger = logging.getLogger("split_bilstm.server")


class SplitTopModel(nn.Module):
    """Top model (server-side in split learning).
    Takes LSTM hidden state embedding and classifies.
    In simulation, this is aggregated across clients via FedAvg.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


def make_strategy(name, num_clients):
    return build_strategy(
        name, num_clients,
        model_init_fn=lambda: SplitTopModel(128),
        metric_name="accuracy", lr=0.001, patience=15,
    )


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 5))
    return ServerAppComponents(
        strategy=make_strategy(name, n_clients),
        config=ServerConfig(num_rounds=n_rounds),
    )


app = ServerApp(server_fn=server_fn)
