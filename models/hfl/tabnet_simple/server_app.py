"""Simple TabNet ServerApp — attention-based tabular model for mortality prediction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy


class SimpleTabNet(nn.Module):
    """Lightweight attention-based tabular model.

    Feature transform -> attention weights (softmax) -> weighted features -> output.
    Attention weights provide interpretable feature importance per sample.
    """
    def __init__(self, input_dim=25, hidden_dim=64):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softmax(dim=-1),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.feature_transform(x)
        attn = self.attention(h)
        h = h * attn
        return self.output_layer(h).squeeze(-1)

    def get_attention_weights(self, x):
        """Return attention weights for interpretability."""
        with torch.no_grad():
            h = self.feature_transform(x)
            return self.attention(h)


def make_strategy(name, num_clients, input_dim=25):
    return build_strategy(name, num_clients,
                          model_init_fn=lambda: SimpleTabNet(input_dim),
                          metric_name="accuracy", lr=0.001, patience=15)


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 5))
    return ServerAppComponents(strategy=make_strategy(name, n_clients),
                               config=ServerConfig(num_rounds=n_rounds))

app = ServerApp(server_fn=server_fn)
