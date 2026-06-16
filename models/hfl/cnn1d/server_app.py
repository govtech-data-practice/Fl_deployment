"""1D CNN ServerApp — for time-series classification (ECG)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy


class CNN1D(nn.Module):
    def __init__(self, in_channels=12, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        h = self.features(x)
        # Global average pooling over the time dimension
        h = h.mean(dim=-1)
        return self.classifier(h)


def make_strategy(name, num_clients, in_channels=12, num_classes=4):
    return build_strategy(name, num_clients,
                          model_init_fn=lambda: CNN1D(in_channels, num_classes),
                          metric_name="accuracy", lr=0.001, patience=15)


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 5))
    return ServerAppComponents(strategy=make_strategy(name, n_clients),
                               config=ServerConfig(num_rounds=n_rounds))

app = ServerApp(server_fn=server_fn)
