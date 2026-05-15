"""Chest X-ray FL ServerApp (DenseNet-121)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import logging
import torch.nn as nn
from torchvision.models import densenet121
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy

logger = logging.getLogger("chest.server")
NUM_CLASSES = 14


class ChestXrayDenseNet121(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.2):
        super().__init__()
        self.base_model = densenet121(weights=None)
        nf = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(nf, num_classes), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.base_model(x)


def make_strategy(name, num_clients):
    return build_strategy(
        name, num_clients,
        model_init_fn=lambda: ChestXrayDenseNet121(),
        metric_name="auc", lr=0.0001,
        patience=10, min_delta=0.001,
    )


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    strategy_name = str(cfg.get("strategy", "IID"))
    num_rounds = int(cfg.get("num-rounds", 2))
    num_clients = int(cfg.get("num-clients", 2))
    logger.info(f"Chest ServerApp: {strategy_name}, {num_rounds} rounds, {num_clients} clients")
    return ServerAppComponents(
        strategy=make_strategy(strategy_name, num_clients),
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
