"""Small ResNet ServerApp — for image classification (satellite imagery)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class SmallResNet(nn.Module):
    """Small ResNet with 2 residual blocks, ~200K parameters.

    Architecture: Conv2d(3,16) -> ResBlock(16) -> ResBlock(32) -> AvgPool -> Linear(32, num_classes)
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResidualBlock(16, 16, stride=1)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def make_strategy(name, num_clients, num_classes=5):
    return build_strategy(name, num_clients,
                          model_init_fn=lambda: SmallResNet(num_classes),
                          metric_name="accuracy", lr=0.001, patience=15)


def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    name = str(cfg.get("strategy", "IID"))
    n_rounds = int(cfg.get("num-rounds", 5))
    n_clients = int(cfg.get("num-clients", 5))
    return ServerAppComponents(strategy=make_strategy(name, n_clients),
                               config=ServerConfig(num_rounds=n_rounds))

app = ServerApp(server_fn=server_fn)
