"""BiLSTM ServerApp — for time series (sepsis, ECG, waveforms)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from fl_common import build_strategy


class BiLSTM(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)


def make_strategy(name, num_clients, input_dim=14):
    return build_strategy(name, num_clients, model_init_fn=lambda: BiLSTM(input_dim=input_dim),
                          metric_name="accuracy", lr=0.001, patience=30)

def server_fn(context: Context) -> ServerAppComponents:
    cfg = context.run_config
    return ServerAppComponents(
        strategy=make_strategy(str(cfg.get("strategy", "IID")),
                               int(cfg.get("num-clients", 2)),
                               int(cfg.get("input-dim", 14))),
        config=ServerConfig(num_rounds=int(cfg.get("num-rounds", 3))),
    )

app = ServerApp(server_fn=server_fn)
