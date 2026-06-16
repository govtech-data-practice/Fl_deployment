"""
Generic ServerApp — auto-adapts to any tabular dataset via config.
Uses the same model as client, initialized with correct dimensions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn as nn
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context

from fl_common.strategies import build_strategy
from models.hfl.generic.client_app import MODEL_REGISTRY, GenericMLP


def _get_model():
    """Build model from env config."""
    input_dim = int(os.environ.get("GENERIC_INPUT_DIM", "30"))
    num_classes = int(os.environ.get("GENERIC_NUM_CLASSES", "2"))
    task_type = os.environ.get("GENERIC_TASK_TYPE", "binary")
    model_name = os.environ.get("GENERIC_MODEL", "mlp")
    hidden = int(os.environ.get("GENERIC_HIDDEN", "64"))

    model_cls = MODEL_REGISTRY.get(model_name, GenericMLP)
    return model_cls(input_dim, num_classes, hidden=hidden, task_type=task_type)


def make_strategy(name, num_clients):
    return build_strategy(name, num_clients, model_init_fn=_get_model,
                          metric_name="accuracy", lr=0.001, patience=15)


def server_fn(ctx):
    return ServerAppComponents(
        strategy=make_strategy("IID", 5),
        config=ServerConfig(num_rounds=10),
    )


app = ServerApp(server_fn=server_fn)
