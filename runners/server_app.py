"""Flower ServerApp for distributed FL.

Used by flower-superlink / flwr run. Replaces the deprecated start_server().

The SuperLink loads this module and calls server_fn() to get the strategy
and config for each experiment run.

Usage:
    # Start SuperLink (on coordinator EC2)
    flower-superlink --insecure

    # Deploy ServerApp
    flwr run . server --app runners.server_app:app --insecure
"""

import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("server_app")

TASK = os.environ.get("FL_TASK", "fraud")
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "10"))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "2"))


def _load_strategy(task, num_clients):
    """Load the strategy for the given task."""
    os.environ["NUM_CLIENTS"] = str(num_clients)

    if task in ("sepsis", "ecg"):
        input_dim = 14 if task == "sepsis" else 12
        os.environ["TASK"] = task
        os.environ["INPUT_DIM"] = str(input_dim)
        from models.hfl.bilstm.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task == "fraud":
        from models.hfl.mlp.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task in ("chest", "transfer"):
        from models.hfl.densenet.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task == "vfl_fraud":
        from models.vfl.vfl_mlp.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task == "split_sepsis":
        os.environ["INPUT_DIM"] = "14"
        from models.vfl.split_bilstm.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task == "anomaly":
        from models.hfl.autoencoder.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task == "mortality":
        from models.hfl.tabnet_simple.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task == "satellite":
        from models.hfl.resnet_small.server_app import make_strategy
        return make_strategy("IID", num_clients)
    elif task == "readmission":
        from models.hfl.logreg.server_app import make_strategy
        return make_strategy("IID", num_clients)
    else:
        from models.hfl.mlp.server_app import make_strategy
        return make_strategy("IID", num_clients)


def server_fn(context: Context) -> ServerAppComponents:
    """Called by SuperLink to get strategy and config."""
    # Read config from context (set via --node-config or env vars)
    task = context.node_config.get("task", TASK) if hasattr(context, 'node_config') and context.node_config else TASK
    num_rounds = int(context.node_config.get("num-rounds", NUM_ROUNDS)) if hasattr(context, 'node_config') and context.node_config else NUM_ROUNDS
    num_clients = int(context.node_config.get("num-clients", NUM_CLIENTS)) if hasattr(context, 'node_config') and context.node_config else NUM_CLIENTS

    logger.info("ServerApp: task=%s rounds=%d clients=%d", task, num_rounds, num_clients)

    strategy = _load_strategy(task, num_clients)

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


# The ServerApp instance that flower-superlink loads
app = ServerApp(server_fn=server_fn)
