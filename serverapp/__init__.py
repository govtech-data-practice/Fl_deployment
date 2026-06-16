"""Coordinator (ServerApp) facade.

Dispatches to per-model server implementations in models/<model>/server_app.py.
The guide references a single ``serverapp/`` entry point; this module provides
that interface while preserving the per-model architecture underneath.

Usage:
    from serverapp import get_server_app, MODEL_REGISTRY
    app = get_server_app("mlp")
"""

import importlib

MODEL_REGISTRY = {
    # HFL
    "autoencoder":   "models.hfl.autoencoder.server_app",
    "bilstm":        "models.hfl.bilstm.server_app",
    "cnn1d":         "models.hfl.cnn1d.server_app",
    "densenet":      "models.hfl.densenet.server_app",
    "generic":       "models.hfl.generic.server_app",
    "logreg":        "models.hfl.logreg.server_app",
    "mlp":           "models.hfl.mlp.server_app",
    "resnet_small":  "models.hfl.resnet_small.server_app",
    "tabnet_simple": "models.hfl.tabnet_simple.server_app",
    # VFL
    "split_bilstm":  "models.vfl.split_bilstm.server_app",
    "vfl_mlp":       "models.vfl.vfl_mlp.server_app",
    # LLM
    "olmo":          "models.llm.olmo.server_app",
}


def get_server_app(model_name: str):
    """Import and return the ServerApp for the given model.

    Returns the module, which exposes ``app`` (the Flower ServerApp instance)
    and ``server_fn`` / ``make_strategy`` for programmatic use.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return importlib.import_module(MODEL_REGISTRY[model_name])
