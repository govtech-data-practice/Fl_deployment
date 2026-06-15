"""Client (ClientApp) facade.

Dispatches to per-model client implementations in models/<model>/client_app.py.
The guide references a single ``clientapp/`` entry point; this module provides
that interface while preserving the per-model architecture underneath.

Usage:
    from clientapp import get_client_app, MODEL_REGISTRY
    app = get_client_app("mlp")
"""

import importlib

MODEL_REGISTRY = {
    "autoencoder":   "models.autoencoder.client_app",
    "bilstm":        "models.bilstm.client_app",
    "cnn1d":         "models.cnn1d.client_app",
    "densenet":      "models.densenet.client_app",
    "generic":       "models.generic.client_app",
    "logreg":        "models.logreg.client_app",
    "mlp":           "models.mlp.client_app",
    "resnet_small":  "models.resnet_small.client_app",
    "split_bilstm":  "models.split_bilstm.client_app",
    "tabnet_simple": "models.tabnet_simple.client_app",
    "vfl_mlp":       "models.vfl_mlp.client_app",
}


def get_client_app(model_name: str):
    """Import and return the ClientApp module for the given model.

    Returns the module, which exposes ``app`` (the Flower ClientApp instance)
    and the client class for programmatic use.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return importlib.import_module(MODEL_REGISTRY[model_name])
