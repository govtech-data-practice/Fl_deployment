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
    # HFL
    "autoencoder":   "models.hfl.autoencoder.client_app",
    "bilstm":        "models.hfl.bilstm.client_app",
    "cnn1d":         "models.hfl.cnn1d.client_app",
    "densenet":      "models.hfl.densenet.client_app",
    "generic":       "models.hfl.generic.client_app",
    "logreg":        "models.hfl.logreg.client_app",
    "mlp":           "models.hfl.mlp.client_app",
    "resnet_small":  "models.hfl.resnet_small.client_app",
    "tabnet_simple": "models.hfl.tabnet_simple.client_app",
    # VFL
    "split_bilstm":  "models.vfl.split_bilstm.client_app",
    "vfl_mlp":       "models.vfl.vfl_mlp.client_app",
    # LLM
    "olmo":          "models.llm.olmo.client_app",
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
