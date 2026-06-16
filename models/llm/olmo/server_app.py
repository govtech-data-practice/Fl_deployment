"""OLMo ServerApp — strategy factory for federated OLMo LoRA."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from fl_common.strategies import build_strategy


def _get_model():
    """Build OLMo with LoRA for parameter initialization."""
    from models.llm.olmo.client_app import _load_model
    model, _ = _load_model()
    return model


def make_strategy(name, num_clients):
    return build_strategy(
        name, num_clients,
        model_init_fn=_get_model,
        metric_name="perplexity",
        lr=2e-4,
        patience=10,
    )
