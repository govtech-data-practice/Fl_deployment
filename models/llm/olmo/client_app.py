"""
OLMo Federated LoRA Client — uses the generic federated adapter framework.

This is a thin wrapper around FederatedAdapterClient. To switch to a
different model (Llama, Mistral, BERT, Whisper), change the preset or config.

Usage:
    FL_TASK=olmo ADAPTER_MODEL_ID=allenai/OLMo-1B-hf python run_client.py
    FL_TASK=olmo ADAPTER_MODEL_ID=meta-llama/Meta-Llama-3-8B python run_client.py
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import flwr as fl
from flwr.client import Client
from flwr.common import Context

from fl_common.federated_adapter import (
    AdapterConfig, FederatedAdapterClient, PRESETS, load_model,
)

logger = logging.getLogger("olmo.client")


class OLMoFlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient wrapper around FederatedAdapterClient."""

    def __init__(self, partition_id: int, num_clients: int):
        # Use preset or env-based config
        preset = os.environ.get("ADAPTER_PRESET", "olmo-1b")
        if preset in PRESETS:
            config = PRESETS[preset]
        else:
            config = AdapterConfig.from_env()

        # Override from env
        config.max_samples = int(os.environ.get("MAX_SAMPLES", str(config.max_samples)))

        self.adapter = FederatedAdapterClient(config, partition_id, num_clients)
        self._data_loaded = False

    def _ensure_data(self):
        if self._data_loaded:
            return
        from tasks.llm.gov_doc.data import prepare_federated_data
        loaders, meta = prepare_federated_data(
            num_clients=self.adapter.n_clients,
            max_samples=self.adapter.config.max_samples,
            seed=42,
        )
        pid = self.adapter.pid
        if pid not in loaders:
            pid = next(iter(loaders))
        self.adapter.load_data(loaders[pid]["train"], loaders[pid]["val"])
        self._data_loaded = True

    def get_parameters(self, config):
        return self.adapter.get_parameters()

    def set_parameters(self, params):
        self.adapter.set_parameters(params)

    def fit(self, parameters, config):
        self._ensure_data()
        return self.adapter.fit(parameters, config)

    def evaluate(self, parameters, config):
        self._ensure_data()
        return self.adapter.evaluate(parameters, config)


# Alias for backward compat
OLMoClient = OLMoFlowerClient


def client_fn(context: Context) -> Client:
    nc = context.node_config
    pid = int(nc.get("partition-id", 0))
    n = int(nc.get("num-clients", 0) or os.environ.get("NUM_CLIENTS", "5"))
    return OLMoFlowerClient(pid, n).to_client()
