"""Flower ClientApp for distributed FL.

Used by flower-supernode / flwr run. Replaces the deprecated start_client().

The SuperNode loads this module and calls client_fn() to create a client
for each training round.

Usage:
    # Start SuperNode (on each client EC2)
    flower-supernode --insecure \
        --superlink coordinator:9092 \
        --clientappio-api-address 0.0.0.0:7070 \
        --node-config "partition-id=0 num-clients=2 task=fraud"
"""

import os
import sys
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flwr.client import ClientApp
from flwr.common import Context

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("client_app")


def client_fn(context: Context):
    """Called by SuperNode to create a client for each round."""
    # Read config from node_config (set via --node-config on CLI)
    cfg = context.node_config if hasattr(context, 'node_config') and context.node_config else {}
    partition_id = int(cfg.get("partition-id", os.environ.get("PARTITION_ID", "0")))
    num_clients = int(cfg.get("num-clients", os.environ.get("NUM_CLIENTS", "2")))
    task = cfg.get("task", os.environ.get("FL_TASK", "fraud"))

    os.environ["PARTITION_ID"] = str(partition_id)
    os.environ["NUM_CLIENTS"] = str(num_clients)
    data_path = os.environ.get("DATA_PATH", "/data")
    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))

    logger.info("ClientApp: task=%s partition=%d/%d", task, partition_id, num_clients)

    if task in ("sepsis", "ecg"):
        input_dim = 14 if task == "sepsis" else 12
        os.environ["TASK"] = task
        os.environ["INPUT_DIM"] = str(input_dim)
        from models.hfl.bilstm.client_app import BiLSTMClient
        return BiLSTMClient(partition_id, num_clients, data_path, max_samples, task, input_dim)
    elif task == "fraud":
        from models.hfl.mlp.client_app import MLPClient
        return MLPClient(partition_id, num_clients)
    elif task in ("chest", "transfer"):
        synthetic = os.environ.get("SYNTHETIC", "1") == "1"
        dataset_path = os.environ.get("DATASET_PATH", "/data/chest_xray")
        csv_path = os.environ.get("CSV_PATH", "Data_Entry_2017.csv")
        from models.hfl.densenet.client_app import ChestClient
        return ChestClient(partition_id, num_clients, synthetic, dataset_path, csv_path)
    elif task == "vfl_fraud":
        from models.vfl.vfl_mlp.client_app import VFLClient
        return VFLClient(partition_id, num_clients)
    elif task == "split_sepsis":
        os.environ["INPUT_DIM"] = "14"
        from models.vfl.split_bilstm.client_app import SplitClient
        return SplitClient(partition_id, num_clients)
    elif task == "anomaly":
        from models.hfl.autoencoder.client_app import AutoencoderClient
        return AutoencoderClient(partition_id, num_clients)
    elif task == "mortality":
        from models.hfl.tabnet_simple.client_app import TabNetClient
        return TabNetClient(partition_id, num_clients)
    elif task == "satellite":
        from models.hfl.resnet_small.client_app import ResNetClient
        return ResNetClient(partition_id, num_clients)
    elif task == "readmission":
        from models.hfl.logreg.client_app import LogRegClient
        return LogRegClient(partition_id, num_clients)
    elif task == "drug":
        os.environ["GENERIC_INPUT_DIM"] = "200"
        os.environ["GENERIC_NUM_CLASSES"] = "2"
        os.environ["GENERIC_TASK_TYPE"] = "binary"
        os.environ["GENERIC_MODEL"] = "mlp"
        os.environ["GENERIC_HIDDEN"] = "128"
        os.environ["GENERIC_DATA_MODULE"] = "tasks.drug.data"
        from models.hfl.generic.client_app import GenericClient
        return GenericClient(partition_id, num_clients)
    elif task == "olmo":
        from models.llm.olmo.client_app import OLMoClient
        return OLMoClient(partition_id, num_clients)
    else:
        raise ValueError(f"Unknown task: {task}. Available: fraud, sepsis, ecg, chest, "
                         "vfl_fraud, split_sepsis, anomaly, mortality, satellite, readmission, drug, olmo")


# The ClientApp instance that flower-supernode loads
app = ClientApp(client_fn=client_fn)
