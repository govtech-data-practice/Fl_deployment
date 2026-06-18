#!/usr/bin/env python3
"""
FL Client Runner — connects to the FL server in distributed mode.

Usage (inside Docker container on each client EC2):
  python run_client.py --server 172.31.4.42:9092 --partition-id 0 --num-clients 5

Data pipeline:
  Each client should have data pre-ingested via ingest.py at ~/fl-deploy/data/<task>/.
  If no ingested data is found, falls back to synthetic data generation.
  Run 'python ingest.py --task <task> --input <data>' to ingest data before training.

The client auto-detects which task the server is running (via environment variables)
and loads the appropriate model + data.
"""

import sys, os, argparse, logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("fl_client")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import flwr as fl
from pathlib import Path
from fl_common.data import DataConfig, DataManifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=os.environ.get("FL_SERVER", "172.31.4.42:9092"))
    p.add_argument("--partition-id", type=int, default=int(os.environ.get("PARTITION_ID", "0")))
    p.add_argument("--num-clients", type=int, default=int(os.environ.get("NUM_CLIENTS", "5")))
    p.add_argument("--task", default=os.environ.get("FL_TASK", "auto"))
    p.add_argument("--certs-dir", default=os.environ.get("CERTS_DIR", "/certs"))
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", ""),
                   help="Path to ingested data directory (default: /data/<task>)")
    return p.parse_args()


def load_certificates(certs_dir, partition_id=0):
    """Load TLS certificates. Supports mTLS if client cert exists."""
    ca = Path(certs_dir) / "ca.pem"
    if not ca.exists():
        logger.warning("No CA cert found at %s — connecting insecure", certs_dir)
        return None

    # Check for mTLS client certificate
    client_cert = Path(certs_dir) / f"client_{partition_id}.pem"
    client_key = Path(certs_dir) / f"client_{partition_id}.key"
    if client_cert.exists() and client_key.exists():
        logger.info("mTLS enabled: using client certificate %s", client_cert.name)
        return (ca.read_bytes(), client_cert.read_bytes(), client_key.read_bytes())

    # Fallback: server-only TLS (no client cert)
    return ca.read_bytes()


def make_client(task, partition_id, num_clients):
    """Create the appropriate Flower NumPyClient based on the task."""

    # Set environment for data loaders
    os.environ["PARTITION_ID"] = str(partition_id)
    os.environ["NUM_CLIENTS"] = str(num_clients)

    data_path = os.environ.get("DATA_PATH", "/data")
    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))

    if task in ("sepsis", "ecg"):
        input_dim = 14 if task == "sepsis" else 12
        os.environ["TASK"] = task
        os.environ["INPUT_DIM"] = str(input_dim)
        from models.hfl.bilstm.client_app import BiLSTMClient
        return BiLSTMClient(partition_id, num_clients, data_path, max_samples, task, input_dim)
    elif task == "fraud":
        from models.hfl.mlp.client_app import MLPClient
        return MLPClient(partition_id, num_clients)
    elif task == "chest":
        synthetic = os.environ.get("SYNTHETIC", "1") == "1"
        dataset_path = os.environ.get("DATASET_PATH", "/data/chest_xray")
        csv_path = os.environ.get("CSV_PATH", "Data_Entry_2017.csv")
        # Fall back to synthetic if real data not available
        if not synthetic and not os.path.exists(os.path.join(dataset_path, csv_path)):
            logger.warning("Real chest X-ray data not found at %s, falling back to synthetic", dataset_path)
            synthetic = True
        from models.hfl.densenet.client_app import ChestClient
        return ChestClient(partition_id, num_clients, synthetic, dataset_path, csv_path)
    elif task == "vfl":
        from models.vfl.vfl_mlp.client_app import VFLClient
        return VFLClient(partition_id, num_clients)
    elif task == "split":
        os.environ["INPUT_DIM"] = "14"
        from models.vfl.split_bilstm.client_app import SplitClient
        return SplitClient(partition_id, num_clients)
    elif task == "anomaly":
        from models.hfl.autoencoder.client_app import AutoencoderClient
        return AutoencoderClient(partition_id, num_clients)
    elif task == "mortality":
        from models.hfl.tabnet_simple.client_app import TabNetClient
        return TabNetClient(partition_id, num_clients)
    elif task == "drug":
        os.environ["GENERIC_INPUT_DIM"] = "200"
        os.environ["GENERIC_NUM_CLASSES"] = "2"
        os.environ["GENERIC_TASK_TYPE"] = "binary"
        os.environ["GENERIC_MODEL"] = "mlp"
        os.environ["GENERIC_HIDDEN"] = "128"
        os.environ["GENERIC_DATA_MODULE"] = "tasks.drug.data"
        from models.hfl.generic.client_app import GenericClient
        return GenericClient(partition_id, num_clients)
    elif task == "satellite":
        from models.hfl.resnet_small.client_app import ResNetClient
        return ResNetClient(partition_id, num_clients)
    elif task == "readmission":
        from models.hfl.logreg.client_app import LogRegClient
        return LogRegClient(partition_id, num_clients)
    elif task == "transfer":
        # Transfer learning uses DenseNet (same as chest)
        synthetic = os.environ.get("SYNTHETIC", "1") == "1"
        dataset_path = os.environ.get("DATASET_PATH", "/data/chest_xray")
        csv_path = os.environ.get("CSV_PATH", "Data_Entry_2017.csv")
        if not synthetic and not os.path.exists(os.path.join(dataset_path, csv_path)):
            logger.warning("Real chest X-ray data not found at %s, falling back to synthetic", dataset_path)
            synthetic = True
        from models.hfl.densenet.client_app import ChestClient
        return ChestClient(partition_id, num_clients, synthetic, dataset_path, csv_path)
    elif task == "olmo":
        from models.llm.olmo.client_app import OLMoClient
        return OLMoClient(partition_id, num_clients)
    else:
        raise ValueError(f"Unknown task: {task}")


def check_data(task, data_dir):
    """Pre-flight data check. Validates manifest if present, logs data status."""
    task_data_dir = data_dir or f"/data/{task}"
    manifest_path = os.path.join(task_data_dir, "manifest.json")

    if os.path.exists(manifest_path):
        try:
            manifest = DataManifest.load(manifest_path)
            config = DataConfig.for_task(task, task_data_dir)
            errors = manifest.validate_against(config)
            if errors:
                logger.warning("Data manifest validation issues: %s", errors)
                return "synthetic"
            logger.info("Data: %s, %d samples, checksum %s",
                        manifest.format, manifest.num_samples, manifest.checksum[:12])
            return "real"
        except Exception as e:
            logger.warning("Failed to load manifest: %s", e)
            return "synthetic"
    else:
        logger.info("No ingested data at %s — using synthetic data", task_data_dir)
        return "synthetic"


def main():
    args = parse_args()
    root_cert = load_certificates(args.certs_dir, args.partition_id)

    task = args.task
    if task == "auto":
        task = os.environ.get("FL_TASK", "fraud")

    # Pre-flight data check
    data_mode = check_data(task, args.data_dir)
    logger.info("Starting FL client: task=%s partition=%d/%d server=%s data=%s",
                task, args.partition_id, args.num_clients, args.server, data_mode)

    # Loop: reconnect to server for each strategy run.
    # The server runs multiple strategies sequentially, each time
    # calling start_server() which accepts new client connections.
    import time
    max_retries = 200  # enough for all strategies across all tasks
    retry_delay = 5
    consecutive_failures = 0

    for attempt in range(max_retries):
        try:
            client = make_client(task, args.partition_id, args.num_clients)
            logger.info("Connecting to server (attempt %d)...", attempt + 1)
            fl.client.start_client(
                server_address=args.server,
                client=client,
                root_certificates=root_cert,
            )
            logger.info("Strategy run completed, waiting for next...")
            consecutive_failures = 0
            time.sleep(2)  # brief pause between strategy runs
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures > 12:  # ~60s of failures = server is done
                logger.info("Server appears to have finished. Exiting.")
                break
            logger.info("Connection failed (%s), retrying in %ds...", e, retry_delay)
            time.sleep(retry_delay)

    logger.info("Client finished")


if __name__ == "__main__":
    main()
