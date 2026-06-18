#!/usr/bin/env python3
"""
FL + PET Sandbox — EC2 Runner
==============================
Run federated learning experiments from CLI or YAML scenarios.

Usage:
  python run_ec2.py                           # all tasks (simulation)
  python run_ec2.py sepsis                    # single task (simulation)
  python run_ec2.py --distributed             # all tasks (distributed via SuperLink)
  python run_ec2.py --distributed sepsis      # single task (distributed)
  python run_ec2.py scenarios/quick_fraud.yaml  # YAML scenario
  python run_ec2.py privacy                   # privacy attack suite
"""

import sys, os, time, json, logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger("run_ec2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import numpy as np
from flwr.server import ServerApp, ServerConfig, ServerAppComponents, start_server
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.common import Context

# ── Mode Configuration ───────────────────────────────────────────────
DISTRIBUTED = "--distributed" in sys.argv or os.environ.get("FL_DISTRIBUTED", "") == "1"
if DISTRIBUTED:
    sys.argv = [a for a in sys.argv if a != "--distributed"]

# ── Scale Configuration ──────────────────────────────────────────────
# 5 clients = 5 hospitals/agencies, realistic for cross-silo FL
NC = 5
GPU_PER_CLIENT = 0.2 if torch.cuda.is_available() else 0.0

# ── TLS Configuration (distributed mode) ─────────────────────────────
SUPERLINK_ADDRESS = os.environ.get("SUPERLINK_ADDRESS", "0.0.0.0:9092")
CERTS_DIR = os.environ.get("CERTS_DIR", "/certs")

def _load_certificates():
    """Load TLS certificates for distributed mode."""
    ca = Path(CERTS_DIR) / "ca.pem"
    cert = Path(CERTS_DIR) / "server.pem"
    key = Path(CERTS_DIR) / "server.key"
    if ca.exists() and cert.exists() and key.exists():
        return (ca.read_bytes(), cert.read_bytes(), key.read_bytes())
    logger.warning("TLS certs not found in %s — running insecure", CERTS_DIR)
    return None

# Strategy configs: name → (num_rounds, description)
STRATEGIES = {
    # Baseline
    "IID": (30, "IID baseline — upper bound"),
    # Non-IID: alpha controls heterogeneity (0.1=extreme, 0.5=moderate, 1.0=mild)
    "FedProx_Mu0.1_Alpha_0.5": (30, "FedProx moderate non-IID"),
    "FedProx_Mu0.1_Alpha_0.1": (40, "FedProx extreme non-IID"),
    "SCAFFOLD_Alpha_0.5": (30, "SCAFFOLD moderate non-IID"),
    "SCAFFOLD_Alpha_0.1": (40, "SCAFFOLD extreme non-IID"),
    "SecAgg_Alpha_0.5": (30, "SecAgg+ moderate non-IID"),
    "DP_Central_Eps50.0_Alpha_0.5": (30, "DP Central epsilon=50"),
    "DP_Central_Eps10.0_Alpha_0.5": (40, "DP Central epsilon=10"),
    "DP_Local_Eps50.0_Alpha_0.5": (30, "DP Local epsilon=50"),
    "DP_Local_Eps10.0_Alpha_0.5": (40, "DP Local epsilon=10"),
    "OneOwner_Alpha_0.5": (30, "Single-owner model distribution"),
}

# Per-task configs
TASK_CONFIG = {
    "sepsis": {
        "rounds_mult": 1.0,      # multiplier on rounds
        "max_samples": 0,        # 0 = use all data
        "metric": "accuracy",
        "strategies": list(STRATEGIES.keys()),
    },
    "ecg": {
        "rounds_mult": 1.0,
        "max_samples": 10000,
        "metric": "accuracy",
        "strategies": list(STRATEGIES.keys()),
    },
    "fraud": {
        "rounds_mult": 1.0,
        "max_samples": 50000,
        "metric": "accuracy",
        "strategies": list(STRATEGIES.keys()),
    },
    "chest": {
        "rounds_mult": 1.5,      # DenseNet needs more rounds
        "max_samples": 0,
        "metric": "auc",
        # Skip DP on DenseNet (8M params — noise destroys signal)
        "strategies": [
            "IID",
            "FedProx_Mu0.1_Alpha_0.5",
            "FedProx_Mu0.1_Alpha_0.1",
            "SCAFFOLD_Alpha_0.5",
            "SCAFFOLD_Alpha_0.1",
            "SecAgg_Alpha_0.5",
            "OneOwner_Alpha_0.5",
        ],
    },
    # ── New tasks ────────────────────────────────────────────────────
    "anomaly": {
        "rounds_mult": 1.0,
        "max_samples": 8000,
        "metric": "auc",
        "strategies": list(STRATEGIES.keys()),
    },
    "mortality": {
        "rounds_mult": 1.0,
        "max_samples": 6000,
        "metric": "accuracy",
        "strategies": list(STRATEGIES.keys()),
    },
    "drug": {
        "rounds_mult": 1.0,
        "max_samples": 5000,
        "metric": "accuracy",
        "strategies": list(STRATEGIES.keys()),
    },
    "satellite": {
        "rounds_mult": 1.0,
        "max_samples": 3000,
        "metric": "accuracy",
        "strategies": [
            "IID",
            "FedProx_Mu0.1_Alpha_0.5",
            "FedProx_Mu0.1_Alpha_0.1",
            "SCAFFOLD_Alpha_0.5",
            "SCAFFOLD_Alpha_0.1",
            "SecAgg_Alpha_0.5",
            "OneOwner_Alpha_0.5",
        ],
    },
    "readmission": {
        "rounds_mult": 1.0,
        "max_samples": 6000,
        "metric": "accuracy",
        "strategies": list(STRATEGIES.keys()),
    },
    "olmo": {
        "rounds_mult": 0.5,      # fewer rounds for LLM
        "max_samples": 200,
        "metric": "perplexity",
        "strategies": ["IID", "FedProx_Mu0.1_Alpha_0.5", "OneOwner_Alpha_0.5"],
    },
}

# ── Results logging ──────────────────────────────────────────────────
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class MetricCapture:
    """Wraps a strategy to capture final metric value and per-round history."""
    def __init__(self, strat, key):
        self.strategy = strat
        self.key = key
        self.val = 0.0
        self.history = []

    def __getattr__(self, n):
        return getattr(self.strategy, n)

    def configure_fit(self, *a, **kw):
        return self.strategy.configure_fit(*a, **kw)

    def configure_evaluate(self, *a, **kw):
        return self.strategy.configure_evaluate(*a, **kw)

    def aggregate_fit(self, *a, **kw):
        return self.strategy.aggregate_fit(*a, **kw)

    def aggregate_evaluate(self, *a, **kw):
        loss, m = self.strategy.aggregate_evaluate(*a, **kw)
        if loss is not None:
            v = float(m.get(self.key, 0.0))
            self.val = v
            self.history.append({"round": len(self.history) + 1, self.key: v, "loss": float(loss)})
        return loss, m


def run_fl(make_strat, client_fn, strat_name, metric_key, num_rounds, gpu=0.0, num_clients=None):
    nc = num_clients or NC
    cap = MetricCapture(make_strat(strat_name, nc), metric_key)

    t0 = time.time()
    try:
        if DISTRIBUTED:
            # Distributed mode: this process IS the FL server.
            # SuperNodes (on client EC2s) connect to this server directly.
            # The SuperLink container must be stopped first — this takes over port 9092.
            certs = _load_certificates()
            logger.info("    [DISTRIBUTED] Starting FL server on %s (%d clients expected)",
                        SUPERLINK_ADDRESS, nc)
            start_server(
                server_address=SUPERLINK_ADDRESS,
                config=ServerConfig(num_rounds=num_rounds, round_timeout=120),
                strategy=cap,
                certificates=certs,
            )
        else:
            # Simulation mode: everything runs locally
            def sf(ctx):
                return ServerAppComponents(strategy=cap, config=ServerConfig(num_rounds=num_rounds))
            run_simulation(
                server_app=ServerApp(server_fn=sf),
                client_app=ClientApp(client_fn=client_fn),
                num_supernodes=nc,
                backend_config={"client_resources": {"num_cpus": 2, "num_gpus": gpu}},
            )
        return cap.val, cap.history, time.time() - t0
    except Exception as e:
        logger.error("  ERROR: %s" % e)
        return 0.0, cap.history, time.time() - t0


# ── Task runners ─────────────────────────────────────────────────────

def run_sepsis():
    logger.info("\n" + "=" * 70)
    logger.info("  SEPSIS — BiLSTM (input_dim=14, real eICU data)")
    logger.info("=" * 70)
    os.environ["TASK"] = "sepsis"
    os.environ["INPUT_DIM"] = "14"
    from models.hfl.bilstm.server_app import make_strategy
    from models.hfl.bilstm.client_app import client_fn

    cfg = TASK_CONFIG["sepsis"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(
            lambda n, nc: make_strategy(n, nc, 14), client_fn,
            s, cfg["metric"], nr, GPU_PER_CLIENT)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_ecg():
    logger.info("\n" + "=" * 70)
    logger.info("  ECG — BiLSTM (input_dim=12, synthetic 12-lead)")
    logger.info("=" * 70)
    os.environ["TASK"] = "ecg"
    os.environ["INPUT_DIM"] = "12"
    os.environ["MAX_SAMPLES"] = str(TASK_CONFIG["ecg"]["max_samples"])
    from models.hfl.bilstm.server_app import make_strategy
    from models.hfl.bilstm.client_app import client_fn

    cfg = TASK_CONFIG["ecg"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(
            lambda n, nc: make_strategy(n, nc, 12), client_fn,
            s, cfg["metric"], nr, GPU_PER_CLIENT)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_fraud():
    logger.info("\n" + "=" * 70)
    logger.info("  FRAUD — MLP (input_dim=30, 50K synthetic transactions)")
    logger.info("=" * 70)
    os.environ["MAX_SAMPLES"] = str(TASK_CONFIG["fraud"]["max_samples"])
    from models.hfl.mlp.server_app import make_strategy
    from models.hfl.mlp.client_app import client_fn

    cfg = TASK_CONFIG["fraud"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_chest():
    logger.info("\n" + "=" * 70)
    logger.info("  CHEST X-RAY — DenseNet-121 (14 pathologies, real NIH data)")
    logger.info("=" * 70)
    os.environ["SYNTHETIC"] = "0"
    from models.hfl.densenet.server_app import make_strategy
    from models.hfl.densenet.client_app import client_fn

    cfg = TASK_CONFIG["chest"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr, GPU_PER_CLIENT)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final auc={val:.4f} ({dt:.0f}s)")
    return results


def run_vfl():
    logger.info("\n" + "=" * 70)
    logger.info("  VFL FRAUD — Vertical FL (3 banks, 10 features each)")
    logger.info("=" * 70)
    from models.vfl.vfl_mlp.server_app import make_strategy
    from models.vfl.vfl_mlp.client_app import client_fn

    results = {}
    strats = ["IID", "FedProx_Mu0.1_Alpha_0.5", "SCAFFOLD_Alpha_0.5", "SecAgg_Alpha_0.5"]
    for s in strats:
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] 20 rounds, 3 clients (vertical)")
        val, hist, dt = run_fl(make_strategy, client_fn, s, "accuracy", 20, num_clients=3)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_split():
    logger.info("\n" + "=" * 70)
    logger.info("  SPLIT LEARNING — BiLSTM (LSTM private, classifier shared)")
    logger.info("=" * 70)
    os.environ["INPUT_DIM"] = "14"
    from models.vfl.split_bilstm.server_app import make_strategy
    from models.vfl.split_bilstm.client_app import client_fn

    results = {}
    strats = ["IID", "FedProx_Mu0.1_Alpha_0.5", "SCAFFOLD_Alpha_0.5"]
    for s in strats:
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] 30 rounds, 5 clients (split)")
        val, hist, dt = run_fl(make_strategy, client_fn, s, "accuracy", 30)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_transfer():
    logger.info("\n" + "=" * 70)
    logger.info("  TRANSFER LEARNING — DenseNet pretrained vs random init")
    logger.info("=" * 70)
    os.environ["SYNTHETIC"] = "1"
    from models.hfl.densenet.server_app import make_strategy
    from models.hfl.densenet.client_app import client_fn

    results = {}
    # Both use IID, compare pretrained=True (default) vs pretrained=False
    for mode in ["pretrained", "random_init"]:
        if mode == "random_init":
            os.environ["PRETRAINED"] = "0"
        else:
            os.environ.pop("PRETRAINED", None)
        lab = f"IID_{mode}"
        logger.info(f"\n  [{lab}] 10 rounds, 3 clients")
        val, hist, dt = run_fl(make_strategy, client_fn, "IID", "auc", 10, num_clients=3)
        results[mode] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final auc={val:.4f} ({dt:.0f}s)")
    return results


# ── New task runners ────────────────────────────────────────────────

def run_anomaly():
    logger.info("\n" + "=" * 70)
    logger.info("  ANOMALY — Autoencoder (input_dim=40, synthetic network traffic)")
    logger.info("=" * 70)
    from models.hfl.autoencoder.server_app import make_strategy
    from models.hfl.autoencoder.client_app import client_fn

    cfg = TASK_CONFIG["anomaly"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_mortality():
    logger.info("\n" + "=" * 70)
    logger.info("  MORTALITY — TabNet (input_dim=25, synthetic ICU data)")
    logger.info("=" * 70)
    from models.hfl.tabnet_simple.server_app import make_strategy
    from models.hfl.tabnet_simple.client_app import client_fn

    cfg = TASK_CONFIG["mortality"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_drug():
    logger.info("\n" + "=" * 70)
    logger.info("  DRUG — Generic MLP (input_dim=200, synthetic molecular fingerprints)")
    logger.info("=" * 70)
    os.environ["GENERIC_INPUT_DIM"] = "200"
    os.environ["GENERIC_NUM_CLASSES"] = "2"
    os.environ["GENERIC_TASK_TYPE"] = "binary"
    os.environ["GENERIC_MODEL"] = "mlp"
    os.environ["GENERIC_HIDDEN"] = "128"
    os.environ["GENERIC_DATA_MODULE"] = "tasks.drug.data"
    from models.hfl.generic.server_app import make_strategy
    from models.hfl.generic.client_app import client_fn

    cfg = TASK_CONFIG["drug"]
    os.environ["MAX_SAMPLES"] = str(cfg["max_samples"])
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_satellite():
    logger.info("\n" + "=" * 70)
    logger.info("  SATELLITE — ResNet-small (64x64x3, 5-class land use)")
    logger.info("=" * 70)
    from models.hfl.resnet_small.server_app import make_strategy
    from models.hfl.resnet_small.client_app import client_fn

    cfg = TASK_CONFIG["satellite"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr, GPU_PER_CLIENT)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_readmission():
    logger.info("\n" + "=" * 70)
    logger.info("  READMISSION — LogReg (input_dim=20, synthetic hospital data)")
    logger.info("=" * 70)
    from models.hfl.logreg.server_app import make_strategy
    from models.hfl.logreg.client_app import client_fn

    cfg = TASK_CONFIG["readmission"]
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final acc={val:.4f} ({dt:.0f}s)")
    return results


def run_olmo():
    logger.info("\n" + "=" * 70)
    logger.info("  OLMO — Federated LoRA (OLMo-1B, government documents)")
    logger.info("=" * 70)
    from models.llm.olmo.server_app import make_strategy
    from models.llm.olmo.client_app import client_fn

    cfg = TASK_CONFIG["olmo"]
    os.environ["MAX_SAMPLES"] = str(cfg["max_samples"])
    results = {}
    for s in cfg["strategies"]:
        nr = int(STRATEGIES[s][0] * cfg["rounds_mult"])
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {STRATEGIES[s][1]} — {nr} rounds")
        val, hist, dt = run_fl(make_strategy, client_fn, s, cfg["metric"], nr)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final perplexity={val:.2f} ({dt:.0f}s)")
    return results


def run_privacy():
    logger.info("\n" + "=" * 70)
    logger.info("  PRIVACY ATTACKS — DLG + MIA on BiLSTM + MLP")
    logger.info("=" * 70)
    from privacy.test_privacy import test_gradient_inversion, test_membership_inference, test_mia_mlp

    results = {}
    t0 = time.time()

    logger.info("\n  [DLG] Gradient Inversion on BiLSTM...")
    cos_no_dp, cos_dp = test_gradient_inversion()
    results["dlg_bilstm"] = {"no_dp": cos_no_dp, "dp": cos_dp}

    logger.info("\n  [MIA] Membership Inference on BiLSTM...")
    adv_no_dp, adv_dp = test_membership_inference()
    results["mia_bilstm"] = {"no_dp": adv_no_dp, "dp": adv_dp}

    logger.info("\n  [MIA] Membership Inference on MLP (Fraud)...")
    mlp_adv_no_dp, mlp_adv_dp = test_mia_mlp()
    results["mia_mlp"] = {"no_dp": mlp_adv_no_dp, "dp": mlp_adv_dp}

    results["time"] = time.time() - t0
    return results


# ── Main ─────────────────────────────────────────────────────────────

def save_results(all_results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"ec2_results_{ts}.json")

    # Convert numpy types for JSON serialization
    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    logger.info(f"\nResults saved to {path}")
    return path


def print_summary(all_results):
    logger.info("\n" + "=" * 70)
    logger.info("  EC2 PRODUCTION RUN — SUMMARY")
    logger.info("=" * 70)

    total_pass, total_fail = 0, 0

    for task, results in all_results.items():
        if task in ("meta", "privacy"):
            continue
        logger.info(f"\n  {task.upper()}:")
        if not isinstance(results, dict):
            continue
        for strat, data in results.items():
            if not isinstance(data, dict) or "value" not in data:
                continue
            val = data["value"]
            lab = data.get("label", strat)
            dt = data.get("time", 0)
            nr = len(data.get("history", []))
            ok = val > 0.01
            status = "PASS" if ok else "FAIL"
            logger.info(f"    {lab:<25s} {val:.4f}  ({nr}r, {dt:.0f}s)  {status}")
            if ok:
                total_pass += 1
            else:
                total_fail += 1

    if "privacy" in all_results:
        p = all_results["privacy"]
        logger.info(f"\n  PRIVACY:")
        if "dlg_bilstm" in p:
            d = p["dlg_bilstm"]
            logger.info(f"    DLG BiLSTM:  no_dp={d['no_dp']:.4f}  dp={d['dp']:.4f}")
        if "mia_bilstm" in p:
            d = p["mia_bilstm"]
            logger.info(f"    MIA BiLSTM:  no_dp={d['no_dp']:.4f}  dp={d['dp']:.4f}")
        if "mia_mlp" in p:
            d = p["mia_mlp"]
            logger.info(f"    MIA MLP:     no_dp={d['no_dp']:.4f}  dp={d['dp']:.4f}")

    logger.info(f"\n  TOTAL: {total_pass} passed, {total_fail} failed")
    logger.info("=" * 70)
    return total_fail


# ── YAML Scenario Support ────────────────────────────────────────────

TASK_RESOLVE = {
    "sepsis":       {"model": "bilstm",       "input_dim": 14, "metric": "accuracy"},
    "ecg":          {"model": "bilstm",       "input_dim": 12, "metric": "accuracy"},
    "fraud":        {"model": "mlp",          "input_dim": 30, "metric": "accuracy"},
    "chest":        {"model": "densenet",     "input_dim": 0,  "metric": "auc"},
    "vfl_fraud":    {"model": "vfl_mlp",      "input_dim": 10, "metric": "accuracy"},
    "split_sepsis": {"model": "split_bilstm", "input_dim": 14, "metric": "accuracy"},
}


def _get_make_strategy_and_client_fn(task, input_dim=None):
    """Resolve task name to (make_strategy_fn, client_fn)."""
    if task == "sepsis":
        os.environ["TASK"] = "sepsis"
        os.environ["INPUT_DIM"] = str(input_dim or 14)
        from models.hfl.bilstm.server_app import make_strategy
        from models.hfl.bilstm.client_app import client_fn
        idim = input_dim or 14
        return lambda n, nc: make_strategy(n, nc, idim), client_fn
    elif task == "ecg":
        os.environ["TASK"] = "ecg"
        os.environ["INPUT_DIM"] = str(input_dim or 12)
        from models.hfl.bilstm.server_app import make_strategy
        from models.hfl.bilstm.client_app import client_fn
        idim = input_dim or 12
        return lambda n, nc: make_strategy(n, nc, idim), client_fn
    elif task == "fraud":
        from models.hfl.mlp.server_app import make_strategy
        from models.hfl.mlp.client_app import client_fn
        return make_strategy, client_fn
    elif task == "chest":
        from models.hfl.densenet.server_app import make_strategy
        from models.hfl.densenet.client_app import client_fn
        return make_strategy, client_fn
    elif task == "vfl_fraud":
        from models.vfl.vfl_mlp.server_app import make_strategy
        from models.vfl.vfl_mlp.client_app import client_fn
        return make_strategy, client_fn
    elif task == "split_sepsis":
        os.environ["INPUT_DIM"] = str(input_dim or 14)
        from models.vfl.split_bilstm.server_app import make_strategy
        from models.vfl.split_bilstm.client_app import client_fn
        return make_strategy, client_fn
    else:
        raise ValueError(f"Unknown task: {task}")


def load_scenario(path):
    """Load a YAML scenario file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def run_scenario(scenario):
    """Run a YAML-defined scenario. Returns results dict."""
    name = scenario.get("name", "Unnamed Scenario")
    logger.info("\n" + "=" * 70)
    logger.info(f"  SCENARIO: {name}")
    logger.info(f"  {scenario.get('description', '')}")
    logger.info("=" * 70)

    # Handle attack-only scenarios
    if scenario.get("mode") == "attack":
        return run_scenario_attacks(scenario)

    # Handle multi-experiment scenarios
    if "experiments" in scenario:
        return run_scenario_multi(scenario)

    # Single-task scenario
    task = scenario["task"]
    strategies = scenario.get("strategies", ["IID"])
    num_rounds = scenario.get("num_rounds", 10)
    num_clients = scenario.get("num_clients", NC)
    max_samples = scenario.get("max_samples", 0)
    synthetic = scenario.get("synthetic", True)
    input_dim = scenario.get("input_dim")
    metric = TASK_RESOLVE.get(task, {}).get("metric", "accuracy")

    os.environ["SYNTHETIC"] = "1" if synthetic else "0"
    if max_samples:
        os.environ["MAX_SAMPLES"] = str(max_samples)

    make_strat, client_fn = _get_make_strategy_and_client_fn(task, input_dim)
    gpu = GPU_PER_CLIENT if task == "chest" else 0.0

    results = {}
    for s in strategies:
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        logger.info(f"\n  [{lab}] {num_rounds} rounds, {num_clients} clients")
        val, hist, dt = run_fl(make_strat, client_fn, s, metric, num_rounds, gpu, num_clients)
        results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
        logger.info(f"    Final {metric}={val:.4f} ({dt:.0f}s)")

    return results


def run_scenario_attacks(scenario):
    """Run attack-only scenario."""
    from privacy.test_privacy import test_gradient_inversion, test_membership_inference, test_mia_mlp
    attacks = scenario.get("attacks", ["dlg", "mia_bilstm", "mia_mlp"])
    results = {}
    t0 = time.time()

    if "dlg" in attacks:
        logger.info("\n  [DLG] Gradient Inversion on BiLSTM...")
        cos_no_dp, cos_dp = test_gradient_inversion()
        results["dlg_bilstm"] = {"no_dp": cos_no_dp, "dp": cos_dp}
    if "mia_bilstm" in attacks:
        logger.info("\n  [MIA] Membership Inference on BiLSTM...")
        adv_no_dp, adv_dp = test_membership_inference()
        results["mia_bilstm"] = {"no_dp": adv_no_dp, "dp": adv_dp}
    if "mia_mlp" in attacks:
        logger.info("\n  [MIA] Membership Inference on MLP...")
        mlp_no_dp, mlp_dp = test_mia_mlp()
        results["mia_mlp"] = {"no_dp": mlp_no_dp, "dp": mlp_dp}

    results["time"] = time.time() - t0
    return results


def run_scenario_multi(scenario):
    """Run multi-experiment scenario (different tasks in one YAML)."""
    experiments = scenario["experiments"]
    num_clients = scenario.get("num_clients", NC)
    num_rounds = scenario.get("num_rounds", 10)
    all_results = {}

    for exp in experiments:
        task = exp["task"]
        strategies = exp.get("strategies", ["IID"])
        nr = exp.get("num_rounds", num_rounds)
        ms = exp.get("max_samples", 0)
        syn = exp.get("synthetic", True)
        idim = exp.get("input_dim")
        metric = TASK_RESOLVE.get(task, {}).get("metric", "accuracy")

        os.environ["SYNTHETIC"] = "1" if syn else "0"
        if ms:
            os.environ["MAX_SAMPLES"] = str(ms)

        make_strat, client_fn = _get_make_strategy_and_client_fn(task, idim)
        gpu = GPU_PER_CLIENT if task == "chest" else 0.0

        task_results = {}
        for s in strategies:
            lab = s.split("_Alpha")[0].split("_Mu")[0]
            logger.info(f"\n  [{task}/{lab}] {nr} rounds")
            val, hist, dt = run_fl(make_strat, client_fn, s, metric, nr, gpu, num_clients)
            task_results[s] = {"label": lab, "value": val, "history": hist, "time": dt}
            logger.info(f"    Final {metric}={val:.4f} ({dt:.0f}s)")

        all_results[task] = task_results

    return all_results


def generate_summary_md(all_results, path):
    """Generate a markdown summary alongside the JSON results."""
    md_path = path.replace(".json", ".md")
    lines = ["# FL + PET Sandbox — Results Summary\n"]
    lines.append(f"**Generated:** {datetime.now().isoformat()}\n")

    meta = all_results.get("meta", {})
    if meta:
        lines.append(f"**Device:** {meta.get('device', 'N/A')}")
        lines.append(f"**Clients:** {meta.get('num_clients', 'N/A')}")
        lines.append(f"**Total time:** {meta.get('total_time', 0):.0f}s\n")

    for task, results in all_results.items():
        if task in ("meta", "privacy"):
            continue
        if not isinstance(results, dict):
            continue

        lines.append(f"\n## {task.upper()}\n")
        lines.append("| Strategy | Metric | Rounds | Time | Status |")
        lines.append("|----------|--------|--------|------|--------|")

        for strat, data in results.items():
            if not isinstance(data, dict) or "value" not in data:
                continue
            val = data["value"]
            lab = data.get("label", strat)
            nr = len(data.get("history", []))
            dt = data.get("time", 0)
            status = "PASS" if val > 0.01 else "FAIL"
            lines.append(f"| {lab} | {val:.4f} | {nr} | {dt:.0f}s | {status} |")

    if "privacy" in all_results:
        p = all_results["privacy"]
        lines.append("\n## PRIVACY ATTACKS\n")
        lines.append("| Attack | No DP | With DP | Protection |")
        lines.append("|--------|-------|---------|------------|")
        for key in ["dlg_bilstm", "mia_bilstm", "mia_mlp"]:
            if key in p:
                d = p[key]
                no_dp = d["no_dp"]
                dp = d["dp"]
                prot = "PROTECTED" if dp < no_dp * 0.8 else "partial"
                lines.append(f"| {key} | {no_dp:.4f} | {dp:.4f} | {prot} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Summary saved to {md_path}")
    return md_path


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()

    # Check if target is a YAML scenario
    is_scenario = target.endswith(".yaml") or target.endswith(".yml")
    if not is_scenario:
        # Check scenarios/ directory for named scenario
        scenario_path = os.path.join(ROOT, "scenarios", f"{target}.yaml")
        if os.path.exists(scenario_path):
            is_scenario = True
            target = scenario_path

    mode = "DISTRIBUTED (SuperLink + 5 SuperNodes)" if DISTRIBUTED else "SIMULATION (local)"
    logger.info("=" * 70)
    logger.info("  FL + PET SANDBOX")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Target: {target}")
    logger.info(f"  Clients: {NC}")
    if DISTRIBUTED:
        logger.info(f"  SuperLink: {SUPERLINK_ADDRESS}")
    logger.info(f"  GPU: {'L4 ' + str(GPU_PER_CLIENT) + '/client' if GPU_PER_CLIENT > 0 else 'CPU only'}")
    logger.info(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 70)

    all_results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "target": os.path.basename(target) if is_scenario else target,
            "num_clients": NC,
            "gpu": GPU_PER_CLIENT,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
    }

    if is_scenario:
        scenario = load_scenario(target)
        scenario_name = scenario.get("name", target)
        logger.info(f"  Scenario: {scenario_name}")
        result = run_scenario(scenario)

        # For attack scenarios, store under "privacy"
        if scenario.get("mode") == "attack":
            all_results["privacy"] = result
        # For multi-experiment, merge task results
        elif "experiments" in scenario:
            all_results.update(result)
        else:
            all_results[scenario.get("task", "experiment")] = result
    else:
        task_map = {
            "sepsis": run_sepsis,
            "ecg": run_ecg,
            "fraud": run_fraud,
            "chest": run_chest,
            "vfl": run_vfl,
            "split": run_split,
            "transfer": run_transfer,
            "anomaly": run_anomaly,
            "mortality": run_mortality,
            "drug": run_drug,
            "satellite": run_satellite,
            "readmission": run_readmission,
            "olmo": run_olmo,
            "privacy": run_privacy,
        }

        if target == "all":
            tasks = ["fraud", "sepsis", "ecg", "anomaly", "mortality", "drug",
                     "readmission", "satellite", "chest",
                     "vfl", "split", "transfer", "privacy"]
        else:
            tasks = [target]

        for t in tasks:
            if t in task_map:
                try:
                    all_results[t] = task_map[t]()
                except Exception as e:
                    logger.error(f"  TASK {t} FAILED: {e}")
                    all_results[t] = {"error": str(e)}

    all_results["meta"]["total_time"] = time.time() - t0

    path = save_results(all_results)
    generate_summary_md(all_results, path)
    total_fail = print_summary(all_results)
    logger.info(f"\n  Total time: {time.time() - t0:.0f}s")
    logger.info(f"  Results: {path}")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
