#!/usr/bin/env python3
"""
Non-IID Benchmark: How data heterogeneity affects FL, and which strategies help
================================================================================

Research question:
  When hospital data distributions differ (non-IID), how much does FL
  performance degrade, and which aggregation strategies recover it?

Experiment matrix (30 runs):
  5 heterogeneity levels × 6 strategies
  10 rounds each, 5 clients, 1000 samples

Heterogeneity levels (Dirichlet alpha):
  - IID (alpha=100): uniform distribution across clients
  - Mild (alpha=10): slight imbalance
  - Moderate (alpha=1.0): noticeable skew
  - Strong (alpha=0.5): significant skew
  - Extreme (alpha=0.1): most data concentrated on few clients

Strategies:
  - FedAvg: baseline weighted averaging
  - FedProx (mu=0.1): proximal regularization to prevent client drift
  - FedAdam: server-side adaptive learning rate
  - FedYogi: conservative adaptive optimizer (less aggressive than Adam)
  - SCAFFOLD: control variates to correct gradient drift
  - DP_Central (eps=1.0): differential privacy (privacy-utility tradeoff)

Outputs:
  - CSV with per-round metrics for all 30 experiments
  - Summary table: final accuracy per (strategy, alpha) combination
  - Analysis of which strategies recover non-IID degradation
"""

import sys
import os
import time
import json
import csv
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s | %(message)s",
)
logger = logging.getLogger("benchmark")

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.clientapp import ClientApp
from flwr.common import Context

from sepsis.server_app import BiLSTMSepsis, make_strategy
from sepsis.client_app import client_fn

# ======================================================================
# Experiment Configuration
# ======================================================================

NUM_CLIENTS = 5
NUM_ROUNDS = 10
MAX_SAMPLES = 1000  # per-client cap for speed

ALPHAS = {
    "IID":      None,   # uniform
    "Mild":     10.0,
    "Moderate": 1.0,
    "Strong":   0.5,
    "Extreme":  0.1,
}

STRATEGIES = [
    "FedAvg",
    "FedProx_Mu0.1",
    "FedAdam",
    "FedYogi",
    "SCAFFOLD",
    "DP_Central_Eps1.0_Clip1.0",
]

# Custom logging strategy wrapper to capture per-round metrics
class MetricCapture:
    """Wraps a strategy to capture per-round evaluation metrics."""

    def __init__(self, strategy, experiment_name):
        self.strategy = strategy
        self.exp_name = experiment_name
        self.round_metrics = []

    def __getattr__(self, name):
        return getattr(self.strategy, name)

    def configure_fit(self, *a, **kw):
        return self.strategy.configure_fit(*a, **kw)

    def configure_evaluate(self, *a, **kw):
        return self.strategy.configure_evaluate(*a, **kw)

    def aggregate_fit(self, *a, **kw):
        return self.strategy.aggregate_fit(*a, **kw)

    def aggregate_evaluate(self, *a, **kw):
        loss, metrics = self.strategy.aggregate_evaluate(*a, **kw)
        if loss is not None:
            rnd = kw.get("server_round", a[0] if a else 0)
            acc = metrics.get("accuracy", 0.0)
            self.round_metrics.append({
                "experiment": self.exp_name,
                "round": rnd,
                "loss": loss,
                "accuracy": acc,
            })
        return loss, metrics


def build_experiment_name(strategy, alpha_name):
    if alpha_name == "IID":
        return f"{strategy}_IID" if strategy != "FedAvg" else "IID"
    alpha = ALPHAS[alpha_name]
    return f"{strategy}_Alpha_{alpha}"


# ======================================================================
# Run Experiments
# ======================================================================

def run_single_experiment(strategy_name, alpha_name, alpha_val, results_writer, all_metrics):
    """Run one experiment and record results."""
    exp_name = build_experiment_name(strategy_name, alpha_name)

    # Build the full strategy name for the factory
    if alpha_name == "IID" and strategy_name == "FedAvg":
        full_name = "IID"
    elif alpha_name == "IID":
        full_name = strategy_name
    else:
        full_name = f"{strategy_name}_Alpha_{alpha_val}"

    logger.info(f"  Strategy: {full_name}")

    base_strategy = make_strategy(full_name, NUM_CLIENTS)
    capture = MetricCapture(base_strategy, exp_name)

    def server_fn(ctx: Context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy=capture,
            config=ServerConfig(num_rounds=NUM_ROUNDS),
        )

    t0 = time.time()
    try:
        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=NUM_CLIENTS,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
        )
        dt = time.time() - t0
        status = "OK"
    except Exception as e:
        dt = time.time() - t0
        status = f"FAIL: {e}"
        logger.error(f"  FAILED: {e}")

    # Record per-round metrics
    for m in capture.round_metrics:
        m["strategy"] = strategy_name
        m["alpha_name"] = alpha_name
        m["alpha"] = alpha_val if alpha_val else 100.0
        results_writer.writerow(m)
        all_metrics.append(m)

    # Final metrics
    final_acc = capture.round_metrics[-1]["accuracy"] if capture.round_metrics else 0.0
    final_loss = capture.round_metrics[-1]["loss"] if capture.round_metrics else 0.0

    logger.info(f"  Result: acc={final_acc:.4f}, loss={final_loss:.4f}, time={dt:.0f}s, status={status}")
    return final_acc, final_loss, dt


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"noniid_benchmark_{timestamp}.csv")
    summary_path = os.path.join(output_dir, f"noniid_summary_{timestamp}.json")

    logger.info("=" * 70)
    logger.info("NON-IID BENCHMARK: Heterogeneity × Strategy")
    logger.info("=" * 70)
    logger.info(f"  Clients: {NUM_CLIENTS}")
    logger.info(f"  Rounds: {NUM_ROUNDS}")
    logger.info(f"  Samples/client: {MAX_SAMPLES}")
    logger.info(f"  Heterogeneity levels: {list(ALPHAS.keys())}")
    logger.info(f"  Strategies: {STRATEGIES}")
    logger.info(f"  Total experiments: {len(ALPHAS) * len(STRATEGIES)}")
    logger.info(f"  Output: {csv_path}")
    logger.info("=" * 70)

    os.environ["MAX_SAMPLES"] = str(MAX_SAMPLES)
    os.environ["DATA_PATH"] = os.environ.get("DATA_PATH", "/data/flower_data")

    all_metrics = []
    summary = {}
    t_start = time.time()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "strategy", "alpha_name", "alpha",
            "round", "loss", "accuracy",
        ])
        writer.writeheader()

        exp_num = 0
        total_exps = len(ALPHAS) * len(STRATEGIES)

        for alpha_name, alpha_val in ALPHAS.items():
            logger.info(f"\n{'#' * 70}")
            logger.info(f"  HETEROGENEITY: {alpha_name} (alpha={alpha_val or 'IID'})")
            logger.info(f"{'#' * 70}")

            for strategy_name in STRATEGIES:
                exp_num += 1
                exp_label = build_experiment_name(strategy_name, alpha_name)
                logger.info(f"\n--- [{exp_num}/{total_exps}] {exp_label} ---")

                acc, loss, dt = run_single_experiment(
                    strategy_name, alpha_name, alpha_val, writer, all_metrics
                )
                f.flush()

                key = f"{alpha_name}_{strategy_name}"
                summary[key] = {
                    "alpha_name": alpha_name,
                    "alpha": alpha_val or 100.0,
                    "strategy": strategy_name,
                    "final_accuracy": acc,
                    "final_loss": loss,
                    "time_s": dt,
                }

    t_total = time.time() - t_start

    # Save summary
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ======== Results Table ========
    logger.info(f"\n{'=' * 70}")
    logger.info("RESULTS: Final Accuracy per (Heterogeneity × Strategy)")
    logger.info(f"{'=' * 70}")

    # Header
    strat_labels = [s.replace("_Mu0.1", "").replace("_Eps1.0_Clip1.0", "")
                    for s in STRATEGIES]
    header = f"  {'Alpha':<12s}" + "".join(f"{s:>12s}" for s in strat_labels)
    logger.info(header)
    logger.info("  " + "-" * (12 + 12 * len(STRATEGIES)))

    # IID baseline row
    for alpha_name in ALPHAS:
        row = f"  {alpha_name:<12s}"
        for strategy_name in STRATEGIES:
            key = f"{alpha_name}_{strategy_name}"
            acc = summary.get(key, {}).get("final_accuracy", 0.0)
            row += f"{acc:>11.4f} "
        logger.info(row)

    # Degradation analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("ANALYSIS: Accuracy degradation from IID baseline")
    logger.info(f"{'=' * 70}")

    for strategy_name in STRATEGIES:
        iid_key = f"IID_{strategy_name}"
        iid_acc = summary.get(iid_key, {}).get("final_accuracy", 0.0)

        label = strategy_name.replace("_Mu0.1", "").replace("_Eps1.0_Clip1.0", "")
        logger.info(f"\n  {label} (IID baseline: {iid_acc:.4f}):")

        for alpha_name in ["Mild", "Moderate", "Strong", "Extreme"]:
            key = f"{alpha_name}_{strategy_name}"
            acc = summary.get(key, {}).get("final_accuracy", 0.0)
            drop = iid_acc - acc
            pct = (drop / iid_acc * 100) if iid_acc > 0 else 0
            logger.info(f"    {alpha_name:<12s}: {acc:.4f} (drop: {drop:+.4f}, {pct:+.1f}%)")

    # Best strategy per heterogeneity level
    logger.info(f"\n{'=' * 70}")
    logger.info("BEST STRATEGY per heterogeneity level")
    logger.info(f"{'=' * 70}")
    for alpha_name in ALPHAS:
        best_key = None
        best_acc = -1
        for strategy_name in STRATEGIES:
            key = f"{alpha_name}_{strategy_name}"
            acc = summary.get(key, {}).get("final_accuracy", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_key = strategy_name
        label = best_key.replace("_Mu0.1", "").replace("_Eps1.0_Clip1.0", "")
        logger.info(f"  {alpha_name:<12s}: {label} ({best_acc:.4f})")

    logger.info(f"\n  Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
    logger.info(f"  Results: {csv_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
