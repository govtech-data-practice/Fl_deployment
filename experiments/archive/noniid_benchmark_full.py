#!/usr/bin/env python3
"""
Full Non-IID Benchmark: Sepsis (accuracy) + Chest X-ray (AUC)
==============================================================
Tests all 8 strategies × 5 heterogeneity levels × 2 tasks = 80 experiments.
Chest X-ray uses synthetic data with multi-label AUC for better differentiation.

Strategies: FedAvg, FedProx, FedAdam, FedYogi, SCAFFOLD, SecAgg+, DP_Central, DP_Local
"""

import sys
import os
import time
import json
import csv
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s")
logger = logging.getLogger("benchmark")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Ensure each task can find its own partition_utils via package import
# by adding __init__.py if missing
for pkg in ["sepsis", "chest_xray"]:
    init = os.path.join(ROOT, pkg, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()

import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.clientapp import ClientApp
from flwr.common import Context

NUM_CLIENTS = 5
NUM_ROUNDS = 15
MAX_SAMPLES = 1000

ALPHAS = {
    "IID":      None,
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
    "SecAgg",
    "DP_Central_Eps1.0_Clip1.0",
    "DP_Local_Eps1.0_Clip1.0",
]


class MetricCapture:
    def __init__(self, strategy, exp_name):
        self.strategy = strategy
        self.exp_name = exp_name
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
            self.round_metrics.append({
                "experiment": self.exp_name,
                "round": rnd,
                "loss": float(loss),
                **{k: float(v) for k, v in metrics.items()},
            })
        return loss, metrics


def build_name(strategy, alpha_name, alpha_val):
    if alpha_name == "IID":
        return strategy if strategy != "FedAvg" else "IID"
    return f"{strategy}_Alpha_{alpha_val}"


def run_experiment(task, strategy_name, alpha_name, alpha_val, make_strat_fn, client_fn, metric_name):
    full_name = build_name(strategy_name, alpha_name, alpha_val)
    exp_label = f"{task}_{strategy_name}_{alpha_name}"

    base = make_strat_fn(full_name, NUM_CLIENTS)
    capture = MetricCapture(base, exp_label)

    def server_fn(ctx: Context) -> ServerAppComponents:
        return ServerAppComponents(strategy=capture, config=ServerConfig(num_rounds=NUM_ROUNDS))

    t0 = time.time()
    try:
        run_simulation(
            server_app=ServerApp(server_fn=server_fn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=NUM_CLIENTS,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
        )
        dt = time.time() - t0
    except Exception as e:
        dt = time.time() - t0
        logger.error(f"  FAILED: {e}")
        return [], 0.0, 0.0, dt

    final = capture.round_metrics[-1] if capture.round_metrics else {}
    final_metric = final.get(metric_name, 0.0)
    final_loss = final.get("loss", 0.0)

    for m in capture.round_metrics:
        m["task"] = task
        m["strategy"] = strategy_name
        m["alpha_name"] = alpha_name
        m["alpha"] = alpha_val or 100.0

    return capture.round_metrics, final_metric, final_loss, dt


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"full_benchmark_{timestamp}.csv")
    summary_path = os.path.join(out_dir, f"full_summary_{timestamp}.json")

    os.environ["MAX_SAMPLES"] = str(MAX_SAMPLES)
    os.environ.setdefault("DATA_PATH", os.path.expanduser("~/healthcare-fl/data/sepsis"))
    os.environ["SYNTHETIC"] = "1"

    from sepsis.server_app import make_strategy as sepsis_strat
    from sepsis.client_app import client_fn as sepsis_cfn
    from chest_xray.server_app import make_strategy as chest_strat
    from chest_xray.client_app import client_fn as chest_cfn

    TASKS = [
        ("sepsis", "accuracy", sepsis_strat, sepsis_cfn),
        ("chest_xray", "auc", chest_strat, chest_cfn),
    ]

    total_exps = len(TASKS) * len(ALPHAS) * len(STRATEGIES)
    logger.info("=" * 70)
    logger.info("FULL NON-IID BENCHMARK")
    logger.info(f"  Tasks: sepsis (accuracy), chest_xray (AUC)")
    logger.info(f"  Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}")
    logger.info(f"  Alphas: {list(ALPHAS.keys())}")
    logger.info(f"  Strategies: {[s.split('_Eps')[0].split('_Mu')[0] for s in STRATEGIES]}")
    logger.info(f"  Total: {total_exps} experiments")
    logger.info("=" * 70)

    all_rows = []
    summary = {}
    exp_num = 0
    t_start = time.time()

    csv_fields = [
        "experiment", "task", "strategy", "alpha_name", "alpha",
        "round", "loss", "accuracy", "auc",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()

        for task, metric_name, make_strat, cfn in TASKS:
            logger.info(f"\n{'#' * 70}")
            logger.info(f"  TASK: {task.upper()} (metric: {metric_name})")
            logger.info(f"{'#' * 70}")

            for alpha_name, alpha_val in ALPHAS.items():
                logger.info(f"\n  --- {alpha_name} (alpha={alpha_val or 'IID'}) ---")

                for strat_name in STRATEGIES:
                    exp_num += 1
                    label = f"{task}/{strat_name}/{alpha_name}"
                    logger.info(f"\n  [{exp_num}/{total_exps}] {label}")

                    metrics, final_val, final_loss, dt = run_experiment(
                        task, strat_name, alpha_name, alpha_val, make_strat, cfn, metric_name
                    )

                    for m in metrics:
                        writer.writerow(m)
                    f.flush()
                    all_rows.extend(metrics)

                    key = f"{task}_{alpha_name}_{strat_name}"
                    summary[key] = {
                        "task": task, "metric": metric_name,
                        "alpha_name": alpha_name, "alpha": alpha_val or 100.0,
                        "strategy": strat_name,
                        "final_value": final_val, "final_loss": final_loss,
                        "time_s": dt,
                    }
                    logger.info(f"    {metric_name}={final_val:.4f}, loss={final_loss:.4f}, {dt:.0f}s")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    t_total = time.time() - t_start

    # ======== Results Tables ========
    strat_short = [s.split("_Eps")[0].split("_Mu")[0].replace("_Clip1.0", "") for s in STRATEGIES]

    for task, metric_name, _, _ in TASKS:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  {task.upper()}: Final {metric_name} per (Alpha × Strategy)")
        logger.info(f"{'=' * 70}")

        header = f"  {'Alpha':<10s}" + "".join(f"{s:>12s}" for s in strat_short)
        logger.info(header)
        logger.info("  " + "-" * (10 + 12 * len(STRATEGIES)))

        for alpha_name in ALPHAS:
            row = f"  {alpha_name:<10s}"
            for strat_name in STRATEGIES:
                key = f"{task}_{alpha_name}_{strat_name}"
                val = summary.get(key, {}).get("final_value", 0.0)
                row += f"{val:>11.4f} "
            logger.info(row)

        # Best per alpha
        logger.info(f"\n  Best strategy per heterogeneity:")
        for alpha_name in ALPHAS:
            best_s, best_v = "", -1
            for strat_name in STRATEGIES:
                key = f"{task}_{alpha_name}_{strat_name}"
                v = summary.get(key, {}).get("final_value", 0.0)
                if v > best_v:
                    best_v = v
                    best_s = strat_name
            label = best_s.split("_Eps")[0].split("_Mu")[0]
            logger.info(f"    {alpha_name:<10s}: {label} ({best_v:.4f})")

    # Degradation analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("DEGRADATION: % drop from IID to Extreme per strategy")
    logger.info(f"{'=' * 70}")
    for task, _, _, _ in TASKS:
        logger.info(f"\n  {task}:")
        for strat_name in STRATEGIES:
            iid_key = f"{task}_IID_{strat_name}"
            ext_key = f"{task}_Extreme_{strat_name}"
            iid_v = summary.get(iid_key, {}).get("final_value", 0.0)
            ext_v = summary.get(ext_key, {}).get("final_value", 0.0)
            drop = iid_v - ext_v
            pct = (drop / iid_v * 100) if iid_v > 0 else 0
            label = strat_name.split("_Eps")[0].split("_Mu")[0]
            logger.info(f"    {label:<15s}: IID={iid_v:.4f} → Extreme={ext_v:.4f} (drop: {pct:+.1f}%)")

    logger.info(f"\n  Total time: {t_total:.0f}s ({t_total/60:.1f} min)")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
