#!/usr/bin/env python3
"""Chest X-ray only benchmark — 8 strategies × 5 alphas × 15 rounds."""

import sys, os, time, json, csv, logging, numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s")
logger = logging.getLogger("bench")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for pkg in ["sepsis", "chest_xray"]:
    init = os.path.join(ROOT, pkg, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()

import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.clientapp import ClientApp
from flwr.common import Context

os.environ["SYNTHETIC"] = "1"

from chest_xray.server_app import make_strategy
from chest_xray.client_app import client_fn

NUM_CLIENTS = 5
NUM_ROUNDS = 15

ALPHAS = {"IID": None, "Mild": 10.0, "Moderate": 1.0, "Strong": 0.5, "Extreme": 0.1}
STRATEGIES = [
    "FedAvg", "FedProx_Mu0.1", "FedAdam", "FedYogi",
    "SCAFFOLD", "SecAgg",
    "DP_Central_Eps1.0_Clip1.0", "DP_Local_Eps1.0_Clip1.0",
]


class MetricCapture:
    def __init__(self, strategy, name):
        self.strategy = strategy
        self.name = name
        self.metrics = []

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
            rnd = kw.get("server_round", a[0] if a else 0)
            self.metrics.append({"round": rnd, "loss": float(loss),
                                  "auc": float(m.get("auc", 0.0))})
        return loss, m


def run_one(strat_name, alpha_name, alpha_val):
    if alpha_name == "IID":
        full = strat_name if strat_name != "FedAvg" else "IID"
    else:
        full = f"{strat_name}_Alpha_{alpha_val}"

    base = make_strategy(full, NUM_CLIENTS)
    cap = MetricCapture(base, full)

    def sfn(ctx: Context) -> ServerAppComponents:
        return ServerAppComponents(strategy=cap, config=ServerConfig(num_rounds=NUM_ROUNDS))

    t0 = time.time()
    try:
        run_simulation(
            server_app=ServerApp(server_fn=sfn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=NUM_CLIENTS,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
        )
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return [], 0.0, 0.0, time.time() - t0

    final = cap.metrics[-1] if cap.metrics else {"auc": 0.0, "loss": 0.0}
    return cap.metrics, final["auc"], final["loss"], time.time() - t0


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"chest_benchmark_{ts}.csv")

    total = len(ALPHAS) * len(STRATEGIES)
    logger.info(f"CHEST X-RAY BENCHMARK: {total} experiments, {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients")

    summary = {}
    n = 0
    t0 = time.time()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["strategy", "alpha_name", "alpha", "round", "loss", "auc"])
        w.writeheader()

        for aname, aval in ALPHAS.items():
            for sname in STRATEGIES:
                n += 1
                label = sname.split("_Eps")[0].split("_Mu")[0]
                logger.info(f"\n[{n}/{total}] {label} / {aname}")

                metrics, auc, loss, dt = run_one(sname, aname, aval)
                for m in metrics:
                    m["strategy"] = sname
                    m["alpha_name"] = aname
                    m["alpha"] = aval or 100.0
                    w.writerow(m)
                f.flush()

                summary[f"{aname}_{sname}"] = {"auc": auc, "loss": loss}
                logger.info(f"  AUC={auc:.4f}, loss={loss:.4f}, {dt:.0f}s")

    # Results table
    short = [s.split("_Eps")[0].split("_Mu")[0] for s in STRATEGIES]
    logger.info(f"\n{'='*80}")
    logger.info(f"CHEST X-RAY: Final AUC per (Alpha × Strategy)")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Alpha':<10s}" + "".join(f"{s:>12s}" for s in short))
    logger.info("  " + "-" * (10 + 12 * len(STRATEGIES)))
    for aname in ALPHAS:
        row = f"  {aname:<10s}"
        for sname in STRATEGIES:
            auc = summary.get(f"{aname}_{sname}", {}).get("auc", 0.0)
            row += f"{auc:>11.4f} "
        logger.info(row)

    # Degradation
    logger.info(f"\n  IID → Extreme degradation:")
    for sname in STRATEGIES:
        iid = summary.get(f"IID_{sname}", {}).get("auc", 0.0)
        ext = summary.get(f"Extreme_{sname}", {}).get("auc", 0.0)
        drop = (iid - ext) / iid * 100 if iid > 0 else 0
        label = sname.split("_Eps")[0].split("_Mu")[0]
        logger.info(f"    {label:<15s}: {iid:.4f} → {ext:.4f} ({drop:+.1f}%)")

    logger.info(f"\n  Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    logger.info(f"  CSV: {csv_path}")

    with open(csv_path.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
