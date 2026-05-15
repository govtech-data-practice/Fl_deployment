#!/usr/bin/env python3
"""
Full Data Benchmark: Sepsis with all 188 hospital files, no sample cap
=======================================================================
9 strategies × 3 heterogeneity levels × 20 rounds × 5 clients
Includes OneOwner strategy where only party 0 gets the final model.
"""

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

from sepsis.server_app import make_strategy
from sepsis.client_app import client_fn

NUM_CLIENTS = 5
NUM_ROUNDS = 20

ALPHAS = {"IID": None, "Moderate": 1.0, "Extreme": 0.1}

STRATEGIES = [
    "FedAvg",
    "FedProx_Mu0.1",
    "FedAdam",
    "FedYogi",
    "SCAFFOLD",
    "SecAgg",
    "DP_Central_Eps1.0_Clip1.0",
    "DP_Local_Eps1.0_Clip1.0",
    "OneOwner_Boost2.0",       # owner (client 0) gets 2x weight
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
                                  "accuracy": float(m.get("accuracy", 0.0))})
        return loss, m


def run_one(sname, aname, aval):
    if aname == "IID":
        full = sname if sname != "FedAvg" else "IID"
    else:
        full = f"{sname}_Alpha_{aval}"

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
            backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.0}},
        )
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return [], 0.0, 0.0, time.time() - t0

    final = cap.metrics[-1] if cap.metrics else {"accuracy": 0.0, "loss": 0.0}
    return cap.metrics, final["accuracy"], final["loss"], time.time() - t0


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, f"full_data_{ts}.csv")

    total = len(ALPHAS) * len(STRATEGIES)
    logger.info("=" * 70)
    logger.info("FULL DATA BENCHMARK — Sepsis (all 188 hospitals, no sample cap)")
    logger.info(f"  Clients: {NUM_CLIENTS}, Rounds: {NUM_ROUNDS}")
    logger.info(f"  Strategies: {len(STRATEGIES)} (including OneOwner)")
    logger.info(f"  Heterogeneity: {list(ALPHAS.keys())}")
    logger.info(f"  Total: {total} experiments")
    logger.info("=" * 70)

    # NO sample cap — use full data
    os.environ["MAX_SAMPLES"] = "0"
    os.environ.setdefault("DATA_PATH", os.path.expanduser("~/healthcare-fl/data/sepsis"))

    summary = {}
    n = 0
    t_start = time.time()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, ["strategy", "alpha_name", "alpha", "round", "loss", "accuracy"])
        w.writeheader()

        for aname, aval in ALPHAS.items():
            logger.info(f"\n{'#'*70}")
            logger.info(f"  {aname} (alpha={aval or 'IID'})")
            logger.info(f"{'#'*70}")

            for sname in STRATEGIES:
                n += 1
                label = sname.split("_Eps")[0].split("_Mu")[0].split("_Boost")[0]
                logger.info(f"\n  [{n}/{total}] {label} / {aname}")

                metrics, acc, loss, dt = run_one(sname, aname, aval)
                for m in metrics:
                    m["strategy"] = sname
                    m["alpha_name"] = aname
                    m["alpha"] = aval or 100.0
                    w.writerow(m)
                f.flush()

                summary[f"{aname}_{sname}"] = {"accuracy": acc, "loss": loss, "time": dt}
                logger.info(f"    acc={acc:.4f}, loss={loss:.4f}, {dt:.0f}s")

    # Results
    short = [s.split("_Eps")[0].split("_Mu")[0].split("_Boost")[0] for s in STRATEGIES]
    logger.info(f"\n{'='*80}")
    logger.info("RESULTS: Final Accuracy")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Alpha':<10s}" + "".join(f"{s:>12s}" for s in short))
    logger.info("  " + "-" * (10 + 12 * len(STRATEGIES)))
    for aname in ALPHAS:
        row = f"  {aname:<10s}"
        for sname in STRATEGIES:
            acc = summary.get(f"{aname}_{sname}", {}).get("accuracy", 0.0)
            row += f"{acc:>11.4f} "
        logger.info(row)

    logger.info(f"\n  RESULTS: Final Loss")
    logger.info(f"  {'Alpha':<10s}" + "".join(f"{s:>12s}" for s in short))
    logger.info("  " + "-" * (10 + 12 * len(STRATEGIES)))
    for aname in ALPHAS:
        row = f"  {aname:<10s}"
        for sname in STRATEGIES:
            loss = summary.get(f"{aname}_{sname}", {}).get("loss", 0.0)
            row += f"{loss:>11.4f} "
        logger.info(row)

    # OneOwner comparison
    logger.info(f"\n  OneOwner vs FedAvg:")
    for aname in ALPHAS:
        avg = summary.get(f"{aname}_FedAvg", {}).get("accuracy", 0.0)
        own = summary.get(f"{aname}_OneOwner_Boost2.0", {}).get("accuracy", 0.0)
        logger.info(f"    {aname}: FedAvg={avg:.4f}, OneOwner={own:.4f} (diff={own-avg:+.4f})")

    t_total = time.time() - t_start
    logger.info(f"\n  Total: {t_total:.0f}s ({t_total/60:.1f} min)")
    logger.info(f"  CSV: {csv_path}")

    with open(csv_path.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
