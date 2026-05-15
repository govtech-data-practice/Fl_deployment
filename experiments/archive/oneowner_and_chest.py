#!/usr/bin/env python3
"""
Part 1: OneOwner fix — rerun on full sepsis data (3 alpha levels)
Part 2: Chest X-ray on REAL data (all strategies, 3 alpha levels, 10 rounds)
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

NUM_CLIENTS = 5
ALPHAS = {"IID": None, "Moderate": 1.0, "Extreme": 0.1}


class MetricCapture:
    def __init__(self, strategy, name, metric_key):
        self.strategy = strategy
        self.name = name
        self.metric_key = metric_key
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
            self.metrics.append({
                "round": rnd, "loss": float(loss),
                self.metric_key: float(m.get(self.metric_key, 0.0)),
            })
        return loss, m


def run_one(make_strat, client_fn, sname, aname, aval, num_rounds, metric_key):
    if aname == "IID":
        full = sname if sname != "FedAvg" else "IID"
    else:
        full = f"{sname}_Alpha_{aval}"

    base = make_strat(full, NUM_CLIENTS)
    cap = MetricCapture(base, full, metric_key)

    def sfn(ctx: Context) -> ServerAppComponents:
        return ServerAppComponents(strategy=cap, config=ServerConfig(num_rounds=num_rounds))

    t0 = time.time()
    try:
        run_simulation(
            server_app=ServerApp(server_fn=sfn),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=NUM_CLIENTS,
            backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.2}},
        )
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return 0.0, 0.0, time.time() - t0

    final = cap.metrics[-1] if cap.metrics else {metric_key: 0.0, "loss": 0.0}
    return final[metric_key], final["loss"], time.time() - t0


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    results = {}

    # ==========================================
    # Part 1: OneOwner on full sepsis data
    # ==========================================
    logger.info("=" * 70)
    logger.info("PART 1: OneOwner strategy on full sepsis data")
    logger.info("=" * 70)

    os.environ["MAX_SAMPLES"] = "0"
    os.environ["DATA_PATH"] = os.path.expanduser("~/healthcare-fl/data/sepsis")

    from sepsis.server_app import make_strategy as sepsis_strat
    from sepsis.client_app import client_fn as sepsis_cfn

    for aname, aval in ALPHAS.items():
        logger.info(f"\n  OneOwner / {aname}")
        acc, loss, dt = run_one(sepsis_strat, sepsis_cfn, "OneOwner_Boost2.0", aname, aval, 20, "accuracy")
        results[f"sepsis_{aname}_OneOwner"] = {"accuracy": acc, "loss": loss}
        logger.info(f"    acc={acc:.4f}, loss={loss:.4f}, {dt:.0f}s")

    # ==========================================
    # Part 2: Chest X-ray on REAL data
    # ==========================================
    logger.info(f"\n{'=' * 70}")
    logger.info("PART 2: Chest X-ray on REAL NIH data")
    logger.info("=" * 70)

    os.environ["SYNTHETIC"] = "0"
    os.environ["DATASET_PATH"] = os.path.expanduser("~/healthcare-fl/data/chest_xray_real/archive")
    os.environ["CSV_PATH"] = os.path.expanduser("~/healthcare-fl/data/chest_xray_real/archive/Data_Entry_2017.csv")

    from chest_xray.server_app import make_strategy as chest_strat
    from chest_xray.client_app import client_fn as chest_cfn

    CHEST_STRATEGIES = [
        "FedAvg", "FedProx_Mu0.1", "FedAdam", "FedYogi",
        "SCAFFOLD", "SecAgg", "OneOwner_Boost2.0",
    ]

    for aname, aval in ALPHAS.items():
        logger.info(f"\n  --- {aname} ---")
        for sname in CHEST_STRATEGIES:
            label = sname.split("_Mu")[0].split("_Boost")[0]
            logger.info(f"\n  {label} / {aname}")
            auc, loss, dt = run_one(chest_strat, chest_cfn, sname, aname, aval, 10, "auc")
            results[f"chest_{aname}_{sname}"] = {"auc": auc, "loss": loss}
            logger.info(f"    AUC={auc:.4f}, loss={loss:.4f}, {dt:.0f}s")

    # ==========================================
    # Report
    # ==========================================
    logger.info(f"\n{'=' * 70}")
    logger.info("COMBINED RESULTS")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n  Sepsis OneOwner (accuracy):")
    for aname in ALPHAS:
        r = results.get(f"sepsis_{aname}_OneOwner", {})
        logger.info(f"    {aname}: acc={r.get('accuracy', 0):.4f}")

    logger.info(f"\n  Chest X-ray (AUC, real data):")
    short = [s.split("_Mu")[0].split("_Boost")[0] for s in CHEST_STRATEGIES]
    logger.info(f"  {'Alpha':<10s}" + "".join(f"{s:>12s}" for s in short))
    for aname in ALPHAS:
        row = f"  {aname:<10s}"
        for sname in CHEST_STRATEGIES:
            auc = results.get(f"chest_{aname}_{sname}", {}).get("auc", 0.0)
            row += f"{auc:>11.4f} "
        logger.info(row)

    t_total = time.time() - t_start
    logger.info(f"\n  Total: {t_total:.0f}s ({t_total/60:.1f} min)")

    with open(os.path.join(out, f"combined_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved: {out}/combined_{ts}.json")


if __name__ == "__main__":
    main()
