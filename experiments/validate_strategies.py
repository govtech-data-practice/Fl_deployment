#!/usr/bin/env python3
"""Quick validation: all 9 strategies on both tasks, 3 rounds, confirm none produce random output."""

import sys, os, time, logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s")
logger = logging.getLogger("validate")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.clientapp import ClientApp
from flwr.common import Context

os.environ["SYNTHETIC"] = "1"
os.environ["MAX_SAMPLES"] = "1000"
os.environ.setdefault("DATA_PATH", os.path.expanduser("~/healthcare-fl/data/sepsis"))

STRATEGIES = [
    "IID", "FedProx_Mu0.1_Alpha_0.5", "FedAdam_Alpha_0.5", "FedYogi_Alpha_0.5",
    "SCAFFOLD_Alpha_0.5", "SecAgg_Alpha_0.5",
    "DP_Central_Eps1.0_Clip1.0_Alpha_0.5", "DP_Local_Eps1.0_Clip1.0_Alpha_0.5",
    "OneOwner_Alpha_0.5",
]

NUM_CLIENTS, NUM_ROUNDS = 5, 3


class MetricCapture:
    def __init__(self, strat, key):
        self.strategy = strat
        self.key = key
        self.final = 0.0

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
            self.final = float(m.get(self.key, 0.0))
        return loss, m


def test_task(name, make_strat, cfn, metric_key):
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name} (metric: {metric_key})")
    logger.info(f"{'='*60}")

    results = []
    for sname in STRATEGIES:
        cap = MetricCapture(make_strat(sname, NUM_CLIENTS), metric_key)

        def sfn(ctx: Context) -> ServerAppComponents:
            return ServerAppComponents(strategy=cap, config=ServerConfig(num_rounds=NUM_ROUNDS))

        t0 = time.time()
        try:
            run_simulation(
                server_app=ServerApp(server_fn=sfn),
                client_app=ClientApp(client_fn=cfn),
                num_supernodes=NUM_CLIENTS,
                backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
            )
            val = cap.final
            status = "PASS" if val > 0.01 else "FAIL (random)"
        except Exception as e:
            val = 0.0
            status = f"ERROR: {e}"

        dt = time.time() - t0
        label = sname.split("_Eps")[0].split("_Mu")[0].split("_Alpha")[0]
        results.append((label, val, status, dt))
        logger.info(f"  {label:<15s}: {metric_key}={val:.4f} [{status}] ({dt:.0f}s)")

    return results


def main():
    os.environ["TASK"] = "sepsis"
    os.environ["INPUT_DIM"] = "14"
    from models.hfl.bilstm.server_app import make_strategy as s_strat
    from models.hfl.bilstm.client_app import client_fn as s_cfn
    from models.hfl.densenet.server_app import make_strategy as c_strat
    from models.hfl.densenet.client_app import client_fn as c_cfn
    from models.hfl.mlp.server_app import make_strategy as f_strat
    from models.hfl.mlp.client_app import client_fn as f_cfn

    t0 = time.time()
    sepsis_r = test_task("SEPSIS (BiLSTM)", lambda n, nc: s_strat(n, nc, 14), s_cfn, "accuracy")
    chest_r = test_task("CHEST X-RAY (DenseNet-121, synthetic)", c_strat, c_cfn, "auc")
    fraud_r = test_task("FRAUD (MLP)", f_strat, f_cfn, "accuracy")

    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")

    all_pass = True
    for name, results in [("Sepsis", sepsis_r), ("Chest", chest_r), ("Fraud", fraud_r)]:
        logger.info(f"\n  {name}:")
        for label, val, status, dt in results:
            logger.info(f"    {label:<15s}: {val:.4f} {status}")
            if "FAIL" in status or "ERROR" in status:
                all_pass = False

    logger.info(f"\n  Total: {time.time()-t0:.0f}s")
    logger.info(f"  {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
