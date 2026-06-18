#!/usr/bin/env python3
"""
Unified test runner: runs all strategies for sepsis, chest_xray, and fraud.
Usage: python run_tests.py [sepsis|chest|fraud|all]
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger("test")

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.clientapp import ClientApp
from flwr.common import Context

STRATEGIES = [
    "IID",
    "FedProx_Mu0.1_Alpha_0.5",
    "FedAdam_Alpha_0.5",
    "FedYogi_Alpha_0.5",
    "SCAFFOLD_Alpha_0.5",
    "SecAgg_Alpha_0.5",
    "DP_Central_Eps1.0_Clip1.0",
    "DP_Local_Eps1.0_Clip1.0",
]
NUM_CLIENTS = 2


def run_suite(name, make_strategy_fn, client_fn, num_rounds):
    logger.info(f"\n{'#' * 60}")
    logger.info(f"  {name.upper()} — {len(STRATEGIES)} strategies, {num_rounds} rounds")
    logger.info(f"{'#' * 60}")

    results = []
    for i, strat in enumerate(STRATEGIES, 1):
        logger.info(f"\n  [{i}/{len(STRATEGIES)}] {strat}")
        t0 = time.time()
        try:
            def make_sfn(s):
                def sfn(ctx: Context) -> ServerAppComponents:
                    return ServerAppComponents(
                        strategy=make_strategy_fn(s, NUM_CLIENTS),
                        config=ServerConfig(num_rounds=num_rounds),
                    )
                return sfn

            run_simulation(
                server_app=ServerApp(server_fn=make_sfn(strat)),
                client_app=ClientApp(client_fn=client_fn),
                num_supernodes=NUM_CLIENTS,
                backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
            )
            dt = time.time() - t0
            results.append((strat, "PASS", dt))
            logger.info(f"  PASS ({dt:.1f}s)")
        except Exception as e:
            dt = time.time() - t0
            results.append((strat, "FAIL", dt))
            logger.error(f"  FAIL ({dt:.1f}s): {e}")

    return results


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    all_results = {}

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    if target in ("sepsis", "all"):
        os.environ["TASK"] = "sepsis"
        os.environ["INPUT_DIM"] = "14"
        from models.hfl.bilstm.server_app import make_strategy as sepsis_strat
        from models.hfl.bilstm.client_app import client_fn as sepsis_cfn
        all_results["sepsis"] = run_suite("Sepsis (BiLSTM)",
            lambda s, nc: sepsis_strat(s, nc, 14), sepsis_cfn, num_rounds=3)

    if target in ("chest", "all"):
        os.environ.setdefault("SYNTHETIC", "1")
        from models.hfl.densenet.server_app import make_strategy as chest_strat
        from models.hfl.densenet.client_app import client_fn as chest_cfn
        all_results["chest_xray"] = run_suite("Chest X-ray (DenseNet-121)", chest_strat, chest_cfn, num_rounds=2)

    if target in ("fraud", "all"):
        from models.hfl.mlp.server_app import make_strategy as fraud_strat
        from models.hfl.mlp.client_app import client_fn as fraud_cfn
        all_results["fraud"] = run_suite("Fraud (MLP)", fraud_strat, fraud_cfn, num_rounds=3)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 60}")
    total_pass, total_fail = 0, 0
    for suite, results in all_results.items():
        p = sum(1 for _, s, _ in results if s == "PASS")
        total_pass += p
        total_fail += len(results) - p
        logger.info(f"\n  {suite}: {p}/{len(results)} passed")
        for strat, status, dt in results:
            logger.info(f"    {status}  {strat:<30s} {dt:.1f}s")

    logger.info(f"\n  TOTAL: {total_pass}/{total_pass + total_fail} passed")
    logger.info(f"{'=' * 60}")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
