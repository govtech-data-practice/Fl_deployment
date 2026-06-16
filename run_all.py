#!/usr/bin/env python3
"""
Unified test: every model x task combination with multiple strategies.
  BiLSTM x sepsis    (tabular time series, 14 features)
  BiLSTM x ecg       (12-lead ECG, 12 features)
  MLP x fraud        (tabular, 30 features)
  DenseNet x chest   (2D images, synthetic mode)
  Mistral x gov_llm  (LLM, skipped if no GPU)
"""
import sys, os, time, logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s")
logger = logging.getLogger("run_all")

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.clientapp import ClientApp
from flwr.common import Context

NC = 5
NR = 3
STRATS = ["IID", "FedProx_Mu0.1_Alpha_0.5", "SCAFFOLD_Alpha_0.5", "SecAgg_Alpha_0.5",
          "DP_Central_Eps50.0_Alpha_0.5"]


class Cap:
    def __init__(s, st, key):
        s.strategy = st; s.key = key; s.val = 0.0
    def __getattr__(s, n):
        return getattr(s.strategy, n)
    def configure_fit(s, *a, **kw):
        return s.strategy.configure_fit(*a, **kw)
    def configure_evaluate(s, *a, **kw):
        return s.strategy.configure_evaluate(*a, **kw)
    def aggregate_fit(s, *a, **kw):
        return s.strategy.aggregate_fit(*a, **kw)
    def aggregate_evaluate(s, *a, **kw):
        l, m = s.strategy.aggregate_evaluate(*a, **kw)
        if l is not None:
            s.val = float(m.get(s.key, 0.0))
        return l, m


def run_fl(make_strat, client_fn, strat_name, metric_key, gpu=0.0):
    c = Cap(make_strat(strat_name, NC), metric_key)

    def sf(ctx):
        return ServerAppComponents(strategy=c, config=ServerConfig(num_rounds=NR))

    t0 = time.time()
    try:
        run_simulation(
            server_app=ServerApp(server_fn=sf),
            client_app=ClientApp(client_fn=client_fn),
            num_supernodes=NC,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": gpu}},
        )
        return c.val, time.time() - t0
    except Exception as e:
        logger.error("  ERROR: %s" % e)
        return 0.0, time.time() - t0


def test_bilstm_sepsis():
    logger.info("\n--- BiLSTM x Sepsis (input_dim=14) ---")
    os.environ["TASK"] = "sepsis"
    os.environ["INPUT_DIM"] = "14"
    from models.hfl.bilstm.server_app import make_strategy
    from models.hfl.bilstm.client_app import client_fn
    results = {}
    for s in STRATS:
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        val, dt = run_fl(lambda n, nc: make_strategy(n, nc, 14), client_fn, s, "accuracy")
        results[lab] = val
        logger.info("  %-15s acc=%.4f (%ds)" % (lab, val, dt))
    return results


def test_bilstm_ecg():
    logger.info("\n--- BiLSTM x ECG (input_dim=12) ---")
    os.environ["TASK"] = "ecg"
    os.environ["INPUT_DIM"] = "12"
    from models.hfl.bilstm.server_app import make_strategy
    from models.hfl.bilstm.client_app import client_fn
    results = {}
    for s in STRATS:
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        val, dt = run_fl(lambda n, nc: make_strategy(n, nc, 12), client_fn, s, "accuracy")
        results[lab] = val
        logger.info("  %-15s acc=%.4f (%ds)" % (lab, val, dt))
    return results


def test_mlp_fraud():
    logger.info("\n--- MLP x Fraud (input_dim=30) ---")
    from models.hfl.mlp.server_app import make_strategy
    from models.hfl.mlp.client_app import client_fn
    results = {}
    for s in STRATS:
        lab = s.split("_Alpha")[0].split("_Mu")[0]
        val, dt = run_fl(make_strategy, client_fn, s, "accuracy")
        results[lab] = val
        logger.info("  %-15s acc=%.4f (%ds)" % (lab, val, dt))
    return results


def test_densenet_chest():
    logger.info("\n--- DenseNet x Chest X-ray (synthetic) ---")
    os.environ["SYNTHETIC"] = "1"
    from models.hfl.densenet.server_app import make_strategy
    from models.hfl.densenet.client_app import client_fn
    # Only test FedAvg + SCAFFOLD (DenseNet is slow)
    short_strats = ["IID", "SCAFFOLD_Alpha_0.5"]
    results = {}
    for s in short_strats:
        lab = s.split("_Alpha")[0]
        val, dt = run_fl(make_strategy, client_fn, s, "auc", gpu=0.0)
        results[lab] = val
        logger.info("  %-15s auc=%.4f (%ds)" % (lab, val, dt))
    return results


def test_mistral_gov():
    if not torch.cuda.is_available():
        logger.info("\n--- Mistral x Gov LLM: SKIPPED (no GPU) ---")
        return {"skipped": True}

    logger.info("\n--- Mistral x Gov LLM (QLoRA, 3 agencies) ---")
    from tasks.llm.gov_llm.data import get_all_agency_data, generate_nonmember_data
    agency_data = get_all_agency_data(num_notes_per_agency=100)
    nonmember = generate_nonmember_data(50)
    logger.info("  Data: 3 agencies x 100 notes + 50 nonmember")
    # Quick test: just verify data generation works
    logger.info("  Tax sample: %s..." % agency_data[0][0][:80])
    logger.info("  Immigration sample: %s..." % agency_data[1][0][:80])
    logger.info("  Health sample: %s..." % agency_data[2][0][:80])
    return {"data_generated": True, "agencies": 3, "notes_per_agency": 100}


def main():
    logger.info("=" * 60)
    logger.info("RUN ALL: Model x Task Combinations")
    logger.info("=" * 60)
    t0 = time.time()

    all_results = {}
    all_results["bilstm_sepsis"] = test_bilstm_sepsis()
    all_results["bilstm_ecg"] = test_bilstm_ecg()
    all_results["mlp_fraud"] = test_mlp_fraud()
    all_results["densenet_chest"] = test_densenet_chest()
    all_results["mistral_gov"] = test_mistral_gov()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    total_pass, total_fail = 0, 0
    for combo, results in all_results.items():
        logger.info("\n  %s:" % combo)
        if isinstance(results, dict) and "skipped" in results:
            logger.info("    SKIPPED")
            continue
        if isinstance(results, dict) and "data_generated" in results:
            logger.info("    Data generated OK")
            total_pass += 1
            continue
        for strat, val in results.items():
            ok = val > 0.01
            logger.info("    %-15s %.4f %s" % (strat, val, "PASS" if ok else "FAIL"))
            if ok:
                total_pass += 1
            else:
                total_fail += 1

    logger.info("\n  %d passed, %d failed, %.0fs total" % (total_pass, total_fail, time.time() - t0))
    logger.info("=" * 60)
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
