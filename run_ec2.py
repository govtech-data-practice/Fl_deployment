#!/usr/bin/env python3
"""
FL + PET Sandbox — EC2 Runner
==============================
Run federated learning experiments from CLI or YAML scenarios.

Usage:
  python run_ec2.py                           # all tasks (production)
  python run_ec2.py sepsis                    # single task
  python run_ec2.py scenarios/quick_fraud.yaml  # YAML scenario
  python run_ec2.py privacy                   # privacy attack suite
"""

import sys, os, time, json, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger("run_ec2")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import numpy as np
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.clientapp import ClientApp
from flwr.common import Context

# ── Scale Configuration ──────────────────────────────────────────────
# 5 clients = 5 hospitals/agencies, realistic for cross-silo FL
NC = 5
GPU_PER_CLIENT = 0.2 if torch.cuda.is_available() else 0.0

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
    "DP_Central_Eps50.0_Alpha_0.5": (30, "DP epsilon=50 moderate"),
    "DP_Central_Eps10.0_Alpha_0.5": (40, "DP epsilon=10 moderate"),
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
        # Skip heavy strategies on DenseNet (too slow for DP, FedAdam diverges)
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

    def sf(ctx):
        return ServerAppComponents(strategy=cap, config=ServerConfig(num_rounds=num_rounds))

    t0 = time.time()
    try:
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
    from models.bilstm.server_app import make_strategy
    from models.bilstm.client_app import client_fn

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
    from models.bilstm.server_app import make_strategy
    from models.bilstm.client_app import client_fn

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
    from models.mlp.server_app import make_strategy
    from models.mlp.client_app import client_fn

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
    from models.densenet.server_app import make_strategy
    from models.densenet.client_app import client_fn

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
        from models.bilstm.server_app import make_strategy
        from models.bilstm.client_app import client_fn
        idim = input_dim or 14
        return lambda n, nc: make_strategy(n, nc, idim), client_fn
    elif task == "ecg":
        os.environ["TASK"] = "ecg"
        os.environ["INPUT_DIM"] = str(input_dim or 12)
        from models.bilstm.server_app import make_strategy
        from models.bilstm.client_app import client_fn
        idim = input_dim or 12
        return lambda n, nc: make_strategy(n, nc, idim), client_fn
    elif task == "fraud":
        from models.mlp.server_app import make_strategy
        from models.mlp.client_app import client_fn
        return make_strategy, client_fn
    elif task == "chest":
        from models.densenet.server_app import make_strategy
        from models.densenet.client_app import client_fn
        return make_strategy, client_fn
    elif task == "vfl_fraud":
        from models.vfl_mlp.server_app import make_strategy
        from models.vfl_mlp.client_app import client_fn
        return make_strategy, client_fn
    elif task == "split_sepsis":
        os.environ["INPUT_DIM"] = str(input_dim or 14)
        from models.split_bilstm.server_app import make_strategy
        from models.split_bilstm.client_app import client_fn
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

    logger.info("=" * 70)
    logger.info("  FL + PET SANDBOX")
    logger.info(f"  Target: {target}")
    logger.info(f"  Clients: {NC}")
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
            "privacy": run_privacy,
        }

        if target == "all":
            tasks = ["sepsis", "ecg", "fraud", "chest", "privacy"]
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
