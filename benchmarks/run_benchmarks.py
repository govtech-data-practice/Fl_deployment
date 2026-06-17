#!/usr/bin/env python3
"""Benchmark runner — centralised vs FL comparison across tasks.

Runs centralised training and FL (multiple strategies) for each task,
then produces a comparison table.

Usage:
    python benchmarks/run_benchmarks.py                        # all tasks
    python benchmarks/run_benchmarks.py --tasks fraud sepsis   # specific tasks
    python benchmarks/run_benchmarks.py --output results.csv   # save CSV
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.centralized.train_task import train_centralised, TASK_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("benchmarks")

DEFAULT_TASKS = ["fraud", "sepsis", "ecg", "anomaly", "mortality", "readmission", "drug"]
FL_STRATEGIES = ["IID", "FedProx", "SCAFFOLD", "DP-Central"]


def run_fl_benchmark(task, strategy, num_rounds=5, num_clients=3):
    """Run FL training and return metrics (simulation mode)."""
    import torch
    from models.hfl.mlp.server_app import FraudMLP
    from fl_common.strategies import build_strategy
    from benchmarks.centralized.train_task import load_data, build_model

    X, y = load_data(task)
    model = build_model(task)
    n_params = sum(p.numel() for p in model.parameters())

    # Simulate FL by partitioning data and averaging
    n = len(X)
    partition_size = n // num_clients
    client_accuracies = []

    t0 = time.time()

    for round_i in range(num_rounds):
        round_weights = []
        for c in range(num_clients):
            start = c * partition_size
            end = start + partition_size
            Xc = torch.from_numpy(X[start:end])
            yc = torch.from_numpy(y[start:end])

            # Local training (1 epoch)
            client_model = build_model(task)
            client_model.load_state_dict(model.state_dict())
            opt = torch.optim.Adam(client_model.parameters(), lr=0.001)
            loss_fn = torch.nn.BCELoss() if task != "anomaly" else torch.nn.MSELoss()

            client_model.train()
            pred = client_model(Xc)
            if pred.ndim > 1:
                pred = pred.squeeze(-1)
            if task == "anomaly":
                loss = loss_fn(pred, Xc)
            else:
                loss = loss_fn(pred, yc)
            opt.zero_grad()
            loss.backward()
            opt.step()

            round_weights.append({k: v.clone() for k, v in client_model.state_dict().items()})

        # FedAvg: average weights
        avg_state = {}
        for key in round_weights[0]:
            avg_state[key] = torch.stack([w[key].float() for w in round_weights]).mean(dim=0)
        model.load_state_dict(avg_state)

    duration = time.time() - t0

    # Evaluate
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X)
        pred = model(Xt)
        if pred.ndim > 1:
            pred = pred.squeeze(-1)
        if task == "anomaly":
            acc = 1.0 - torch.nn.MSELoss()(pred, Xt).item()
        else:
            pred_np = pred.numpy()
            acc = float(((pred_np > 0.5) == (y > 0.5)).mean())

    return {
        "task": task,
        "mode": f"FL ({strategy})",
        "accuracy": round(float(acc), 4),
        "duration": round(duration, 2),
        "rounds": num_rounds,
        "clients": num_clients,
        "params": n_params,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Centralised vs FL benchmark comparison."
    )
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_CONFIG.keys()),
                        default=DEFAULT_TASKS,
                        help="Tasks to benchmark (default: all)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Centralised training epochs")
    parser.add_argument("--fl-rounds", type=int, default=5,
                        help="FL training rounds")
    parser.add_argument("--output", help="Save results to CSV")
    args = parser.parse_args()

    print("=" * 80)
    print("FL Benchmark: Centralised vs Federated Training")
    print("=" * 80)
    print(f"Tasks:      {args.tasks}")
    print(f"Central:    {args.epochs} epochs")
    print(f"FL:         {args.fl_rounds} rounds, 3 clients")
    print()

    results = []

    for task in args.tasks:
        print(f"\n{'─' * 80}")
        print(f"Task: {task}")
        print(f"{'─' * 80}")

        # Centralised baseline
        print("\n[Centralised]")
        central = train_centralised(task, epochs=args.epochs)
        results.append({
            "task": task,
            "mode": "Centralised",
            "accuracy": central["final_accuracy"],
            "duration": central["duration_seconds"],
        })

        # FL
        print("\n[FL FedAvg]")
        fl_result = run_fl_benchmark(task, "FedAvg", num_rounds=args.fl_rounds)
        results.append({
            "task": task,
            "mode": "FL (FedAvg)",
            "accuracy": fl_result["accuracy"],
            "duration": fl_result["duration"],
        })

    # Summary table
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Task':<15} {'Mode':<20} {'Accuracy':>10} {'Time':>10}")
    print("-" * 58)
    for r in results:
        print(f"{r['task']:<15} {r['mode']:<20} {r['accuracy']:>10.4f} {r['duration']:>8.1f}s")

    # Save CSV
    if args.output:
        results_dir = REPO_ROOT / "benchmarks" / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / args.output
        with open(str(output_path), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task", "mode", "accuracy", "duration"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
