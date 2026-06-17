#!/usr/bin/env python3
"""Generate synthetic sample datasets for all FL tasks.

Writes to data/samples/<task>/ with:
    - data.npz (features + labels)
    - manifest.json (metadata)
    - data_card.md (dataset documentation)

Every run is logged to data/audit/generation_log.jsonl.

Usage:
    python data/generators/generate_all.py                     # all tasks, 500 samples each
    python data/generators/generate_all.py --task fraud         # single task
    python data/generators/generate_all.py --num-samples 1000   # custom size
    python data/generators/generate_all.py --seed 123           # reproducible
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("data.generators")

GENERATOR_VERSION = "1.0.0"

SAMPLES_DIR = REPO_ROOT / "data" / "samples"
AUDIT_DIR = REPO_ROOT / "data" / "audit"

# ── Task generators ───────────────────────────────────────────────

TASKS = {
    "fraud": {
        "input_dim": 30,
        "num_classes": 2,
        "task_type": "binary",
        "description": "Synthetic transaction records (30 features, binary fraud label)",
    },
    "sepsis": {
        "input_dim": 14,
        "num_classes": 2,
        "task_type": "binary",
        "seq_len": 48,
        "description": "Synthetic vitals/labs time series (14 features, 48 timesteps)",
    },
    "ecg": {
        "input_dim": 12,
        "num_classes": 2,
        "task_type": "binary",
        "seq_len": 250,
        "description": "Synthetic 12-lead ECG signals (250 timesteps)",
    },
    "anomaly": {
        "input_dim": 40,
        "num_classes": 2,
        "task_type": "reconstruction",
        "description": "Synthetic embeddings for anomaly detection (40-dim)",
    },
    "mortality": {
        "input_dim": 25,
        "num_classes": 2,
        "task_type": "binary",
        "description": "Synthetic ICU records (25 features, mortality label)",
    },
    "drug": {
        "input_dim": 200,
        "num_classes": 2,
        "task_type": "binary",
        "description": "Synthetic molecular fingerprints (200-dim)",
    },
    "readmission": {
        "input_dim": 20,
        "num_classes": 2,
        "task_type": "binary",
        "description": "Synthetic discharge records (20 features)",
    },
    "satellite": {
        "input_dim": 3,
        "num_classes": 5,
        "task_type": "multiclass",
        "image_size": 64,
        "description": "Synthetic 64x64 multispectral patches (5 land-use classes)",
    },
}


def generate_tabular(n, input_dim, num_classes, task_type, seed=42):
    """Generate synthetic tabular data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, input_dim).astype(np.float32)

    if task_type == "binary" or num_classes == 2:
        # Logistic separation with noise
        weights = rng.randn(input_dim)
        logits = X @ weights + rng.randn(n) * 0.5
        y = (logits > 0).astype(np.float32)
        # Ensure minimum 10% positive rate
        if y.mean() < 0.1:
            flip = rng.choice(np.where(y == 0)[0], int(n * 0.15), replace=False)
            y[flip] = 1.0
    elif task_type == "multiclass":
        y = rng.randint(0, num_classes, n).astype(np.float32)
    else:  # reconstruction
        y = np.zeros(n, dtype=np.float32)

    return X, y


def generate_timeseries(n, input_dim, seq_len, seed=42):
    """Generate synthetic time-series data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, seq_len, input_dim).astype(np.float32)
    # Add temporal structure
    for i in range(seq_len):
        X[:, i, :] += np.sin(i / seq_len * np.pi) * 0.3
    weights = rng.randn(input_dim)
    logits = X[:, -1, :] @ weights + rng.randn(n) * 0.5
    y = (logits > 0).astype(np.float32)
    return X, y


def generate_image(n, num_classes, image_size, seed=42):
    """Generate synthetic image patches."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3, image_size, image_size).astype(np.float32) * 0.3
    y = rng.randint(0, num_classes, n).astype(np.float32)
    # Add class-dependent patterns
    for cls in range(num_classes):
        mask = y == cls
        X[mask, cls % 3, :, :] += 0.5
    return X, y


def compute_checksum(path):
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(output_dir, task, task_config, n, checksum):
    """Write a manifest.json for the generated dataset."""
    unique_labels = task_config.get("num_classes", 2)
    manifest = {
        "task": task,
        "format": "npz",
        "num_samples": n,
        "schema": {
            "input_dim": task_config["input_dim"],
            "num_classes": task_config["num_classes"],
            "seq_len": task_config.get("seq_len", 0),
            "task_type": task_config["task_type"],
        },
        "label_distribution": {str(i): round(1.0 / unique_labels, 4)
                               for i in range(unique_labels)},
        "checksum": checksum,
        "client_id": "synthetic",
        "version": "1.0",
        "data_path": "data.npz",
        "created": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "synthetic": True,
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def write_data_card(output_dir, task, task_config, n, seed, checksum):
    """Write a data_card.md documenting the generated dataset."""
    card = f"""# Dataset Card — {task} (synthetic)

## Overview
- **Task:** {task}
- **Description:** {task_config['description']}
- **Samples:** {n}
- **Synthetic:** Yes (not derived from real data)

## Schema
- **Input dim:** {task_config['input_dim']}
- **Classes:** {task_config['num_classes']}
- **Task type:** {task_config['task_type']}
- **Sequence length:** {task_config.get('seq_len', 'N/A')}

## Provenance
- **Generator:** data/generators/generate_all.py v{GENERATOR_VERSION}
- **Seed:** {seed}
- **Checksum:** {checksum}
- **Generated:** {datetime.now(timezone.utc).isoformat()}

## Usage
This is synthetic data for testing pipeline integration, partitioning
strategies, and non-IID scenarios. It is NOT intended to replicate
real-world data distributions.
"""
    path = os.path.join(output_dir, "data_card.md")
    with open(path, "w") as f:
        f.write(card)
    return path


def log_generation(task, n, output_dir, checksum, seed, duration):
    """Append an entry to the audit log."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = AUDIT_DIR / "generation_log.jsonl"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "num_samples": n,
        "output_dir": str(output_dir),
        "checksum": checksum,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "duration_seconds": round(duration, 2),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def generate_task(task, num_samples=500, seed=42):
    """Generate synthetic data for a single task."""
    cfg = TASKS[task]
    output_dir = SAMPLES_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Generate based on data type
    if cfg.get("seq_len"):
        X, y = generate_timeseries(num_samples, cfg["input_dim"],
                                   cfg["seq_len"], seed)
    elif cfg.get("image_size"):
        X, y = generate_image(num_samples, cfg["num_classes"],
                              cfg["image_size"], seed)
    else:
        X, y = generate_tabular(num_samples, cfg["input_dim"],
                                cfg["num_classes"], cfg["task_type"], seed)

    # Save
    npz_path = output_dir / "data.npz"
    np.savez_compressed(str(npz_path), X=X, y=y)
    checksum = compute_checksum(str(npz_path))
    duration = time.time() - t0

    # Write manifest and data card
    write_manifest(str(output_dir), task, cfg, num_samples, checksum)
    write_data_card(str(output_dir), task, cfg, num_samples, seed, checksum)

    # Audit log
    log_generation(task, num_samples, output_dir, checksum, seed, duration)

    # Summary
    size_kb = os.path.getsize(str(npz_path)) / 1024
    pos_rate = float(y.mean()) if cfg["task_type"] == "binary" else -1

    logger.info(
        "  %-12s %5d samples  shape=%-20s  size=%7.1f KB  checksum=%s...  %.2fs",
        task, num_samples, str(list(X.shape)), size_kb, checksum[:12], duration
    )

    return {
        "task": task,
        "samples": num_samples,
        "shape": list(X.shape),
        "size_kb": round(size_kb, 1),
        "checksum": checksum,
        "positive_rate": round(pos_rate, 4) if pos_rate >= 0 else None,
        "duration": round(duration, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sample datasets for FL tasks."
    )
    parser.add_argument("--task", choices=list(TASKS.keys()),
                        help="Generate for a single task (default: all)")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Samples per task (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASKS.keys())

    print(f"Generating synthetic data for {len(tasks)} task(s)")
    print(f"Samples per task: {args.num_samples}")
    print(f"Seed: {args.seed}")
    print(f"Output: {SAMPLES_DIR}/")
    print()

    results = []
    for task in tasks:
        r = generate_task(task, args.num_samples, args.seed)
        results.append(r)

    print()
    print("=" * 70)
    print(f"{'Task':<12} {'Samples':>8} {'Shape':<22} {'Size':>10} {'Checksum':<14}")
    print("-" * 70)
    for r in results:
        print(f"{r['task']:<12} {r['samples']:>8} {str(r['shape']):<22} "
              f"{r['size_kb']:>8.1f} KB  {r['checksum'][:12]}...")
    print("-" * 70)
    total_kb = sum(r["size_kb"] for r in results)
    print(f"{'Total':<12} {sum(r['samples'] for r in results):>8} "
          f"{'':22} {total_kb:>8.1f} KB")
    print()
    print(f"Audit log: {AUDIT_DIR / 'generation_log.jsonl'}")


if __name__ == "__main__":
    main()
