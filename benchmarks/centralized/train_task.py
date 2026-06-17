#!/usr/bin/env python3
"""Centralised training baseline — trains any supported task on pooled data.

This is the upper-bound baseline: all data centralised, no federation overhead,
no DP noise, no SecAgg. FL results are compared against these numbers.

Usage:
    python benchmarks/centralized/train_task.py fraud
    python benchmarks/centralized/train_task.py sepsis --epochs 20
    python benchmarks/centralized/train_task.py ecg --data-dir data/samples/ecg
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("benchmarks.centralized")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Task -> (model module path, model class, default data dir)
TASK_CONFIG = {
    "fraud":       {"model": "models.hfl.mlp.server_app",       "class": "FraudMLP",       "input_dim": 30},
    "sepsis":      {"model": "models.hfl.bilstm.server_app",    "class": "SepsisBiLSTM",   "input_dim": 14, "seq_len": 48},
    "ecg":         {"model": "models.hfl.bilstm.server_app",    "class": "SepsisBiLSTM",   "input_dim": 12, "seq_len": 250},
    "anomaly":     {"model": "models.hfl.autoencoder.server_app","class": "AnomalyAutoencoder", "input_dim": 40},
    "mortality":   {"model": "models.hfl.tabnet_simple.server_app", "class": "SimpleTabNet", "input_dim": 25},
    "readmission": {"model": "models.hfl.logreg.server_app",    "class": "LogReg",         "input_dim": 20},
    "drug":        {"model": "models.hfl.generic.server_app",   "class": "GenericMLP",     "input_dim": 200},
    "satellite":   {"model": "models.hfl.resnet_small.server_app", "class": "SmallResNet",  "input_dim": 3},
}


def load_data(task, data_dir=None):
    """Load data from samples or generate synthetic."""
    if data_dir and os.path.exists(os.path.join(data_dir, "data.npz")):
        data = np.load(os.path.join(data_dir, "data.npz"))
        X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
        logger.info("Loaded %d samples from %s", len(X), data_dir)
    else:
        # Generate synthetic
        sample_dir = REPO_ROOT / "data" / "samples" / task
        if sample_dir.exists():
            data = np.load(str(sample_dir / "data.npz"))
            X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
            logger.info("Loaded %d samples from data/samples/%s", len(X), task)
        else:
            logger.info("No data found — generating synthetic")
            cfg = TASK_CONFIG[task]
            rng = np.random.RandomState(42)
            n = 500
            dim = cfg["input_dim"]
            if "seq_len" in cfg:
                X = rng.randn(n, cfg["seq_len"], dim).astype(np.float32)
            else:
                X = rng.randn(n, dim).astype(np.float32)
            weights = rng.randn(dim)
            if X.ndim == 3:
                logits = X[:, -1, :] @ weights
            else:
                logits = X @ weights
            y = (logits > 0).astype(np.float32)

    return X, y


def build_model(task):
    """Build the model for a task."""
    cfg = TASK_CONFIG[task]
    dim = cfg["input_dim"]

    if task == "fraud":
        from models.hfl.mlp.server_app import FraudMLP
        return FraudMLP(input_dim=dim)
    elif task in ("sepsis", "ecg"):
        model = nn.Sequential(
            nn.LSTM(dim, 64, batch_first=True, bidirectional=True),
        )
        # Simple BiLSTM + classifier
        class BiLSTMClassifier(nn.Module):
            def __init__(self, input_dim, hidden=64):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
                self.classifier = nn.Sequential(nn.Linear(hidden * 2, 1), nn.Sigmoid())
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.classifier(out[:, -1, :]).squeeze(-1)
        return BiLSTMClassifier(dim)
    elif task == "anomaly":
        class Autoencoder(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(d, 20), nn.ReLU(), nn.Linear(20, 10))
                self.decoder = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, d))
            def forward(self, x):
                return self.decoder(self.encoder(x))
        return Autoencoder(dim)
    elif task == "readmission":
        return nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
    else:
        # Generic MLP
        return nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid(),
        )


def train_centralised(task, epochs=10, batch_size=32, lr=0.001, data_dir=None):
    """Train a model on pooled data (centralised baseline)."""
    X, y = load_data(task, data_dir)
    model = build_model(task).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Task: %s  Model: %d params  Device: %s", task, n_params, DEVICE)

    # Split train/val
    n = len(X)
    n_val = max(int(n * 0.2), 1)
    X_train, y_train = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    is_reconstruction = (task == "anomaly")
    loss_fn = nn.MSELoss() if is_reconstruction else nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            if is_reconstruction:
                loss = loss_fn(pred, xb)
            else:
                if pred.ndim > 1:
                    pred = pred.squeeze(-1)
                loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        # Validation
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).to(DEVICE)
            yv = torch.from_numpy(y_val)
            pv = model(xv).cpu()
            if pv.ndim > 1:
                pv = pv.squeeze(-1)

            if is_reconstruction:
                val_loss = nn.MSELoss()(pv, xv.cpu()).item()
                val_acc = 1.0 - val_loss  # proxy metric
            else:
                val_loss = nn.BCELoss()(pv, yv).item()
                val_acc = accuracy_score(yv.numpy() > 0.5, pv.numpy() > 0.5)

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / len(X_train), 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

    duration = time.time() - t0

    result = {
        "task": task,
        "mode": "centralised",
        "epochs": epochs,
        "samples": len(X),
        "params": n_params,
        "device": DEVICE,
        "final_accuracy": history[-1]["val_accuracy"],
        "final_loss": history[-1]["val_loss"],
        "duration_seconds": round(duration, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "history": history,
    }

    # Print summary
    print()
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>10}")
    print("-" * 42)
    for h in history:
        print(f"{h['epoch']:>6} {h['train_loss']:>12.4f} {h['val_loss']:>10.4f} {h['val_accuracy']:>10.4f}")
    print("-" * 42)
    print(f"Final: accuracy={result['final_accuracy']:.4f}  "
          f"loss={result['final_loss']:.4f}  time={duration:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train centralised baseline (no federation)."
    )
    parser.add_argument("task", choices=list(TASK_CONFIG.keys()),
                        help="Task to benchmark")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument("--output", help="Save results JSON to file")
    args = parser.parse_args()

    result = train_centralised(args.task, args.epochs, args.batch_size,
                                args.lr, args.data_dir)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
