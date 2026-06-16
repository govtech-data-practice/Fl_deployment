"""
Generic Data Pipeline — config-driven, no code changes needed
==============================================================
Supports arbitrary tabular CSV/NPZ datasets and image folders.

Usage:
  # Via YAML config
  task: generic
  data_config:
    type: tabular          # or "image"
    path: /data/my_dataset.csv
    target_column: label
    feature_columns: [age, bp, glucose, ...]  # optional, default=all except target
    input_dim: auto        # auto-detect from data
    num_classes: auto       # auto-detect from target
    task_type: binary       # binary, multiclass, regression

  # Or via environment variables
  GENERIC_DATA_PATH=/data/my_dataset.csv
  GENERIC_TARGET_COL=label
  GENERIC_TASK_TYPE=binary
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Optional, List
import logging
import os
import json

logger = logging.getLogger("pipeline.generic")


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        if y.dtype in (np.float32, np.float64):
            self.y = torch.from_numpy(y).float()
        else:
            self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImageDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, H, W, C) -> (N, C, H, W)
        self.X = torch.from_numpy(X.transpose(0, 3, 1, 2)).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Data Loading ─────────────────────────────────────────────────────

def _load_csv(path, target_column, feature_columns=None, max_samples=0):
    """Load tabular data from CSV."""
    import csv
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_samples > 0:
        rows = rows[:max_samples]

    if feature_columns is None or len(feature_columns) == 0:
        feature_columns = [c for c in rows[0].keys() if c != target_column]

    X = np.array([[float(r.get(c, 0)) for c in feature_columns] for r in rows], dtype=np.float32)
    y = np.array([float(r[target_column]) for r in rows], dtype=np.float32)

    logger.info(f"[Load CSV] {path}: {len(X)} samples, {X.shape[1]} features, target={target_column}")
    return X, y, feature_columns


def _load_npz(path, feature_key="X", target_key="y", max_samples=0):
    """Load tabular data from NPZ."""
    data = np.load(path, allow_pickle=True)
    X = data[feature_key].astype(np.float32)
    y = data[target_key].astype(np.float32)
    if max_samples > 0:
        X, y = X[:max_samples], y[:max_samples]
    logger.info(f"[Load NPZ] {path}: {len(X)} samples, {X.shape[1]} features")
    return X, y


def _generate_synthetic(n, input_dim, num_classes, task_type, seed=42):
    """Generate synthetic data when no real dataset provided."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, input_dim).astype(np.float32)

    if task_type == "binary":
        # Linear decision boundary with noise
        w = rng.randn(input_dim).astype(np.float32)
        logits = X @ w + rng.randn(n).astype(np.float32) * 0.5
        y = (logits > np.median(logits)).astype(np.float32)
    elif task_type == "multiclass":
        # Simple cluster-based assignment
        w = rng.randn(input_dim, num_classes).astype(np.float32)
        logits = X @ w
        y = logits.argmax(axis=1).astype(np.int64)
    else:  # regression
        w = rng.randn(input_dim).astype(np.float32)
        y = (X @ w + rng.randn(n).astype(np.float32) * 0.1).astype(np.float32)

    logger.info(f"[Synthetic] {n} samples, {input_dim} features, {num_classes} classes, {task_type}")
    return X, y


# ── Partition ────────────────────────────────────────────────────────

def partition_iid(y, num_clients, seed):
    rng = np.random.RandomState(seed)
    return {i: s for i, s in enumerate(np.array_split(rng.permutation(len(y)), num_clients))}


def partition_label_skew(y, num_clients, alpha, seed):
    rng = np.random.RandomState(seed)
    classes = sorted(set(y.astype(int)))
    partitions = {i: [] for i in range(num_clients)}
    for c in classes:
        idx = np.where(y.astype(int) == c)[0].copy()
        rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        counts = (props * len(idx)).astype(int)
        counts[-1] = len(idx) - counts[:-1].sum()
        splits = np.split(idx, np.cumsum(counts)[:-1])
        for i in range(num_clients):
            partitions[i].extend(splits[i])
    for i in range(num_clients):
        rng.shuffle(partitions[i])
    return partitions


# ── Normalize ────────────────────────────────────────────────────────

def normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


# ── Main Entry Point ────────────────────────────────────────────────

def prepare_federated_data(
    data_path: str = "",
    num_clients: int = 5,
    partition_type: str = "iid",
    alpha: float = 0.5,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int = 0,
    seed: int = 42,
    # Generic config (from env or passed directly)
    data_config: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Config-driven federated data preparation.

    data_config can be passed directly or loaded from environment:
      GENERIC_DATA_PATH: path to CSV/NPZ file
      GENERIC_TARGET_COL: target column name (CSV)
      GENERIC_FEATURE_COLS: comma-separated feature columns (CSV, optional)
      GENERIC_INPUT_DIM: number of features (for synthetic)
      GENERIC_NUM_CLASSES: number of classes (for synthetic)
      GENERIC_TASK_TYPE: binary/multiclass/regression
      GENERIC_DATA_TYPE: tabular/image
    """
    # Load config from env if not provided
    if data_config is None:
        data_config = {
            "type": os.environ.get("GENERIC_DATA_TYPE", "tabular"),
            "path": os.environ.get("GENERIC_DATA_PATH", ""),
            "target_column": os.environ.get("GENERIC_TARGET_COL", "label"),
            "feature_columns": os.environ.get("GENERIC_FEATURE_COLS", "").split(",") if os.environ.get("GENERIC_FEATURE_COLS") else None,
            "input_dim": int(os.environ.get("GENERIC_INPUT_DIM", "30")),
            "num_classes": int(os.environ.get("GENERIC_NUM_CLASSES", "2")),
            "task_type": os.environ.get("GENERIC_TASK_TYPE", "binary"),
        }

    data_type = data_config.get("type", "tabular")
    file_path = data_config.get("path", "") or data_path
    target_column = data_config.get("target_column", "label")
    feature_columns = data_config.get("feature_columns")
    input_dim = data_config.get("input_dim", 30)
    num_classes = data_config.get("num_classes", 2)
    task_type = data_config.get("task_type", "binary")

    # ── Load data ──
    if file_path and os.path.exists(file_path):
        if file_path.endswith(".csv"):
            X, y, feature_columns = _load_csv(file_path, target_column, feature_columns, max_samples)
            input_dim = X.shape[1]
        elif file_path.endswith(".npz"):
            X, y = _load_npz(file_path, max_samples=max_samples)
            input_dim = X.shape[1]
        else:
            logger.warning(f"Unsupported file format: {file_path}, generating synthetic")
            total = max_samples if max_samples > 0 else 5000
            X, y = _generate_synthetic(total, input_dim, num_classes, task_type, seed)
    else:
        # No file — generate synthetic
        total = max_samples if max_samples > 0 else 5000
        X, y = _generate_synthetic(total, input_dim, num_classes, task_type, seed)

    # Auto-detect dimensions
    input_dim = X.shape[1] if X.ndim == 2 else X.shape[1:]
    unique_y = sorted(set(y.astype(int).tolist()))
    num_classes = len(unique_y)
    if num_classes == 2:
        task_type = "binary"
    elif num_classes > 2:
        task_type = "multiclass"

    # ── Clean ──
    if X.ndim == 2:
        valid = np.isfinite(X).all(axis=1)
        X, y = X[valid], y[valid]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Normalize (tabular only) ──
    if data_type == "tabular" and X.ndim == 2:
        X, feat_mean, feat_std = normalize_features(X)
    else:
        feat_mean, feat_std = None, None

    # ── Partition ──
    logger.info(f"[Partition] {partition_type} (alpha={alpha}, clients={num_clients})")
    if partition_type == "label_skew":
        client_indices = partition_label_skew(y, num_clients, alpha, seed)
    else:
        client_indices = partition_iid(y, num_clients, seed)

    # ── Split + DataLoaders ──
    rng = np.random.RandomState(seed)
    DatasetClass = ImageDataset if data_type == "image" else TabularDataset
    loaders = {}
    for cid, indices in client_indices.items():
        if len(indices) == 0:
            continue
        indices = np.array(indices)
        rng.shuffle(indices)
        Xc, yc = X[indices], y[indices]
        n = len(Xc)
        n_test = max(int(n * test_ratio), 1)
        n_val = max(int(n * val_ratio), 1)
        n_train = n - n_test - n_val
        if n_train < 1:
            continue
        loaders[cid] = {
            "train": DataLoader(DatasetClass(Xc[:n_train], yc[:n_train]), batch_size=batch_size, shuffle=True),
            "val": DataLoader(DatasetClass(Xc[n_train:n_train+n_val], yc[n_train:n_train+n_val]), batch_size=batch_size),
            "test": DataLoader(DatasetClass(Xc[n_train+n_val:], yc[n_train+n_val:]), batch_size=batch_size),
        }
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}")

    metadata = {
        "num_clients": len(loaders),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "task_type": task_type,
        "total_samples": len(X),
        "feature_columns": feature_columns,
        "feature_mean": feat_mean,
        "feature_std": feat_std,
    }

    logger.info(f"[Ready] {len(loaders)} clients, input_dim={input_dim}, "
                f"classes={num_classes}, type={task_type}")
    return loaders, metadata
