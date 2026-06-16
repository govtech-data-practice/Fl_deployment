"""
Fraud Transaction Data Pipeline
================================
Pipeline: Load → Validate → Clean → Normalize → Partition → Split → DataLoader

Synthetic tabular: 30 features, 2% fraud rate (highly imbalanced).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple
import logging

from fl_common.data import DataConfig, load_dataset, validate_tabular, partition_local

logger = logging.getLogger("pipeline.fraud")


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Step 1: Load ──────────────────────────────────────────────────────

def _generate_fraud(n, n_features=30, fraud_rate=0.02, seed=42):
    """Synthetic fraud transactions.
    Fraud samples have shifted distribution in features 0-4 (simulated anomaly).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (rng.rand(n) < fraud_rate).astype(np.float32)
    # Ensure at least some positive samples for small datasets
    if y.sum() == 0:
        n_pos = max(1, int(n * fraud_rate))
        y[:n_pos] = 1.0
    # Add signal: fraud transactions have higher values in features 0-4
    fraud_mask = y == 1
    X[fraud_mask, :5] += 2.0
    return X, y


# ── Step 2: Validate ─────────────────────────────────────────────────

def validate_data(X: np.ndarray, y: np.ndarray):
    assert X.ndim == 2, f"Expected 2D input (N, F), got shape {X.shape}"
    assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    pos_rate = y.mean()
    n_pos = int(y.sum())
    logger.info(f"[Validate] samples={len(X)}, features={X.shape[1]}, "
                f"NaN={nan_count}, Inf={inf_count}, "
                f"fraud={n_pos} ({pos_rate:.4f})")

    if nan_count > 0 or inf_count > 0:
        logger.warning(f"[Validate] Found {nan_count} NaN and {inf_count} Inf values")
    if pos_rate < 0.001:
        logger.warning(f"[Validate] Very low fraud rate ({pos_rate:.5f}), may affect training")
    return nan_count, inf_count


# ── Step 3: Clean ────────────────────────────────────────────────────

def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(X).all(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"[Clean] Dropped {n_dropped} samples with NaN/Inf")
        X, y = X[valid_mask], y[valid_mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


# ── Step 4: Normalize ────────────────────────────────────────────────

def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score normalization."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std
    logger.info(f"[Normalize] {X.shape[1]} features, "
                f"mean range=[{mean.min():.3f}, {mean.max():.3f}], "
                f"std range=[{std.min():.3f}, {std.max():.3f}]")
    return X_norm.astype(np.float32), mean, std


# ── Step 5: Partition ────────────────────────────────────────────────

def partition_iid(y, num_clients, seed):
    rng = np.random.RandomState(seed)
    return {i: s for i, s in enumerate(np.array_split(rng.permutation(len(y)), num_clients))}


def partition_label_skew(y, num_clients, alpha, seed):
    """Dirichlet-based label skew for imbalanced binary data."""
    rng = np.random.RandomState(seed)
    partitions = {i: [] for i in range(num_clients)}
    for c in range(2):
        idx = np.where(y == c)[0].copy()
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


# ── Step 6: Split + DataLoader ───────────────────────────────────────

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
    local_mode: bool = False,
) -> Tuple[Dict, Dict]:
    # ── Try loading real data via fl_common.data ──
    config = DataConfig.for_task("fraud", data_path)
    if data_path:
        try:
            X, y, meta = load_dataset(config)
            result = validate_tabular(X, y, config)
            if not result.is_valid:
                raise ValueError(f"Data validation failed: {result.errors}")
            logger.info(f"[Load] Loaded real data from {data_path}: {len(X)} samples")

            if local_mode:
                # Each client has its own data file — no cross-client partition
                splits = partition_local(X, y, val_ratio, test_ratio, seed)
                loaders = {0: {
                    "train": DataLoader(TabularDataset(*splits["train"]), batch_size=batch_size, shuffle=True),
                    "val": DataLoader(TabularDataset(*splits["val"]), batch_size=batch_size),
                    "test": DataLoader(TabularDataset(*splits["test"]), batch_size=batch_size),
                }}
                metadata = {
                    "num_clients": 1,
                    "input_dim": X.shape[1],
                    "total_samples": len(X),
                    "fraud_rate": float(y.mean()),
                    "source": meta.get("source", data_path),
                }
                return loaders, metadata
            # else: fall through to partition logic below with real data
        except FileNotFoundError:
            if local_mode:
                raise FileNotFoundError(
                    f"No data found at {data_path}. Provide data or set synthetic=True. "
                    f"Run 'python ingest.py --task fraud --input <your_data>' to ingest data."
                )
            logger.info("[Load] Real data not found, falling back to synthetic")
            X, y = None, None
    else:
        X, y = None, None

    # ── Synthetic fallback ──
    if X is None:
        total = 5000 if max_samples == 0 else max_samples
        logger.info(f"[Load] Generating {total} synthetic fraud transactions")
        X, y = _generate_fraud(total, seed=seed)

    # ── Validate ──
    validate_data(X, y)

    # ── Clean ──
    X, y = clean_data(X, y)

    # ── Normalize ──
    X, feat_mean, feat_std = normalize_features(X)

    # ── Partition ──
    logger.info(f"[Partition] {partition_type} (alpha={alpha}, clients={num_clients})")
    if partition_type == "label_skew":
        client_indices = partition_label_skew(y, num_clients, alpha, seed)
    else:
        client_indices = partition_iid(y, num_clients, seed)

    # ── Split → DataLoaders ──
    rng = np.random.RandomState(seed)
    loaders = {}
    for cid, indices in client_indices.items():
        if len(indices) == 0:
            continue
        indices = np.array(indices)
        rng.shuffle(indices)
        Xc, yc = X[indices], y[indices]
        n = len(Xc)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_test - n_val
        loaders[cid] = {
            "train": DataLoader(TabularDataset(Xc[:n_train], yc[:n_train]), batch_size=batch_size, shuffle=True),
            "val": DataLoader(TabularDataset(Xc[n_train:n_train+n_val], yc[n_train:n_train+n_val]), batch_size=batch_size),
            "test": DataLoader(TabularDataset(Xc[n_train+n_val:], yc[n_train+n_val:]), batch_size=batch_size),
        }
        n_fraud = int(yc.sum())
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}, "
                     f"fraud={n_fraud}/{n} ({yc.mean():.4f})")

    metadata = {
        "num_clients": len(loaders),
        "input_dim": 30,
        "total_samples": len(X),
        "fraud_rate": float(y.mean()),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
    }
    return loaders, metadata
