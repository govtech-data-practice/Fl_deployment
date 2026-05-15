"""
Sepsis Data Pipeline
====================
Pipeline: Load → Validate → Clean → Normalize → Partition → Split → DataLoader

Real data: eICU NPZ files (48 timesteps × 14 vitals)
Fallback:  Synthetic when no NPZ files available
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple
import os
import glob
import logging

logger = logging.getLogger("pipeline.sepsis")


class SepsisDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Step 1: Load ──────────────────────────────────────────────────────

def load_all_npz_data(data_path: str, max_clients: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(data_path, "client_*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in {data_path}")

    files = files[:max_clients]
    X_list, y_list = [], []
    for f in files:
        try:
            data = np.load(f)
            X_list.append(data["X"])
            y_list.append(data["y"])
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not X_list:
        raise RuntimeError("No valid data loaded")

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def _generate_synthetic(n, seq_len=48, features=14, seed=42):
    """Fallback synthetic sepsis data when no NPZ files are available."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, seq_len, features).astype(np.float32)
    y = (rng.rand(n) > 0.7).astype(np.float32)
    return X, y


# ── Step 2: Validate ─────────────────────────────────────────────────

def validate_data(X: np.ndarray, y: np.ndarray):
    """Check data integrity: shape, NaN, label distribution."""
    assert X.ndim == 3, f"Expected 3D input (N, T, F), got shape {X.shape}"
    assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"
    assert X.shape[2] > 0, "Zero features"

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    pos_rate = y.mean()
    logger.info(f"[Validate] samples={len(X)}, shape={X.shape}, "
                f"NaN={nan_count}, Inf={inf_count}, pos_rate={pos_rate:.3f}")

    if nan_count > 0 or inf_count > 0:
        logger.warning(f"[Validate] Found {nan_count} NaN and {inf_count} Inf values")
    if pos_rate < 0.01 or pos_rate > 0.99:
        logger.warning(f"[Validate] Extreme label imbalance: pos_rate={pos_rate:.4f}")

    return nan_count, inf_count


# ── Step 3: Clean ────────────────────────────────────────────────────

def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove samples with NaN/Inf, replace remaining NaN with 0."""
    # Drop rows where any timestep has NaN/Inf
    valid_mask = np.isfinite(X).all(axis=(1, 2))
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"[Clean] Dropped {n_dropped} samples with NaN/Inf")
        X, y = X[valid_mask], y[valid_mask]

    # Safety: replace any remaining NaN (shouldn't happen after filter)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


# ── Step 4: Normalize ────────────────────────────────────────────────

def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature z-score normalization across all timesteps.
    Returns normalized X, mean, std (for inverse transform / inference).
    """
    # X shape: (N, T, F) → compute stats over (N, T) per feature
    flat = X.reshape(-1, X.shape[2])  # (N*T, F)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0  # prevent div-by-zero for constant features
    X_norm = (X - mean) / std
    logger.info(f"[Normalize] {X.shape[2]} features, "
                f"mean range=[{mean.min():.3f}, {mean.max():.3f}], "
                f"std range=[{std.min():.3f}, {std.max():.3f}]")
    return X_norm.astype(np.float32), mean, std


# ── Step 5: Partition ────────────────────────────────────────────────

def partition_iid(y, num_clients, seed):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(y))
    splits = np.array_split(indices, num_clients)
    return {i: splits[i] for i in range(num_clients)}


def partition_label_skew(y, num_clients, alpha, seed):
    rng = np.random.RandomState(seed)
    indices_by_class = [np.where(y == c)[0] for c in range(2)]
    partitions = {i: [] for i in range(num_clients)}

    for c in range(2):
        class_idx = indices_by_class[c].copy()
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        counts = (proportions * len(class_idx)).astype(int)
        counts[-1] = len(class_idx) - counts[:-1].sum()
        splits = np.split(class_idx, np.cumsum(counts)[:-1])
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
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int = 0,
    seed: int = 42,
) -> Tuple[Dict, Dict]:
    # ── Load ──
    try:
        logger.info(f"[Load] Loading from {data_path}...")
        X_pool, y_pool = load_all_npz_data(data_path)
    except (FileNotFoundError, RuntimeError):
        total = 2000 if max_samples == 0 else max_samples
        logger.info(f"[Load] No NPZ files found, generating {total} synthetic samples")
        X_pool, y_pool = _generate_synthetic(total, seed=seed)

    # ── Cap ──
    if max_samples > 0 and len(X_pool) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_pool), max_samples, replace=False)
        X_pool, y_pool = X_pool[idx], y_pool[idx]
        logger.info(f"[Load] Capped to {max_samples} samples")

    # ── Validate ──
    validate_data(X_pool, y_pool)

    # ── Clean ──
    X_pool, y_pool = clean_data(X_pool, y_pool)

    # ── Normalize ──
    X_pool, feat_mean, feat_std = normalize_features(X_pool)

    # ── Partition ──
    logger.info(f"[Partition] {partition_type} (alpha={alpha}, clients={num_clients})")
    if partition_type == "label_skew":
        client_indices = partition_label_skew(y_pool, num_clients, alpha, seed)
    else:
        client_indices = partition_iid(y_pool, num_clients, seed)

    # ── Split → DataLoaders ──
    rng = np.random.RandomState(seed)
    client_loaders = {}

    for cid, indices in client_indices.items():
        if len(indices) == 0:
            continue
        indices = np.array(indices)
        rng.shuffle(indices)

        X_c, y_c = X_pool[indices], y_pool[indices]
        n = len(X_c)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_test - n_val

        client_loaders[cid] = {
            "train": DataLoader(
                SepsisDataset(X_c[:n_train], y_c[:n_train]),
                batch_size=batch_size, shuffle=True,
            ),
            "val": DataLoader(
                SepsisDataset(X_c[n_train:n_train + n_val], y_c[n_train:n_train + n_val]),
                batch_size=batch_size, shuffle=False,
            ),
            "test": DataLoader(
                SepsisDataset(X_c[n_train + n_val:], y_c[n_train + n_val:]),
                batch_size=batch_size, shuffle=False,
            ),
        }
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}, "
                     f"pos_rate={y_c.mean():.3f}")

    metadata = {
        "num_clients": len(client_loaders),
        "input_dim": X_pool.shape[2],
        "seq_len": X_pool.shape[1],
        "total_samples": len(X_pool),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
    }
    return client_loaders, metadata
