"""
Drug Property Prediction (Molecular) Data Pipeline
====================================================
Pipeline: Load -> Validate -> Clean -> Normalize -> Partition -> Split -> DataLoader

Synthetic tabular: 200 features (molecular fingerprints), binary classification.
Use case: pharma, drug discovery.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple
import logging

from fl_common.data import DataConfig, load_dataset, validate_tabular, partition_local

logger = logging.getLogger("pipeline.drug")


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- Step 1: Load --------------------------------------------------------

def _generate_drug(n, n_features=200, active_rate=0.15, seed=42):
    """Synthetic molecular fingerprint data.

    Each sample represents a compound encoded as a 200-dimensional binary/
    continuous fingerprint vector.  Active compounds share a sparse
    pharmacophore pattern in features 0-19 and have elevated lipophilicity
    descriptors in features 20-29.
    """
    rng = np.random.RandomState(seed)

    # Base fingerprints: sparse binary bits with some continuous descriptors
    # First 150 features: binary fingerprint bits (Bernoulli p=0.3)
    X_bits = (rng.rand(n, 150) < 0.3).astype(np.float32)
    # Last 50 features: continuous molecular descriptors (mol weight, logP, etc.)
    X_desc = rng.randn(n, 50).astype(np.float32) * 0.5 + 0.0
    X = np.concatenate([X_bits, X_desc], axis=1)

    y = (rng.rand(n) < active_rate).astype(np.float32)
    active_mask = y == 1
    n_active = int(active_mask.sum())

    # Pharmacophore pattern: active compounds have specific bits turned on
    # in positions 0-19 with higher probability
    X[active_mask, :20] = (rng.rand(n_active, 20) < 0.7).astype(np.float32)
    # Lipophilicity descriptors shifted for active compounds
    X[active_mask, 150:160] += rng.uniform(0.5, 1.5, size=(n_active, 10))
    # Hydrogen-bond donors/acceptors pattern
    X[active_mask, 160:165] += 0.8

    return X, y


# -- Step 2: Validate -----------------------------------------------------

def validate_data(X: np.ndarray, y: np.ndarray):
    assert X.ndim == 2, f"Expected 2D input (N, F), got shape {X.shape}"
    assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    pos_rate = y.mean()
    n_pos = int(y.sum())
    logger.info(f"[Validate] samples={len(X)}, features={X.shape[1]}, "
                f"NaN={nan_count}, Inf={inf_count}, "
                f"active={n_pos} ({pos_rate:.4f})")

    if nan_count > 0 or inf_count > 0:
        logger.warning(f"[Validate] Found {nan_count} NaN and {inf_count} Inf values")
    if pos_rate < 0.01:
        logger.warning(f"[Validate] Very low active rate ({pos_rate:.5f}), may affect training")
    return nan_count, inf_count


# -- Step 3: Clean --------------------------------------------------------

def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(X).all(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"[Clean] Dropped {n_dropped} samples with NaN/Inf")
        X, y = X[valid_mask], y[valid_mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


# -- Step 4: Normalize ----------------------------------------------------

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


# -- Step 5: Partition -----------------------------------------------------

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


# -- Step 6: Split + DataLoader -------------------------------------------

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
    # -- Try loading real data via fl_common.data --
    config = DataConfig.for_task("drug", data_path)
    if data_path:
        try:
            X, y, meta = load_dataset(config)
            result = validate_tabular(X, y, config)
            if not result.is_valid:
                raise ValueError(f"Data validation failed: {result.errors}")
            logger.info(f"[Load] Loaded real data from {data_path}: {len(X)} samples")

            if local_mode:
                X, y = clean_data(X, y)
                X, feat_mean, feat_std = normalize_features(X)
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
                    "active_rate": float(y.mean()),
                    "feature_mean": feat_mean,
                    "feature_std": feat_std,
                    "source": meta.get("source", data_path),
                }
                return loaders, metadata
        except FileNotFoundError:
            if local_mode:
                raise FileNotFoundError(
                    f"No data found at {data_path}. Provide data or set synthetic=True. "
                    f"Run 'python ingest.py --task drug --input <your_data>' to ingest data."
                )
            logger.info("[Load] Real data not found, falling back to synthetic")
            X, y = None, None
    else:
        X, y = None, None

    # -- Synthetic fallback --
    if X is None:
        total = 5000 if max_samples == 0 else max_samples
        logger.info(f"[Load] Generating {total} synthetic molecular fingerprints")
        X, y = _generate_drug(total, seed=seed)

    # -- Validate --
    validate_data(X, y)

    # -- Clean --
    X, y = clean_data(X, y)

    # -- Normalize --
    X, feat_mean, feat_std = normalize_features(X)

    # -- Partition --
    logger.info(f"[Partition] {partition_type} (alpha={alpha}, clients={num_clients})")
    if partition_type == "label_skew":
        client_indices = partition_label_skew(y, num_clients, alpha, seed)
    else:
        client_indices = partition_iid(y, num_clients, seed)

    # -- Split -> DataLoaders --
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
        n_active = int(yc.sum())
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}, "
                     f"active={n_active}/{n} ({yc.mean():.4f})")

    metadata = {
        "num_clients": len(loaders),
        "input_dim": 200,
        "total_samples": len(X),
        "active_rate": float(y.mean()),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
    }
    return loaders, metadata
