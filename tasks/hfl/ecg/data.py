"""
ECG Data Pipeline
=================
Pipeline: Load → Validate → Clean → Normalize → Partition → Split → DataLoader

Synthetic 12-lead ECG: (N, 250, 12) with binary arrhythmia labels.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple
import logging

from fl_common.data import DataConfig, load_dataset, validate_tabular, partition_local

logger = logging.getLogger("pipeline.ecg")


class ECGDataset(Dataset):
    """12-lead ECG: (N, 250, 12) with binary labels."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Step 1: Load ──────────────────────────────────────────────────────

def _generate_ecg(n, n_leads=12, seq_len=250, seed=42):
    """Synthetic 12-lead ECG with subtle temporal pattern for arrhythmia.
    Signal is a periodic pattern in the middle timesteps (not a simple mean shift),
    so non-IID partitioning doesn't create trivially separable splits.
    """
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) > 0.7).astype(np.float32)  # ~30% positive (arrhythmia)
    X = rng.randn(n, seq_len, n_leads).astype(np.float32)
    # Add subtle temporal pattern: positive class has sinusoidal component
    # in timesteps 80-170 across leads 0-5 (requires temporal learning, not just mean)
    t = np.linspace(0, 4 * np.pi, 90)
    pattern = 0.3 * np.sin(t)  # subtle amplitude
    pos_idx = np.where(y == 1)[0]
    for i in pos_idx:
        # Each positive sample gets a slightly different phase (per-sample noise)
        phase = rng.uniform(0, np.pi)
        for lead in range(6):
            X[i, 80:170, lead] += pattern * np.cos(phase + lead * 0.5)
    return X, y


# ── Step 2: Validate ─────────────────────────────────────────────────

def validate_data(X: np.ndarray, y: np.ndarray):
    assert X.ndim == 3, f"Expected 3D input (N, T, leads), got shape {X.shape}"
    assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    pos_rate = y.mean()
    logger.info(f"[Validate] samples={len(X)}, shape={X.shape}, "
                f"NaN={nan_count}, Inf={inf_count}, pos_rate={pos_rate:.3f}")

    if nan_count > 0 or inf_count > 0:
        logger.warning(f"[Validate] Found {nan_count} NaN and {inf_count} Inf values")
    return nan_count, inf_count


# ── Step 3: Clean ────────────────────────────────────────────────────

def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_mask = np.isfinite(X).all(axis=(1, 2))
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"[Clean] Dropped {n_dropped} samples with NaN/Inf")
        X, y = X[valid_mask], y[valid_mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


# ── Step 4: Normalize ────────────────────────────────────────────────

def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-lead z-score normalization across all timesteps."""
    flat = X.reshape(-1, X.shape[2])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std
    logger.info(f"[Normalize] {X.shape[2]} leads, "
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
    batch_size: int = 32,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int = 0,
    seed: int = 42,
    local_mode: bool = False,
) -> Tuple[Dict, Dict]:
    # ── Try loading real data via fl_common.data ──
    config = DataConfig.for_task("ecg", data_path)
    if data_path:
        try:
            X, y, meta = load_dataset(config)
            result = validate_tabular(X, y, config)
            if not result.is_valid:
                raise ValueError(f"Data validation failed: {result.errors}")
            logger.info(f"[Load] Loaded real data from {data_path}: {len(X)} samples")

            if local_mode:
                X, y = clean_data(X, y)
                X, lead_mean, lead_std = normalize_features(X)
                splits = partition_local(X, y, val_ratio, test_ratio, seed)
                loaders = {0: {
                    "train": DataLoader(ECGDataset(*splits["train"]), batch_size=batch_size, shuffle=True),
                    "val": DataLoader(ECGDataset(*splits["val"]), batch_size=batch_size),
                    "test": DataLoader(ECGDataset(*splits["test"]), batch_size=batch_size),
                }}
                metadata = {
                    "num_clients": 1,
                    "input_dim": X.shape[-1],
                    "seq_len": X.shape[1] if X.ndim == 3 else 250,
                    "total_samples": len(X),
                    "lead_mean": lead_mean,
                    "lead_std": lead_std,
                    "source": meta.get("source", data_path),
                }
                return loaders, metadata
            # else: fall through to partition logic below with real data
        except FileNotFoundError:
            if local_mode:
                raise FileNotFoundError(
                    f"No data found at {data_path}. Provide data or set synthetic=True. "
                    f"Run 'python ingest.py --task ecg --input <your_data>' to ingest data."
                )
            logger.info("[Load] Real data not found, falling back to synthetic")
            X, y = None, None
    else:
        X, y = None, None

    # ── Synthetic fallback ──
    if X is None:
        total = 2000 if max_samples == 0 else max_samples
        logger.info(f"[Load] Generating {total} synthetic ECG samples")
        X, y = _generate_ecg(total, seed=seed)

    # ── Validate ──
    validate_data(X, y)

    # ── Clean ──
    X, y = clean_data(X, y)

    # ── Normalize ──
    X, lead_mean, lead_std = normalize_features(X)

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
            "train": DataLoader(ECGDataset(Xc[:n_train], yc[:n_train]), batch_size=batch_size, shuffle=True),
            "val": DataLoader(ECGDataset(Xc[n_train:n_train+n_val], yc[n_train:n_train+n_val]), batch_size=batch_size),
            "test": DataLoader(ECGDataset(Xc[n_train+n_val:], yc[n_train+n_val:]), batch_size=batch_size),
        }
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}, "
                     f"pos_rate={yc.mean():.3f}")

    metadata = {
        "num_clients": len(loaders),
        "input_dim": 12,
        "seq_len": 250,
        "total_samples": len(X),
        "lead_mean": lead_mean,
        "lead_std": lead_std,
    }
    return loaders, metadata
