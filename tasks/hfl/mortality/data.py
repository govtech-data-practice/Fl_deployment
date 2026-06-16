"""
ICU Mortality Prediction Data Pipeline
========================================
Pipeline: Load -> Validate -> Clean -> Normalize -> Partition -> Split -> DataLoader

Synthetic tabular: 25 features (vitals, labs, demographics), binary classification.
Use case: hospital ICU, triage.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple
import logging

from fl_common.data import DataConfig, load_dataset, validate_tabular, partition_local

logger = logging.getLogger("pipeline.mortality")


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- Step 1: Load --------------------------------------------------------

def _generate_mortality(n, n_features=25, mortality_rate=0.12, seed=42):
    """Synthetic ICU patient data.

    Feature layout (before normalization):
      0       age              (40-90)
      1       gender           (0/1)
      2       heart_rate       (50-150 bpm)
      3       systolic_bp      (70-200 mmHg)
      4       diastolic_bp     (40-120 mmHg)
      5       resp_rate        (8-40 breaths/min)
      6       spo2             (80-100 %)
      7       temperature      (35-40 C)
      8       gcs_total        (3-15)
      9       bun              (5-80 mg/dL)
      10      creatinine       (0.3-8.0 mg/dL)
      11      wbc              (2-30 k/uL)
      12      hemoglobin       (5-18 g/dL)
      13      platelets        (20-500 k/uL)
      14      sodium           (125-155 mEq/L)
      15      potassium        (2.5-7.0 mEq/L)
      16      glucose          (50-400 mg/dL)
      17      lactate          (0.5-15 mmol/L)
      18      bilirubin        (0.2-20 mg/dL)
      19      albumin          (1.5-5.0 g/dL)
      20      pao2             (50-300 mmHg)
      21      fio2             (0.21-1.0)
      22      ph               (7.0-7.6)
      23      mechanical_vent  (0/1)
      24      vasopressor_use  (0/1)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)

    # Build realistic ranges via location/scale
    locs = np.array([65, 0.5, 90, 130, 75, 18, 95, 37.0,
                     11, 25, 1.5, 10, 12, 250, 140, 4.2,
                     150, 3.0, 2.0, 3.5, 150, 0.5, 7.35,
                     0.3, 0.2], dtype=np.float32)
    scales = np.array([12, 0.5, 25, 30, 18, 6, 4, 1.0,
                       3, 18, 1.5, 6, 3, 100, 6, 0.8,
                       80, 3.0, 3.0, 0.8, 60, 0.2, 0.1,
                       0.45, 0.4], dtype=np.float32)

    X = X * scales + locs

    # Determine mortality label using a latent risk score
    risk = (
        0.02 * (X[:, 0] - 65)            # age
        - 0.03 * (X[:, 8] - 11)          # lower GCS -> higher risk
        + 0.02 * (X[:, 17] - 3.0)        # lactate
        - 0.04 * (X[:, 6] - 95)          # low SpO2
        + 0.015 * (X[:, 24])             # vasopressor use
        + rng.randn(n) * 0.5
    )
    threshold = np.quantile(risk, 1.0 - mortality_rate)
    y = (risk >= threshold).astype(np.float32)

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
                f"died={n_pos} ({pos_rate:.4f})")

    if nan_count > 0 or inf_count > 0:
        logger.warning(f"[Validate] Found {nan_count} NaN and {inf_count} Inf values")
    if pos_rate < 0.01:
        logger.warning(f"[Validate] Very low mortality rate ({pos_rate:.5f}), may affect training")
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
    config = DataConfig.for_task("mortality", data_path)
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
                    "mortality_rate": float(y.mean()),
                    "feature_mean": feat_mean,
                    "feature_std": feat_std,
                    "source": meta.get("source", data_path),
                }
                return loaders, metadata
        except FileNotFoundError:
            if local_mode:
                raise FileNotFoundError(
                    f"No data found at {data_path}. Provide data or set synthetic=True. "
                    f"Run 'python ingest.py --task mortality --input <your_data>' to ingest data."
                )
            logger.info("[Load] Real data not found, falling back to synthetic")
            X, y = None, None
    else:
        X, y = None, None

    # -- Synthetic fallback --
    if X is None:
        total = 6000 if max_samples == 0 else max_samples
        logger.info(f"[Load] Generating {total} synthetic ICU patient records")
        X, y = _generate_mortality(total, seed=seed)

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
        n_died = int(yc.sum())
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}, "
                     f"died={n_died}/{n} ({yc.mean():.4f})")

    metadata = {
        "num_clients": len(loaders),
        "input_dim": 25,
        "total_samples": len(X),
        "mortality_rate": float(y.mean()),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
    }
    return loaders, metadata
