"""
Hospital Readmission Prediction Data Pipeline
===============================================
Pipeline: Load -> Validate -> Clean -> Normalize -> Partition -> Split -> DataLoader

Synthetic tabular: 20 features (demographics, diagnoses, procedures),
binary classification (readmitted within 30 days or not).
Use case: healthcare quality, insurance.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple
import logging

from fl_common.data import DataConfig, load_dataset, validate_tabular, partition_local

logger = logging.getLogger("pipeline.readmission")


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- Step 1: Load --------------------------------------------------------

def _generate_readmission(n, n_features=20, readmit_rate=0.18, seed=42):
    """Synthetic hospital discharge data.

    Feature layout (before normalization):
      0   age                    (18-95)
      1   gender                 (0/1)
      2   num_prior_admissions   (0-10)
      3   length_of_stay_days    (1-30)
      4   num_diagnoses          (1-15)
      5   num_procedures         (0-8)
      6   num_medications        (1-20)
      7   has_diabetes           (0/1)
      8   has_heart_failure      (0/1)
      9   has_copd               (0/1)
      10  has_renal_disease      (0/1)
      11  charlson_index         (0-10)
      12  discharge_disposition  (0=home, 1=SNF, 2=rehab) encoded as float
      13  payer_code             (0=private, 1=medicare, 2=medicaid) float
      14  emergency_admission    (0/1)
      15  lab_abnormal_count     (0-10)
      16  hba1c                  (4.0-14.0 %)
      17  creatinine             (0.5-6.0 mg/dL)
      18  hemoglobin             (6-18 g/dL)
      19  bmi                    (15-50)
    """
    rng = np.random.RandomState(seed)

    X = np.zeros((n, n_features), dtype=np.float32)
    X[:, 0] = rng.uniform(18, 95, n)            # age
    X[:, 1] = (rng.rand(n) < 0.48).astype(np.float32)  # gender
    X[:, 2] = rng.poisson(1.5, n).clip(0, 10)   # prior admissions
    X[:, 3] = rng.exponential(5, n).clip(1, 30)  # LOS
    X[:, 4] = rng.poisson(4, n).clip(1, 15)      # diagnoses
    X[:, 5] = rng.poisson(2, n).clip(0, 8)       # procedures
    X[:, 6] = rng.poisson(6, n).clip(1, 20)      # medications
    X[:, 7] = (rng.rand(n) < 0.25).astype(np.float32)   # diabetes
    X[:, 8] = (rng.rand(n) < 0.15).astype(np.float32)   # heart failure
    X[:, 9] = (rng.rand(n) < 0.10).astype(np.float32)   # COPD
    X[:, 10] = (rng.rand(n) < 0.12).astype(np.float32)  # renal disease
    X[:, 11] = rng.poisson(2, n).clip(0, 10).astype(np.float32)  # Charlson
    X[:, 12] = rng.choice([0, 1, 2], n, p=[0.6, 0.25, 0.15]).astype(np.float32)
    X[:, 13] = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25]).astype(np.float32)
    X[:, 14] = (rng.rand(n) < 0.35).astype(np.float32)  # emergency
    X[:, 15] = rng.poisson(2, n).clip(0, 10).astype(np.float32)  # lab abnormals
    X[:, 16] = rng.normal(6.5, 1.5, n).clip(4.0, 14.0)  # HbA1c
    X[:, 17] = rng.exponential(1.2, n).clip(0.5, 6.0)    # creatinine
    X[:, 18] = rng.normal(12, 2, n).clip(6, 18)           # hemoglobin
    X[:, 19] = rng.normal(28, 6, n).clip(15, 50)          # BMI

    # Readmission risk score
    risk = (
        0.01 * (X[:, 0] - 60)           # older age
        + 0.15 * X[:, 2]                 # prior admissions
        + 0.05 * (X[:, 3] - 5)          # longer stay
        + 0.08 * X[:, 4]                 # more diagnoses
        + 0.2 * X[:, 8]                  # heart failure
        + 0.15 * X[:, 10]                # renal disease
        + 0.1 * X[:, 11]                 # Charlson
        + 0.12 * X[:, 14]                # emergency admission
        + 0.06 * X[:, 15]                # lab abnormals
        + rng.randn(n) * 0.8
    )
    threshold = np.quantile(risk, 1.0 - readmit_rate)
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
                f"readmitted={n_pos} ({pos_rate:.4f})")

    if nan_count > 0 or inf_count > 0:
        logger.warning(f"[Validate] Found {nan_count} NaN and {inf_count} Inf values")
    if pos_rate < 0.01:
        logger.warning(f"[Validate] Very low readmission rate ({pos_rate:.5f}), may affect training")
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
    config = DataConfig.for_task("readmission", data_path)
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
                    "readmission_rate": float(y.mean()),
                    "feature_mean": feat_mean,
                    "feature_std": feat_std,
                    "source": meta.get("source", data_path),
                }
                return loaders, metadata
        except FileNotFoundError:
            if local_mode:
                raise FileNotFoundError(
                    f"No data found at {data_path}. Provide data or set synthetic=True. "
                    f"Run 'python ingest.py --task readmission --input <your_data>' to ingest data."
                )
            logger.info("[Load] Real data not found, falling back to synthetic")
            X, y = None, None
    else:
        X, y = None, None

    # -- Synthetic fallback --
    if X is None:
        total = 6000 if max_samples == 0 else max_samples
        logger.info(f"[Load] Generating {total} synthetic hospital discharge records")
        X, y = _generate_readmission(total, seed=seed)

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
        n_readmit = int(yc.sum())
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}, "
                     f"readmitted={n_readmit}/{n} ({yc.mean():.4f})")

    metadata = {
        "num_clients": len(loaders),
        "input_dim": 20,
        "total_samples": len(X),
        "readmission_rate": float(y.mean()),
        "feature_mean": feat_mean,
        "feature_std": feat_std,
    }
    return loaders, metadata
