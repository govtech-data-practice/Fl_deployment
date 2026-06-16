"""
Land Use Classification from Satellite Data Pipeline
======================================================
Pipeline: Load -> Validate -> Clean -> Normalize -> Partition -> Split -> DataLoader

Synthetic image-like data: 64x64x3 patches, 5-class classification.
Classes: 0=urban, 1=forest, 2=water, 3=agriculture, 4=barren.
Use case: defence, environment, agriculture.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple
import logging

from fl_common.data import DataConfig, load_dataset, validate_tabular, partition_local

logger = logging.getLogger("pipeline.satellite")

CLASS_NAMES = ["urban", "forest", "water", "agriculture", "barren"]
NUM_CLASSES = 5


class ImageDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 64, 64, 3) -> (N, 3, 64, 64) for PyTorch conv layers
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -- Step 1: Load --------------------------------------------------------

def _generate_satellite(n, img_size=64, n_channels=3, num_classes=5, seed=42):
    """Synthetic satellite image patches.

    Each class has a distinctive spectral/texture signature:
      - urban:       high-frequency texture, grey-ish (R~G~B moderate)
      - forest:      green-dominant, mid-frequency texture
      - water:       blue-dominant, smooth (low frequency)
      - agriculture: green-yellow, periodic stripe pattern
      - barren:      brown/tan (R>G>B), low texture
    """
    rng = np.random.RandomState(seed)
    y = rng.randint(0, num_classes, size=n).astype(np.int64)
    X = np.zeros((n, img_size, img_size, n_channels), dtype=np.float32)

    for cls in range(num_classes):
        mask = y == cls
        count = int(mask.sum())
        if count == 0:
            continue

        if cls == 0:  # urban
            base = rng.uniform(0.4, 0.6, size=(count, img_size, img_size, n_channels))
            noise = rng.randn(count, img_size, img_size, n_channels) * 0.15
            X[mask] = base + noise

        elif cls == 1:  # forest
            base = np.zeros((count, img_size, img_size, n_channels), dtype=np.float32)
            base[:, :, :, 0] = rng.uniform(0.1, 0.3, size=(count, img_size, img_size))  # R low
            base[:, :, :, 1] = rng.uniform(0.4, 0.8, size=(count, img_size, img_size))  # G high
            base[:, :, :, 2] = rng.uniform(0.05, 0.2, size=(count, img_size, img_size))  # B low
            noise = rng.randn(count, img_size, img_size, n_channels) * 0.08
            X[mask] = base + noise

        elif cls == 2:  # water
            base = np.zeros((count, img_size, img_size, n_channels), dtype=np.float32)
            base[:, :, :, 0] = rng.uniform(0.0, 0.15, size=(count, img_size, img_size))
            base[:, :, :, 1] = rng.uniform(0.1, 0.3, size=(count, img_size, img_size))
            base[:, :, :, 2] = rng.uniform(0.4, 0.8, size=(count, img_size, img_size))
            noise = rng.randn(count, img_size, img_size, n_channels) * 0.04
            X[mask] = base + noise

        elif cls == 3:  # agriculture
            base = np.zeros((count, img_size, img_size, n_channels), dtype=np.float32)
            # Periodic stripe pattern
            rows = np.arange(img_size).reshape(1, img_size, 1, 1)
            stripe = (0.1 * np.sin(2 * np.pi * rows / 8)).astype(np.float32)
            base[:, :, :, 0] = 0.4 + stripe[:, :, :, 0]
            base[:, :, :, 1] = 0.6 + stripe[:, :, :, 0]
            base[:, :, :, 2] = 0.15
            noise = rng.randn(count, img_size, img_size, n_channels) * 0.06
            X[mask] = base + noise

        elif cls == 4:  # barren
            base = np.zeros((count, img_size, img_size, n_channels), dtype=np.float32)
            base[:, :, :, 0] = rng.uniform(0.5, 0.7, size=(count, img_size, img_size))  # R high
            base[:, :, :, 1] = rng.uniform(0.35, 0.5, size=(count, img_size, img_size))  # G mid
            base[:, :, :, 2] = rng.uniform(0.15, 0.3, size=(count, img_size, img_size))  # B low
            noise = rng.randn(count, img_size, img_size, n_channels) * 0.05
            X[mask] = base + noise

    # Clip to valid pixel range [0, 1]
    X = np.clip(X, 0.0, 1.0)
    return X, y


# -- Step 2: Validate -----------------------------------------------------

def validate_data(X: np.ndarray, y: np.ndarray):
    assert X.ndim == 4, f"Expected 4D input (N, H, W, C), got shape {X.shape}"
    assert X.shape[1:] == (64, 64, 3), f"Expected (64,64,3) patches, got {X.shape[1:]}"
    assert len(X) == len(y), f"X/y length mismatch: {len(X)} vs {len(y)}"

    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    class_counts = {CLASS_NAMES[c]: int((y == c).sum()) for c in range(NUM_CLASSES)}
    logger.info(f"[Validate] samples={len(X)}, shape={X.shape[1:]}, "
                f"NaN={nan_count}, Inf={inf_count}")
    logger.info(f"[Validate] Class distribution: {class_counts}")

    if nan_count > 0 or inf_count > 0:
        logger.warning(f"[Validate] Found {nan_count} NaN and {inf_count} Inf values")
    return nan_count, inf_count


# -- Step 3: Clean --------------------------------------------------------

def clean_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Flatten to 2D for per-sample check, then reshape back
    X_flat = X.reshape(len(X), -1)
    valid_mask = np.isfinite(X_flat).all(axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info(f"[Clean] Dropped {n_dropped} samples with NaN/Inf")
        X, y = X[valid_mask], y[valid_mask]
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    return X, y


# -- Step 4: Normalize ----------------------------------------------------

def normalize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel normalization across all pixels."""
    # X shape: (N, 64, 64, 3)
    mean = X.mean(axis=(0, 1, 2))  # shape (3,)
    std = X.std(axis=(0, 1, 2))    # shape (3,)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std
    logger.info(f"[Normalize] {X.shape[3]} channels, "
                f"mean=[{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}], "
                f"std=[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    return X_norm.astype(np.float32), mean, std


# -- Step 5: Partition -----------------------------------------------------

def partition_iid(y, num_clients, seed):
    rng = np.random.RandomState(seed)
    return {i: s for i, s in enumerate(np.array_split(rng.permutation(len(y)), num_clients))}


def partition_label_skew(y, num_clients, alpha, seed):
    """Dirichlet-based label skew for multi-class data."""
    rng = np.random.RandomState(seed)
    partitions = {i: [] for i in range(num_clients)}
    for c in range(NUM_CLASSES):
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
    config = DataConfig.for_task("satellite", data_path)
    if data_path:
        try:
            X, y, meta = load_dataset(config)
            result = validate_tabular(X, y, config)
            if not result.is_valid:
                raise ValueError(f"Data validation failed: {result.errors}")
            logger.info(f"[Load] Loaded real data from {data_path}: {len(X)} samples")

            if local_mode:
                X, y = clean_data(X, y)
                X, chan_mean, chan_std = normalize_features(X)
                splits = partition_local(X, y, val_ratio, test_ratio, seed)
                loaders = {0: {
                    "train": DataLoader(ImageDataset(*splits["train"]), batch_size=batch_size, shuffle=True),
                    "val": DataLoader(ImageDataset(*splits["val"]), batch_size=batch_size),
                    "test": DataLoader(ImageDataset(*splits["test"]), batch_size=batch_size),
                }}
                metadata = {
                    "num_clients": 1,
                    "input_shape": (3, 64, 64),
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "total_samples": len(X),
                    "channel_mean": chan_mean,
                    "channel_std": chan_std,
                    "source": meta.get("source", data_path),
                }
                return loaders, metadata
        except FileNotFoundError:
            if local_mode:
                raise FileNotFoundError(
                    f"No data found at {data_path}. Provide data or set synthetic=True. "
                    f"Run 'python ingest.py --task satellite --input <your_data>' to ingest data."
                )
            logger.info("[Load] Real data not found, falling back to synthetic")
            X, y = None, None
    else:
        X, y = None, None

    # -- Synthetic fallback --
    if X is None:
        total = 3000 if max_samples == 0 else max_samples
        logger.info(f"[Load] Generating {total} synthetic 64x64x3 satellite patches")
        X, y = _generate_satellite(total, seed=seed)

    # -- Validate --
    validate_data(X, y)

    # -- Clean --
    X, y = clean_data(X, y)

    # -- Normalize --
    X, chan_mean, chan_std = normalize_features(X)

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
            "train": DataLoader(ImageDataset(Xc[:n_train], yc[:n_train]), batch_size=batch_size, shuffle=True),
            "val": DataLoader(ImageDataset(Xc[n_train:n_train+n_val], yc[n_train:n_train+n_val]), batch_size=batch_size),
            "test": DataLoader(ImageDataset(Xc[n_train+n_val:], yc[n_train+n_val:]), batch_size=batch_size),
        }
        class_dist = {CLASS_NAMES[c]: int((yc == c).sum()) for c in range(NUM_CLASSES)}
        logger.info(f"[Split] Client {cid}: train={n_train}, val={n_val}, test={n_test}, "
                     f"classes={class_dist}")

    metadata = {
        "num_clients": len(loaders),
        "input_shape": (3, 64, 64),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "total_samples": len(X),
        "channel_mean": chan_mean,
        "channel_std": chan_std,
    }
    return loaders, metadata
