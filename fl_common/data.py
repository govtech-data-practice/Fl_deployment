"""
FL Data Pipeline — Production-grade data management for federated learning.

Each client (hospital/agency) ingests, validates, and registers its own data locally.
The server never sees raw data — only metadata manifests.

Components:
    DataConfig      — typed configuration for a task's data requirements
    DataManifest    — describes what data a client has (schema, counts, checksums)
    validate()      — gates training: returns errors, not warnings
    load_dataset()  — loads data from standardized file format
    generate_manifest() — creates manifest from ingested data
"""

import os
import json
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger("fl.data")

# ── Standardized data directory layout ──────────────────────────────
# ~/fl-deploy/data/<task>/
#   manifest.json       — DataManifest (schema, counts, checksums)
#   data.npz            — features (X) and labels (y)
#   OR data.csv         — tabular data with header row
#   OR images/           — image directory (for chest_xray, satellite)
#       metadata.csv    — image paths and labels

SUPPORTED_FORMATS = ("npz", "csv", "images")


# ── DataConfig ──────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Typed configuration for a task's data requirements.
    Replaces environment variable configuration.
    """
    task: str
    data_dir: str                           # path to task data directory
    input_dim: int = 0                      # feature dimension (tabular/time series)
    num_classes: int = 2                     # output classes
    seq_len: int = 0                        # sequence length (time series)
    task_type: str = "binary"               # binary, multiclass, multilabel, regression, reconstruction
    batch_size: int = 32
    max_samples: int = 0                    # 0 = use all
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    image_size: int = 224                   # for image tasks
    synthetic: bool = False                 # explicitly request synthetic data
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest_path: str, **overrides) -> "DataConfig":
        """Load config from a manifest file."""
        manifest = DataManifest.load(manifest_path)
        cfg = cls(
            task=manifest.task,
            data_dir=str(Path(manifest_path).parent),
            input_dim=manifest.schema.get("input_dim", 0),
            num_classes=manifest.schema.get("num_classes", 2),
            seq_len=manifest.schema.get("seq_len", 0),
            task_type=manifest.schema.get("task_type", "binary"),
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def for_task(cls, task: str, data_dir: str = "", synthetic: bool = False) -> "DataConfig":
        """Create config from task defaults. Used when no manifest exists."""
        defaults = TASK_DEFAULTS.get(task, {})
        return cls(
            task=task,
            data_dir=data_dir or f"/data/{task}",
            synthetic=synthetic,
            **defaults,
        )


# Per-task default configurations
TASK_DEFAULTS = {
    "fraud":       {"input_dim": 30, "num_classes": 2, "task_type": "binary", "max_samples": 50000},
    "sepsis":      {"input_dim": 14, "num_classes": 2, "task_type": "binary", "seq_len": 48},
    "ecg":         {"input_dim": 12, "num_classes": 2, "task_type": "binary", "seq_len": 250},
    "anomaly":     {"input_dim": 40, "num_classes": 2, "task_type": "reconstruction", "max_samples": 8000},
    "mortality":   {"input_dim": 25, "num_classes": 2, "task_type": "binary", "max_samples": 6000},
    "drug":        {"input_dim": 200, "num_classes": 2, "task_type": "binary", "max_samples": 5000},
    "readmission": {"input_dim": 20, "num_classes": 2, "task_type": "binary", "max_samples": 6000},
    "satellite":   {"input_dim": 3, "num_classes": 5, "task_type": "multiclass", "image_size": 64, "max_samples": 3000},
    "chest":       {"input_dim": 3, "num_classes": 14, "task_type": "multilabel", "image_size": 224},
    "transfer":    {"input_dim": 3, "num_classes": 14, "task_type": "multilabel", "image_size": 224},
}


# ── DataManifest ────────────────────────────────────────────────────

@dataclass
class DataManifest:
    """Describes a client's local dataset. Shared with server for coordination,
    but never contains raw data.
    """
    task: str
    format: str                             # npz, csv, images
    num_samples: int
    schema: Dict[str, Any]                  # input_dim, num_classes, seq_len, columns, etc.
    label_distribution: Dict[str, float]    # class -> proportion
    checksum: str                           # SHA-256 of the data file
    client_id: str = ""                     # hospital/agency identifier
    version: str = "1.0"
    data_path: str = ""                     # relative path to data file within data_dir
    created: str = ""                       # ISO 8601 timestamp

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Manifest saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "DataManifest":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def validate_against(self, config: DataConfig) -> List[str]:
        """Check manifest compatibility with a DataConfig. Returns list of errors."""
        errors = []
        if self.task != config.task:
            errors.append(f"Task mismatch: manifest={self.task}, config={config.task}")
        if config.input_dim and self.schema.get("input_dim", 0) != config.input_dim:
            errors.append(f"input_dim mismatch: manifest={self.schema.get('input_dim')}, config={config.input_dim}")
        if self.num_samples == 0:
            errors.append("Dataset is empty (0 samples)")
        if self.num_samples < 10:
            errors.append(f"Dataset too small ({self.num_samples} samples, need ≥10)")
        return errors


# ── Validation ──────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of data validation. Training proceeds only if is_valid is True."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

    def __str__(self):
        status = "PASS" if self.is_valid else "FAIL"
        lines = [f"Validation: {status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def validate_tabular(X: np.ndarray, y: np.ndarray, config: DataConfig) -> ValidationResult:
    """Validate tabular/time-series data. Blocks training on errors."""
    errors, warnings, stats = [], [], {}

    # Shape checks
    if X.ndim not in (2, 3):
        errors.append(f"Expected 2D or 3D input, got {X.ndim}D shape {X.shape}")
    if len(X) != len(y):
        errors.append(f"X/y length mismatch: {len(X)} vs {len(y)}")
    if len(X) == 0:
        errors.append("Dataset is empty")
    if config.input_dim and X.ndim >= 2 and X.shape[-1] != config.input_dim:
        errors.append(f"Feature dim mismatch: got {X.shape[-1]}, expected {config.input_dim}")
    if config.seq_len and X.ndim == 3 and X.shape[1] != config.seq_len:
        errors.append(f"Sequence length mismatch: got {X.shape[1]}, expected {config.seq_len}")

    # Data quality
    if len(X) > 0:
        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        nan_ratio = nan_count / X.size if X.size > 0 else 0

        if nan_ratio > 0.5:
            errors.append(f"More than 50% NaN values ({nan_count}/{X.size})")
        elif nan_count > 0:
            warnings.append(f"{nan_count} NaN values ({nan_ratio:.4f} ratio)")
        if inf_count > 0:
            warnings.append(f"{inf_count} Inf values")

        # Label checks
        unique_labels = np.unique(y[~np.isnan(y)]) if y.dtype in (np.float32, np.float64) else np.unique(y)
        if config.task_type == "binary":
            if len(unique_labels) < 2:
                errors.append(f"Binary task but only {len(unique_labels)} unique label(s): {unique_labels}")
            pos_rate = float(np.mean(y > 0.5)) if y.ndim == 1 else 0
            if pos_rate < 0.01 or pos_rate > 0.99:
                warnings.append(f"Extreme class imbalance: positive rate = {pos_rate:.4f}")
            stats["positive_rate"] = round(pos_rate, 4)
        elif config.task_type == "multiclass":
            if len(unique_labels) < 2:
                errors.append(f"Multiclass task but only {len(unique_labels)} unique label(s)")
            stats["num_classes_found"] = len(unique_labels)

        stats.update({
            "num_samples": len(X),
            "shape": list(X.shape),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "dtype": str(X.dtype),
        })

    is_valid = len(errors) == 0
    result = ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, stats=stats)
    if is_valid:
        logger.info("Data validation PASSED: %d samples, shape %s", len(X), X.shape)
    else:
        logger.error("Data validation FAILED:\n%s", result)
    return result


def validate_images(image_dir: str, metadata_csv: str, config: DataConfig) -> ValidationResult:
    """Validate image dataset. Checks directory structure and metadata CSV."""
    errors, warnings, stats = [], [], {}

    if not os.path.isdir(image_dir):
        errors.append(f"Image directory not found: {image_dir}")
        return ValidationResult(False, errors, warnings, stats)

    if not os.path.exists(metadata_csv):
        errors.append(f"Metadata CSV not found: {metadata_csv}")
        return ValidationResult(False, errors, warnings, stats)

    import pandas as pd
    df = pd.read_csv(metadata_csv)

    required_cols = {"image_path", "label"} if config.task_type != "multilabel" else {"image_path"}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(f"Missing columns in metadata CSV: {missing}")

    if "image_path" in df.columns:
        sample_paths = df["image_path"].head(20)
        missing_files = [p for p in sample_paths if not os.path.exists(os.path.join(image_dir, p))]
        if missing_files:
            errors.append(f"{len(missing_files)}/{len(sample_paths)} sampled images not found")

    stats["num_images"] = len(df)
    stats["columns"] = list(df.columns)

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, stats)


# ── File Loading ────────────────────────────────────────────────────

def load_dataset(config: DataConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load dataset from standardized file format.

    Returns (X, y, metadata) where:
        X: features array (2D tabular, 3D time-series, or image paths)
        y: labels array
        metadata: dict with schema info
    """
    data_dir = Path(config.data_dir)
    manifest_path = data_dir / "manifest.json"

    # If manifest exists, validate it
    if manifest_path.exists():
        manifest = DataManifest.load(str(manifest_path))
        errs = manifest.validate_against(config)
        if errs:
            raise ValueError(f"Manifest validation failed: {errs}")
        logger.info("Loaded manifest: %s, %d samples", manifest.task, manifest.num_samples)

    # Try loading in order of preference
    npz_path = data_dir / "data.npz"
    csv_path = data_dir / "data.csv"

    if npz_path.exists():
        return _load_npz(str(npz_path), config)
    elif csv_path.exists():
        return _load_csv(str(csv_path), config)
    elif config.synthetic:
        logger.info("No data files found, synthetic mode requested")
        raise FileNotFoundError(f"Synthetic mode — caller should generate data")
    else:
        raise FileNotFoundError(
            f"No data found at {data_dir}. Expected data.npz or data.csv. "
            f"Run 'python ingest.py --task {config.task} --input <your_data>' to ingest data."
        )


def _load_npz(path: str, config: DataConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load from NPZ format (features + labels)."""
    data = np.load(path, allow_pickle=False)

    # Support common key names
    X_keys = ["X", "features", "data", "x"]
    y_keys = ["y", "labels", "targets", "label"]

    X = y = None
    for k in X_keys:
        if k in data:
            X = data[k]
            break
    for k in y_keys:
        if k in data:
            y = data[k]
            break

    if X is None:
        raise ValueError(f"NPZ file missing features. Keys found: {list(data.keys())}. Expected one of: {X_keys}")
    if y is None:
        raise ValueError(f"NPZ file missing labels. Keys found: {list(data.keys())}. Expected one of: {y_keys}")

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Apply max_samples
    if config.max_samples and len(X) > config.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), config.max_samples, replace=False)
        X, y = X[idx], y[idx]

    metadata = {
        "source": path,
        "format": "npz",
        "input_dim": X.shape[-1] if X.ndim >= 2 else 0,
        "num_samples": len(X),
    }
    return X, y, metadata


def _load_csv(path: str, config: DataConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load from CSV format. Last column or 'label'/'target' column is the label."""
    import pandas as pd
    df = pd.read_csv(path)

    # Find label column
    label_col = None
    for candidate in ["label", "target", "y", "class", "Label", "Target"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df.columns[-1]
        logger.info("No explicit label column found, using last column: %s", label_col)

    y = df[label_col].values.astype(np.float32)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values.astype(np.float32)

    if config.max_samples and len(X) > config.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), config.max_samples, replace=False)
        X, y = X[idx], y[idx]

    metadata = {
        "source": path,
        "format": "csv",
        "input_dim": X.shape[1],
        "num_samples": len(X),
        "columns": list(df.columns),
    }
    return X, y, metadata


# ── Manifest Generation ─────────────────────────────────────────────

def compute_checksum(path: str) -> str:
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(
    data_dir: str,
    task: str,
    client_id: str = "",
) -> DataManifest:
    """Generate a manifest from ingested data files."""
    from datetime import datetime

    data_dir = Path(data_dir)
    npz_path = data_dir / "data.npz"
    csv_path = data_dir / "data.csv"

    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=False)
        X_key = next((k for k in ["X", "features", "data", "x"] if k in data), None)
        y_key = next((k for k in ["y", "labels", "targets", "label"] if k in data), None)
        X, y = data[X_key], data[y_key]
        fmt = "npz"
        data_file = "data.npz"
        checksum = compute_checksum(str(npz_path))

    elif csv_path.exists():
        import pandas as pd
        df = pd.read_csv(str(csv_path))
        label_col = next((c for c in ["label", "target", "y", "class"] if c in df.columns), df.columns[-1])
        y = df[label_col].values
        X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
        fmt = "csv"
        data_file = "data.csv"
        checksum = compute_checksum(str(csv_path))
    else:
        raise FileNotFoundError(f"No data.npz or data.csv in {data_dir}")

    # Compute label distribution
    unique, counts = np.unique(y.astype(int) if y.max() < 100 else (y > 0.5).astype(int), return_counts=True)
    label_dist = {str(u): round(float(c / len(y)), 4) for u, c in zip(unique, counts)}

    # Schema
    defaults = TASK_DEFAULTS.get(task, {})
    schema = {
        "input_dim": int(X.shape[-1]) if X.ndim >= 2 else 0,
        "num_classes": defaults.get("num_classes", len(unique)),
        "seq_len": int(X.shape[1]) if X.ndim == 3 else 0,
        "task_type": defaults.get("task_type", "binary"),
        "dtype": str(X.dtype),
    }

    manifest = DataManifest(
        task=task,
        format=fmt,
        num_samples=len(X),
        schema=schema,
        label_distribution=label_dist,
        checksum=checksum,
        client_id=client_id,
        version="1.0",
        data_path=data_file,
        created=datetime.utcnow().isoformat() + "Z",
    )

    manifest.save(str(data_dir / "manifest.json"))
    return manifest


# ── Partition (client-side, no global view) ─────────────────────────

def partition_local(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split a single client's local data into train/val/test.
    Unlike the demo pipeline, this does NOT partition across clients —
    each client only sees its own data.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    indices = rng.permutation(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
    }
