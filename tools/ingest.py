#!/usr/bin/env python3
"""
FL Data Ingestion CLI — Production data pipeline for federated learning.

Each hospital/agency runs this tool to ingest raw data into the standardized
format expected by the FL client. The tool validates, converts, and generates
a manifest. The server never sees raw data.

Usage:
    # Ingest tabular data (CSV)
    python ingest.py --task sepsis --input /path/to/hospital_data.csv

    # Ingest tabular data (NPZ)
    python ingest.py --task fraud --input /path/to/transactions.npz

    # Ingest with custom output directory
    python ingest.py --task mortality --input data.csv --output ~/fl-deploy/data/mortality

    # Ingest image dataset
    python ingest.py --task chest --input /path/to/images/ --metadata /path/to/labels.csv

    # Validate existing ingested data without re-ingesting
    python ingest.py --task sepsis --validate-only --output ~/fl-deploy/data/sepsis

    # Generate synthetic data for testing
    python ingest.py --task fraud --synthetic --num-samples 5000

    # Show manifest of ingested data
    python ingest.py --show-manifest ~/fl-deploy/data/sepsis

Examples for government use cases:
    # Hospital A ingests its own patient records
    python ingest.py --task sepsis --input /mnt/ehr/sepsis_cohort.csv --client-id hospital_a

    # Bank ingests transaction data
    python ingest.py --task fraud --input /data/warehouse/transactions_2026.csv --client-id bank_sg_01

    # Defence agency ingests satellite imagery
    python ingest.py --task satellite --input /data/sat/patches/ --metadata /data/sat/labels.csv
"""

import sys
import os
import argparse
import shutil
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fl_common.data import (
    DataConfig, DataManifest, ValidationResult,
    validate_tabular, validate_images,
    compute_checksum, generate_manifest,
    TASK_DEFAULTS, SUPPORTED_FORMATS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ingest")

DEFAULT_DATA_ROOT = os.path.expanduser("~/fl-deploy/data")


def parse_args():
    p = argparse.ArgumentParser(
        description="FL Data Ingestion — prepare local data for federated learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--task", required=False, help="Task name (sepsis, fraud, ecg, etc.)")
    p.add_argument("--input", "-i", help="Path to input data file (CSV/NPZ) or directory (images)")
    p.add_argument("--metadata", help="Metadata CSV for image datasets (image_path, label columns)")
    p.add_argument("--output", "-o", help=f"Output directory (default: {DEFAULT_DATA_ROOT}/<task>)")
    p.add_argument("--client-id", default="", help="Client identifier (e.g., hospital_a, bank_sg_01)")
    p.add_argument("--validate-only", action="store_true", help="Validate existing data without re-ingesting")
    p.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing")
    p.add_argument("--num-samples", type=int, default=2000, help="Number of synthetic samples (default: 2000)")
    p.add_argument("--show-manifest", nargs="?", const=".", help="Show manifest of ingested data at path")
    p.add_argument("--force", action="store_true", help="Overwrite existing ingested data")
    return p.parse_args()


def ingest_csv(input_path: str, output_dir: str, config: DataConfig) -> ValidationResult:
    """Ingest a CSV file into standardized NPZ format."""
    import pandas as pd

    logger.info("Reading CSV: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("  Rows: %d, Columns: %d", len(df), len(df.columns))
    logger.info("  Columns: %s", list(df.columns))

    # Find label column
    label_col = None
    for candidate in ["label", "target", "y", "class", "Label", "Target", "outcome"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df.columns[-1]
        logger.info("  No explicit label column, using last: '%s'", label_col)
    else:
        logger.info("  Label column: '%s'", label_col)

    y = df[label_col].values.astype(np.float32)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values.astype(np.float32)

    dropped = len(df.columns) - 1 - X.shape[1]
    if dropped > 0:
        logger.info("  Dropped %d non-numeric columns", dropped)

    # Validate
    result = validate_tabular(X, y, config)
    if not result.is_valid:
        logger.error("Validation FAILED — data not ingested")
        return result

    # Clean NaN/Inf
    nan_mask = np.isfinite(X).all(axis=1) if X.ndim == 2 else np.isfinite(X).all(axis=(1, 2))
    if not nan_mask.all():
        n_dropped = (~nan_mask).sum()
        logger.info("  Dropping %d rows with NaN/Inf", n_dropped)
        X, y = X[nan_mask], y[nan_mask]

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "data.npz")
    np.savez_compressed(out_path, X=X, y=y)
    logger.info("  Saved: %s (%d samples, %.1f MB)",
                out_path, len(X), os.path.getsize(out_path) / 1e6)

    return result


def ingest_npz(input_path: str, output_dir: str, config: DataConfig) -> ValidationResult:
    """Ingest an NPZ file — validate and copy to standard location."""
    logger.info("Reading NPZ: %s", input_path)
    data = np.load(input_path, allow_pickle=False)
    logger.info("  Keys: %s", list(data.keys()))

    # Find X and y
    X_keys = ["X", "features", "data", "x"]
    y_keys = ["y", "labels", "targets", "label"]
    X = next((data[k] for k in X_keys if k in data), None)
    y = next((data[k] for k in y_keys if k in data), None)

    if X is None or y is None:
        result = ValidationResult(False, [f"NPZ missing X or y. Keys: {list(data.keys())}"], [], {})
        logger.error("Validation FAILED — data not ingested")
        return result

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    result = validate_tabular(X, y, config)
    if not result.is_valid:
        logger.error("Validation FAILED — data not ingested")
        return result

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "data.npz")
    np.savez_compressed(out_path, X=X, y=y)
    logger.info("  Saved: %s (%d samples, %.1f MB)",
                out_path, len(X), os.path.getsize(out_path) / 1e6)

    return result


def ingest_images(input_dir: str, metadata_csv: str, output_dir: str, config: DataConfig) -> ValidationResult:
    """Ingest an image dataset — validate and set up directory structure."""
    result = validate_images(input_dir, metadata_csv, config)
    if not result.is_valid:
        logger.error("Validation FAILED — data not ingested")
        return result

    os.makedirs(output_dir, exist_ok=True)

    # Symlink image directory (don't copy — images are large)
    img_link = os.path.join(output_dir, "images")
    if os.path.exists(img_link):
        os.remove(img_link) if os.path.islink(img_link) else shutil.rmtree(img_link)
    os.symlink(os.path.abspath(input_dir), img_link)
    logger.info("  Linked images: %s -> %s", img_link, input_dir)

    # Copy metadata CSV
    shutil.copy2(metadata_csv, os.path.join(output_dir, "metadata.csv"))
    logger.info("  Copied metadata: %s", metadata_csv)

    return result


def generate_synthetic(task: str, output_dir: str, num_samples: int) -> ValidationResult:
    """Generate synthetic data for testing. Uses the task's built-in generator."""
    logger.info("Generating %d synthetic samples for task: %s", num_samples, task)

    # Import the task's data module and call its generator
    task_modules = {
        "fraud": "tasks.fraud.data",
        "sepsis": "tasks.sepsis.data",
        "ecg": "tasks.ecg.data",
        "anomaly": "tasks.anomaly.data",
        "mortality": "tasks.mortality.data",
        "drug": "tasks.drug.data",
        "satellite": "tasks.satellite.data",
        "readmission": "tasks.readmission.data",
    }

    if task not in task_modules:
        return ValidationResult(False, [f"No synthetic generator for task: {task}"], [], {})

    import importlib
    mod = importlib.import_module(task_modules[task])

    # All task modules have a _generate_* function
    gen_fn = None
    for name in dir(mod):
        if name.startswith("_generate"):
            gen_fn = getattr(mod, name)
            break

    if gen_fn is None:
        return ValidationResult(False, [f"No generator function in {task_modules[task]}"], [], {})

    X, y = gen_fn(num_samples)
    config = DataConfig.for_task(task, output_dir, synthetic=True)
    result = validate_tabular(X, y, config)

    if result.is_valid:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "data.npz")
        np.savez_compressed(out_path, X=X, y=y)
        logger.info("  Saved: %s (%d samples)", out_path, len(X))

    return result


def show_manifest(path: str):
    """Display a manifest file."""
    manifest_path = os.path.join(path, "manifest.json") if os.path.isdir(path) else path
    if not os.path.exists(manifest_path):
        print(f"No manifest found at {manifest_path}")
        return

    manifest = DataManifest.load(manifest_path)
    print(f"Task:         {manifest.task}")
    print(f"Format:       {manifest.format}")
    print(f"Samples:      {manifest.num_samples}")
    print(f"Client:       {manifest.client_id or '(not set)'}")
    print(f"Version:      {manifest.version}")
    print(f"Created:      {manifest.created}")
    print(f"Checksum:     {manifest.checksum[:16]}...")
    print(f"Data path:    {manifest.data_path}")
    print(f"Schema:       {manifest.schema}")
    print(f"Label dist:   {manifest.label_distribution}")


def main():
    args = parse_args()

    # Show manifest mode
    if args.show_manifest:
        show_manifest(args.show_manifest)
        return

    if not args.task:
        print("Error: --task is required (unless using --show-manifest)")
        sys.exit(1)

    output_dir = args.output or os.path.join(DEFAULT_DATA_ROOT, args.task)
    config = DataConfig.for_task(args.task, output_dir)

    # Validate-only mode
    if args.validate_only:
        manifest_path = os.path.join(output_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            print(f"No manifest at {manifest_path}. Run ingestion first.")
            sys.exit(1)
        manifest = DataManifest.load(manifest_path)
        errors = manifest.validate_against(config)
        if errors:
            print(f"Validation FAILED: {errors}")
            sys.exit(1)
        print(f"Validation PASSED: {manifest.num_samples} samples, checksum {manifest.checksum[:16]}...")
        # Verify checksum
        data_path = os.path.join(output_dir, manifest.data_path)
        if os.path.exists(data_path):
            actual = compute_checksum(data_path)
            if actual != manifest.checksum:
                print(f"CHECKSUM MISMATCH: expected {manifest.checksum[:16]}, got {actual[:16]}")
                sys.exit(1)
            print("Checksum verified.")
        return

    # Synthetic mode
    if args.synthetic:
        result = generate_synthetic(args.task, output_dir, args.num_samples)
        if not result.is_valid:
            print(f"Failed: {result.errors}")
            sys.exit(1)
        generate_manifest(output_dir, args.task, args.client_id)
        print(f"Synthetic data generated: {output_dir}")
        return

    # Ingest mode
    if not args.input:
        print("Error: --input is required (or use --synthetic for test data)")
        sys.exit(1)

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: input not found: {input_path}")
        sys.exit(1)

    # Check for existing data
    if os.path.exists(os.path.join(output_dir, "manifest.json")) and not args.force:
        print(f"Data already ingested at {output_dir}. Use --force to overwrite.")
        sys.exit(1)

    # Determine format and ingest
    if os.path.isdir(input_path):
        if not args.metadata:
            print("Error: --metadata required for image directory input")
            sys.exit(1)
        result = ingest_images(input_path, args.metadata, output_dir, config)
    elif input_path.endswith(".csv"):
        result = ingest_csv(input_path, output_dir, config)
    elif input_path.endswith(".npz"):
        result = ingest_npz(input_path, output_dir, config)
    else:
        print(f"Error: unsupported format. Expected .csv, .npz, or directory. Got: {input_path}")
        sys.exit(1)

    if not result.is_valid:
        print(f"\nIngestion FAILED:")
        for e in result.errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    # Generate manifest
    manifest = generate_manifest(output_dir, args.task, args.client_id)
    print(f"\nIngestion complete:")
    print(f"  Task:     {args.task}")
    print(f"  Samples:  {manifest.num_samples}")
    print(f"  Output:   {output_dir}")
    print(f"  Manifest: {output_dir}/manifest.json")
    print(f"  Checksum: {manifest.checksum[:16]}...")


if __name__ == "__main__":
    main()
