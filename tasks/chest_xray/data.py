"""
Chest X-ray Data Pipeline
==========================
Pipeline: Load CSV → Parse labels → Validate → Filter → Partition → Patient-level split

Real data: NIH Chest X-ray (112K images, 14 pathology labels)
Image transforms handled in models/densenet/client_app.py (Resize, Crop, Normalize)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict

logger = logging.getLogger("pipeline.chest_xray")

CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


def partition_data_dynamic(
    csv_path: str,
    client_id: int,
    num_clients: int,
    method: str = "iid",
    alpha: float = 0.5,
    classes_per_client: int = 3,
    min_samples: int = 100,
    val_split: float = 0.2,
    seed: int = 42,
) -> Dict:
    combined_seed = int(seed + alpha * 1000) % (2**31)

    # ── Load ──
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"Invalid client_id {client_id}")

    df = pd.read_csv(csv_path)
    logger.info(f"[Load] {len(df)} rows from {csv_path}")

    # ── Parse labels ──
    df['Finding Labels List'] = df['Finding Labels'].apply(
        lambda x: x.split('|') if isinstance(x, str) else []
    )
    for cls in CLASSES:
        df[cls] = df['Finding Labels List'].apply(lambda x: 1 if cls in x else 0)
    df = df.drop('Finding Labels List', axis=1)

    # ── Validate ──
    n_missing = df['Image Index'].isna().sum()
    n_no_label = (df['Finding Labels'].isna() | (df['Finding Labels'] == '')).sum()
    if n_missing > 0:
        logger.warning(f"[Validate] {n_missing} rows missing Image Index")
    if n_no_label > 0:
        logger.warning(f"[Validate] {n_no_label} rows missing Finding Labels")

    # ── Filter to samples with findings ──
    has_finding = df[CLASSES].sum(axis=1) > 0
    df_with_findings = df[has_finding].copy()

    # ── Log label distribution ──
    label_counts = df_with_findings[CLASSES].sum()
    logger.info(f"[Validate] {len(df_with_findings)} samples with findings, "
                f"{len(df) - len(df_with_findings)} 'No Finding' excluded")
    logger.info(f"[Validate] Top labels: "
                f"{', '.join(f'{c}={int(v)}' for c, v in label_counts.nlargest(5).items())}")

    if method == "label_skew":
        client_df = _partition_label_skew(df_with_findings, client_id, num_clients, classes_per_client, combined_seed)
    elif method == "iid":
        client_df = _partition_iid(df_with_findings, client_id, num_clients, combined_seed)
    elif method == "patient":
        client_df = _partition_patient(df_with_findings, client_id, num_clients, alpha, combined_seed)
    elif method == "quantity_skew":
        client_df = _partition_quantity(df_with_findings, client_id, num_clients, alpha, combined_seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ── Enforce minimum samples ──
    if len(client_df) < min_samples:
        np.random.seed(combined_seed + client_id)
        additional = df_with_findings.sample(
            n=min_samples - len(client_df), random_state=combined_seed + client_id
        )
        client_df = pd.concat([client_df, additional]).drop_duplicates(subset=['Image Index'])
        logger.info(f"[Partition] Client {client_id}: padded to {len(client_df)} (min={min_samples})")

    logger.info(f"[Partition] Client {client_id}: {len(client_df)} samples via {method}")

    # ── Patient-level train/val split ──
    return _split_train_val(client_df, val_split, combined_seed)


def _partition_label_skew(df, client_id, num_clients, classes_per_client, seed):
    np.random.seed(seed)
    start_idx = (client_id * classes_per_client) % len(CLASSES)
    assigned = [CLASSES[(start_idx + i) % len(CLASSES)] for i in range(classes_per_client)]
    mask = df[assigned].sum(axis=1) > 0
    client_df = df[mask].copy()
    target_size = len(df) // num_clients
    if len(client_df) > target_size * 1.5:
        client_df = client_df.sample(n=int(target_size * 1.2), random_state=seed + client_id)
    return client_df


def _partition_iid(df, client_id, num_clients, seed):
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    chunk = len(df) // num_clients
    start = client_id * chunk
    end = len(df) if client_id == num_clients - 1 else start + chunk
    return df_shuffled.iloc[start:end].copy()


def _partition_patient(df, client_id, num_clients, alpha, seed):
    np.random.seed(seed)
    patient_ids = df['Patient ID'].unique()
    np.random.shuffle(patient_ids)
    proportions = np.random.dirichlet([alpha] * num_clients)
    split_points = np.cumsum([int(len(patient_ids) * p) for p in proportions[:-1]])
    patients_per_client = np.split(patient_ids, split_points)
    return df[df['Patient ID'].isin(patients_per_client[client_id])].copy()


def _partition_quantity(df, client_id, num_clients, alpha, seed):
    np.random.seed(seed)
    proportions = np.random.dirichlet([alpha] * num_clients)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    cumsum = np.cumsum(proportions)
    start = int(cumsum[client_id - 1] * len(df)) if client_id > 0 else 0
    end = int(cumsum[client_id] * len(df))
    return df_shuffled.iloc[start:end].copy()


def _split_train_val(df, val_split=0.2, seed=42):
    """Patient-level split to avoid data leakage."""
    np.random.seed(seed)
    df = df.copy()
    df['Patient ID'] = df['Image Index'].apply(lambda x: x.split('_')[0])

    patients = df['Patient ID'].unique()
    np.random.shuffle(patients)
    val_count = int(len(patients) * val_split)
    val_patients = set(patients[:val_count])

    val_df = df[df['Patient ID'].isin(val_patients)]
    train_df = df[~df['Patient ID'].isin(val_patients)]

    def to_dict(d):
        return {
            'images': d['Image Index'].values,
            'labels': d[CLASSES].values.astype(np.float32),
        }

    logger.info(f"[Split] train={len(train_df)} ({len(patients)-val_count} patients), "
                f"val={len(val_df)} ({val_count} patients)")

    return {'train': to_dict(train_df), 'val': to_dict(val_df)}
