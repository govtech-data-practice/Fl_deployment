# Tutorial 3: Data Pipeline

**Time:** 15 minutes | **Level:** Beginner | **Prerequisites:** [Tutorial 2](02-first-model.md)

## What You'll Learn

- Ingest data using `tools/ingest.py`
- Understand data manifests and validation
- Use the generic pipeline for custom datasets

## Step 1: Generate Synthetic Data

Each participant ingests their own data locally. The server never sees raw data.

```bash
# Generate synthetic fraud data for two "banks"
python tools/ingest.py --task fraud --synthetic --num-samples 5000 --client-id bank_01
python tools/ingest.py --task fraud --synthetic --num-samples 3000 --client-id bank_02
```

This creates:
- `~/fl-deploy/data/fraud/data.npz` — features and labels
- `~/fl-deploy/data/fraud/manifest.json` — metadata about the dataset

## Step 2: Inspect the Manifest

```bash
python tools/ingest.py --show-manifest ~/fl-deploy/data/fraud
```

A manifest describes the dataset without revealing any raw data:

```json
{
  "task": "fraud",
  "format": "npz",
  "num_samples": 5000,
  "schema": {
    "input_dim": 30,
    "num_classes": 2,
    "task_type": "binary"
  },
  "label_distribution": {"0": 0.85, "1": 0.15},
  "checksum": "sha256:abc123...",
  "client_id": "bank_01",
  "version": "1.0"
}
```

The coordinator collects manifests (not data) to verify compatibility before training.

## Step 3: Validate a Manifest

```bash
python tools/validate_manifest.py ~/fl-deploy/data/fraud/manifest.json --task fraud
```

**Expected output:**
```
Manifest:  ~/fl-deploy/data/fraud/manifest.json
Task:      fraud
Client:    bank_01
Samples:   5000
Format:    npz
Checksum:  a1b2c3d4e5f6...

PASSED — manifest is valid.
```

Validation checks:
- Feature dimensions match the task definition
- Minimum sample count (>= 10)
- Label distribution is not degenerate
- Data file checksum matches

## Step 4: Ingest a CSV File

You can ingest any tabular CSV. The pipeline auto-detects the schema:

```bash
# Ingest a CSV — last column (or column named 'label') becomes the target
python tools/ingest.py --task generic --input /path/to/your_data.csv --client-id my_site

# Validate
python tools/ingest.py --task generic --validate-only
```

## Step 5: Understand the Data Flow

```
                    Participant's Site
                    ==================
  Raw Data (CSV/NPZ)
       |
  [tools/ingest.py] ──> data.npz + manifest.json
       |
  [tools/validate_manifest.py] ──> PASS/FAIL
       |
  [client_app.py] ──> loads data.npz, trains locally
       |
  Model updates ──────────> Coordinator (never raw data)
```

Key principles:
- **Data stays local** — `tools/ingest.py` runs on the participant's machine
- **Only metadata is shared** — manifests describe shape, not content
- **Validation gates training** — errors in validation block training from starting
- **Checksums ensure integrity** — SHA-256 hash verifies data hasn't changed

## Step 6: Data Manifest Template

For production, use the manifest template:

```bash
cat templates/data_manifest.yaml
```

This template is submitted by each participant before a training run, documenting their dataset version, validation status, and approval.

## What You Learned

- `tools/ingest.py` converts raw data into the standardised format
- Manifests describe datasets without exposing raw data
- `tools/validate_manifest.py` ensures data compatibility before training
- The generic pipeline supports arbitrary tabular CSV data

## Next Steps

You've completed the beginner tutorials. Move on to:

- [Tutorial 4: Differential Privacy](../intermediate/04-differential-privacy.md) — add privacy guarantees
