# Data Directory

Synthetic data generation, sample datasets, and audit trail for FL experiments.

## Structure

```
data/
  generators/          Synthetic data generators (one per task)
    generate_all.py      Generate samples for all tasks
  samples/             Pre-generated sample datasets (ready to use)
    fraud/               500 synthetic transactions
    sepsis/              500 synthetic vitals (48-step time series)
    ecg/                 500 synthetic 12-lead ECG signals
    anomaly/             500 synthetic embeddings
    mortality/           500 synthetic ICU records
    drug/                500 synthetic molecular fingerprints
    readmission/         500 synthetic discharge records
    satellite/           500 synthetic multispectral patches
  audit/               Generation audit trail (JSONL)
    generation_log.jsonl   Timestamped record of every generation run
```

## Quick Start

```bash
# Generate all sample datasets (writes to data/samples/)
python data/generators/generate_all.py

# Generate a specific task with custom size
python data/generators/generate_all.py --task fraud --num-samples 1000

# View audit trail
cat data/audit/generation_log.jsonl

# Use samples in training
python runners/run_ec2.py fraud --data-dir data/samples/fraud
```

## Audit Trail

Every generation run is logged to `data/audit/generation_log.jsonl` with:
- Timestamp (ISO 8601)
- Task name
- Number of samples generated
- Output path
- SHA-256 checksum of generated data
- Generator version
- Random seed used

This provides full provenance for all synthetic data used in experiments.
