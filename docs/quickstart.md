# Quick Start Guide

Step-by-step guide to running your first FL simulation with synthetic data on a single machine.

## Prerequisites

- Python 3.10+
- pip
- Git
- 4 GB RAM minimum (8 GB recommended)
- GPU optional (CPU works for smoke tests)

## 1. Clone and Install

```bash
git clone <repository-url> fl-reference
cd fl-reference

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## 2. Verify Installation

```bash
# Check tooling
./scripts/preflight.sh --check tooling
```

Expected output:
```
  [PASS] Python 3.12.x
  [PASS] Flower 1.30.0
  [PASS] PyTorch 2.x.x (CPU or CUDA)
```

## 3. Run Smoke Test

The smoke test runs FedAvg with 2 simulated clients for 3 rounds on synthetic fraud data:

```bash
python runners/run_ec2.py fraud --synthetic
```

This takes approximately 30-60 seconds on CPU. You should see:
- Data generation for 2 clients
- 3 rounds of federated training
- Final accuracy metrics

## 4. Explore DP Privacy Budget

```bash
# Show privacy budget for all DP presets at 100 rounds
python tools/dp_budget.py --all --rounds 100
```

Output:
```
Preset          sigma      C    epsilon
------------------------------------------
DP_STRONG         1.5    1.0       ~4.xx
DP_MODERATE       0.8    1.0      ~10.xx
DP_RELAXED        0.5    1.0      ~25.xx
```

## 5. Try a Different Task

```bash
# ECG arrhythmia classification
python runners/run_ec2.py ecg --synthetic

# Sepsis early warning
python runners/run_ec2.py sepsis --synthetic

# Anomaly detection (autoencoder)
python runners/run_ec2.py anomaly --synthetic
```

## 6. Ingest Your Own Data

```bash
# Ingest a CSV file
python tools/ingest.py --task fraud --input /path/to/transactions.csv --client-id site_01

# Generate synthetic data for testing
python tools/ingest.py --task fraud --synthetic --num-samples 5000

# Validate the manifest
python tools/validate_manifest.py ~/fl-deploy/data/fraud/manifest.json --task fraud
```

## 7. Run with Multiple Strategies

Edit a scenario YAML or use the built-in configurations:

```bash
# Run all strategies on fraud data
python runners/run_ec2.py fraud --synthetic --strategies all
```

## Next Steps

- [Configuration Reference](configuration.md) — tune parameters
- [Deployment Guide](deployment.md) — deploy to cloud infrastructure
- [Distributed Deployment](Distributed_Deployment_Guide.md) — full multi-node setup
- [PET Reference](PET_Reference.md) — privacy-enhancing technology details
