# FL Reference Implementation

Federated Learning reference implementation with Privacy-Enhancing Technologies (PETs) for cross-silo deployment on AWS/GCC.

## Overview

This repository provides a working implementation of cross-silo federated learning, designed as a companion to the [FL Deployment Guide](docs/deployment.md). It demonstrates how organisations can train shared models across institutional boundaries **without moving raw data**.

Key capabilities:

- **Horizontal FL** — same features, different samples across sites
- **Vertical FL** — different features, same entities (with PSI alignment)
- **Split Learning** — model partitioned across sites
- **Transfer Learning** — pretrained models fine-tuned across sites
- **Federated LoRA** — only adapter weights are federated (for LLMs)

## Prerequisites

- Python 3.10+
- Docker (for containerised deployment)
- NVIDIA GPU with CUDA 12.4+ (optional, for GPU training)

## Quick Start

```bash
# Clone and install
git clone <repository-url> fl-reference
cd fl-reference
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run smoke test (synthetic data, ~60 seconds)
python run_ec2.py fraud --synthetic

# Validate a data manifest
python validate_manifest.py ~/fl-deploy/data/fraud/manifest.json

# Check DP privacy budget
python dp_budget.py --all --rounds 100
```

See [docs/quickstart.md](docs/quickstart.md) for the full step-by-step guide.

## Architecture

```
                     Coordinator (ServerApp)
                            |
              +-------------+-------------+
              |             |             |
         Client A      Client B      Client C
         [local data]  [local data]  [local data]

  - Clients train locally, send only encrypted model updates
  - Coordinator aggregates updates (FedAvg, SCAFFOLD, etc.)
  - mTLS secures all communication
  - SecAgg masks individual updates
  - Differential Privacy bounds information leakage
```

## Repository Structure

```
fl-reference/
  serverapp/          Coordinator (ServerApp) facade
  clientapp/          Client (ClientApp) facade
  fl_common/          Shared FL library (strategies, DP, SecAgg, data pipeline)
  models/             Model implementations (13 architectures)
  tasks/              Task/data definitions (12 tasks)
  privacy/            Privacy attack suite and DP testing
  secagg/             Secure aggregation configuration
  psi/                Private Set Intersection (entity alignment for VFL)
  secure_inference/   Secure inference demos (HE, MPC, TEE)
  scripts/            Operational scripts (preflight, diagnostics)
  deploy/             Deployment tooling (Docker, Terraform, mTLS)
  infra/              Infrastructure-as-code (Terraform, CDK)
  configs/            Environment-specific configuration (dev, staging, prod)
  templates/          Manifest and agreement templates
  runbooks/           Operational runbooks
  scenarios/          Experiment definitions (YAML)
  docs/               Documentation
  runs/               Run records output
```

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| BiLSTM | 500K | Time-series (sepsis, ECG) |
| MLP | 50K | Tabular (fraud, drug) |
| DenseNet-121 | 8M | Medical imaging (chest X-ray) |
| ResNet-small | 5M | Satellite imagery |
| Autoencoder | 500K | Anomaly detection |
| LogReg | 10K | Risk scoring (readmission) |
| 1D CNN | 200K | Signal classification (ECG) |
| TabNet | 1M | Structured data (mortality) |
| VFL MLP | 50K | Vertical FL (multi-party) |
| Split BiLSTM | 500K | Split learning |
| Mistral 7B QLoRA | 7B (160MB adapter) | Clinical NLP |
| Generic MLP | configurable | Custom datasets |

## FL Strategies

FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi, SecAgg+, DP-Central, DP-Local, DP-Local (low epsilon), OneOwner

## Privacy Controls

- **Differential Privacy**: DP-SGD with RDP accounting. Three presets: `DP_STRONG` (default), `DP_MODERATE`, `DP_RELAXED`
- **Secure Aggregation**: Pairwise masking (SecAgg+)
- **Privacy Attack Testing**: Membership inference, attribute inference, model inversion, gradient leakage, canary insertion

## Configuration

- `env.example.yaml` — YAML configuration template
- `deploy/cluster.env.template` — Shell-format configuration
- `configs/` — Environment-specific configs (dev, staging, production)
- `scenarios/` — Experiment definitions

See [docs/configuration.md](docs/configuration.md) for all parameters.

## Documentation

- [Quick Start](docs/quickstart.md) — Local simulation setup
- [Configuration Reference](docs/configuration.md) — All configurable parameters
- [Deployment Guide](docs/deployment.md) — Per-environment deployment
- [Troubleshooting](docs/troubleshooting.md) — Common issues and diagnostics
- [Cost Reporting](docs/cost-reporting.md) — Cost tracking methodology
- [Distributed Deployment](docs/Distributed_Deployment_Guide.md) — Full multi-node setup
- [PET Reference](docs/PET_Reference.md) — DP, SecAgg, HE, MPC, TEE details

## Operational Scripts

```bash
# Pre-flight validation
./scripts/preflight.sh --check tooling --check endpoints

# Diagnostic bundle
./scripts/diagnose.sh --run-id <id> --env production --since 2h

# Cluster health check
./deploy/health_check.sh

# Certificate rotation
./deploy/rotate_certs.sh

# Deploy to cluster
./deploy/distributed/deploy.sh up
```
