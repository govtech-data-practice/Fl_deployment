# FL Reference Implementation

Federated Learning reference implementation with Privacy-Enhancing Technologies (PETs) for cross-silo deployment on AWS/GCC.

Key capabilities:

- **Horizontal FL** — same features, different samples across sites
- **Vertical FL** — different features, same entities (with Private Set Alignment (PSA))
- **Split Learning** — model partitioned across sites
- **Transfer Learning** — pretrained models fine-tuned across sites
- **Federated LoRA** — only adapter weights are federated (for LLMs)

## Quick Start

```bash
# Clone and install
git clone https://github.com/govtech-data-practice/Fl_deployment.git fl-reference
cd fl-reference
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run smoke test (synthetic data, ~60 seconds)
python runners/run_ec2.py fraud --synthetic

# Validate a data manifest
python tools/validate_manifest.py ~/fl-deploy/data/fraud/manifest.json

# Check DP privacy budget
python tools/dp_budget.py --all --rounds 100
```

See [Tutorial 1: Setup & First Run](tutorials/beginner/01-setup.ipynb) for the full step-by-step guide.

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required |
| [Flower](https://flower.ai/) | >= 1.30 | FL framework |
| [PyTorch](https://pytorch.org/) | >= 2.2 | Model training |
| Docker | 24+ | Tutorials 8-9, 12 only |
| Terraform | 1.5+ | Tutorial 9 only |
| GPU (CUDA 12.4+) | Optional | Tutorial 11 (LLM), beneficial for 8-9 |

Install optional PET libraries: `pip install -e ".[pets]"` (TenSEAL, anonlink, clkhash)

## Architecture

### Horizontal FL — same features, different samples

```
Site A              Site B              Site C
[patient records]   [patient records]   [patient records]
     |                   |                   |
     |  model updates    |  model updates    |  model updates
     +--------->---------+--------->---------+
                         |
                  FL Server (aggregates)
                  No raw data leaves any site
```

### Vertical FL — different features, same entities

```
Org A               Org B               Org C
[transactions]      [credit scores]     [demographics]
     |                   |                   |
     |  partial model    |  partial model    |  partial model
     +--------->---------+--------->---------+
                         |
                  FL Server (combines partial models)
                  Each org only sees its own feature columns
```

## Data

The repo includes real and synthetic datasets. See [data/README.md](data/README.md) for details.

| Dataset | Records | Source | Licence |
|---------|---------|--------|---------|
| **Credit Card Fraud 2023** | 25K sample (568K full) | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) | CC BY 4.0 |
| **METABRIC Breast Cancer** | 1,904 | [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric) | Open access |
| **Singapore PSA Records** | 1K + hard negatives | Synthetic (multi-ethnic names, HDB addresses) | — |
| **Sepsis, ECG, etc.** | 500 each | Synthetic generators | — |

## Models

| Model | Parameters | Use Case |
|-------|-----------|----------|
| MLP | 50K | Tabular (fraud, drug) |
| BiLSTM | 500K | Time-series (sepsis, ECG) |
| DenseNet-121 | 8M | Medical imaging (chest X-ray) |
| VFL MLP | 50K | Vertical FL (multi-party) |
| Split BiLSTM | 500K | Split learning |
| Mistral 7B QLoRA | 7B (160MB adapter) | Clinical NLP |

## CLI Tools

| Tool | Command | Purpose |
|------|---------|---------|
| **FL Server** | `python runners/run_ec2.py fraud --synthetic` | Run FL training (simulation or distributed) |
| **FL Client** | `python runners/run_client.py --server host:9092` | Connect to a distributed FL coordinator |
| **Data Ingest** | `python tools/ingest.py --task sepsis --input data.csv` | Ingest and validate participant data |
| **DP Budget** | `python tools/dp_budget.py --all --rounds 100` | Calculate privacy budget (epsilon) for all presets |
| **Manifest Validator** | `python tools/validate_manifest.py manifest.json` | Validate data manifest against task requirements |
| **Data Generator** | `python data/generators/generate_all.py --task fraud` | Generate synthetic sample data for any task |
| **SG Synthetic** | `from data.generators.sg_synthetic import generate_records` | Generate Singapore patient data for PSA testing |
| **Benchmark** | `python tests/benchmarks/run_benchmarks.py --tasks fraud` | Run centralised vs FL accuracy comparison |
| **Test Suite** | `python tests/run_tests.py fraud` | Run strategy validation tests |

## FL Strategies

FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi, SecAgg+, DP-Central, DP-Local, OneOwner

## PET Toolkit (`fl_pets/`)

Privacy-Enhancing Technologies organised by FL lifecycle stage:

| Stage | PET | Library | What it does |
|-------|-----|---------|-------------|
| **Pre-training** | [PSA](tutorials/pets/psa-entity-alignment.ipynb) | anonlink + clkhash (Data61) | Fuzzy entity alignment across parties without revealing raw data |
| **During training** | [DP](tutorials/pets/dp-gradient-privacy.ipynb) | Opacus (Meta) | Per-sample gradient clipping + noise (DP-SGD with RDP accounting) |
| **During training** | [SecAgg](tutorials/pets/secagg-update-masking.ipynb) | Flower SecAgg+ | Pairwise masking so server only sees aggregate updates |
| **Inference** | [HE vs MPC](tutorials/pets/secure-inference.ipynb) | TenSEAL + CrypTen | Encrypted inference comparison: polynomial approx vs secret sharing |
| **Post-training** | Privacy Attacks | Custom suite | MIA, gradient leakage (DLG), model inversion, canary insertion |

```
Pre-training       During training        Inference          Post-training
+---------+        +----+  +------+       +----+  +-----+    +---------+
|  PSA    |  --->  | DP |  |SecAgg| --->  | HE |  | MPC | -> | Privacy |
| (align) |        |(noise) (mask)|       |(enc)  (split)|   | attacks |
+---------+        +----+  +------+       +----+  +-----+    +---------+
```

## Tutorials

Hands-on tutorials organised by experience level. See [tutorials/README.md](tutorials/README.md) for the full index.

| Level | Tutorials | Format |
|-------|-----------|--------|
| **Beginner** | 1. Setup, 2. First Model, 3. Data Pipeline | Jupyter Notebooks |
| **Intermediate** | 4. DP, 5. SecAgg, 6. Strategies, 7. Privacy Attacks | Jupyter Notebooks |
| **Advanced** | 8. Distributed, 9. Terraform, 10. VFL & PSA, 11. LLM, 12. Operations | Markdown guides |
| **PET Tools** | PSA, DP, SecAgg, HE, MPC | Jupyter Notebooks |
| **Reference** | Configuration, PET Reference, Deployment Guide | Markdown |

## Results (Validated on EC2)

| Task | Model | Best Strategy | Metric |
|------|-------|--------------|--------|
| Chest X-ray (NIH 112K) | DenseNet-121 | FedAvg | AUC 0.819 |
| Sepsis (eICU 100K+) | BiLSTM | SCAFFOLD | Acc 0.809 |
| Fraud (50K) | MLP | FedAvg | Acc 0.98 |
| Clinical NLP | Mistral 7B QLoRA | FL LoRA + DP | MIA 1.0->0.83 |

## Deployment

```bash
# Microservices (Docker Compose)
cd deploy/microservices
docker compose up

# Multi-node (Terraform + Docker)
cd deploy/terraform
terraform init && terraform apply
```

See [Tutorial 8](tutorials/advanced/08-distributed-deployment.md) and [Tutorial 9](tutorials/advanced/09-terraform.md).

## Licence

Copyright 2026. This software includes code released under the [GovTech Public Sector Licence](LICENSE) by Government Technology Agency and other contributing Singapore public sector agencies.
