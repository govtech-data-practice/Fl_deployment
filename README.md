# FL Reference Implementation

Federated Learning reference implementation with Privacy-Enhancing Technologies (PETs) for cross-silo deployment on AWS/GCC.

## Overview

This repository provides a working implementation of cross-silo federated learning, designed as a companion to the [FL Deployment Guide](tutorials/advanced/08-distributed-deployment.md). It demonstrates how organisations can train shared models across institutional boundaries **without moving raw data**.

Key capabilities:

- **Horizontal FL** — same features, different samples across sites
- **Vertical FL** — different features, same entities (with PSI alignment)
- **Split Learning** — model partitioned across sites
- **Transfer Learning** — pretrained models fine-tuned across sites
- **Federated LoRA** — only adapter weights are federated (for LLMs)

## Prerequisites

### System

- Python 3.10+
- Docker 24+ (for containerised deployment)
- NVIDIA GPU with CUDA 12.4+ driver (optional, for GPU training)
- OpenSSL (for certificate operations)
- Terraform 1.5+ (optional, for infrastructure provisioning)

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| [Flower](https://flower.ai/) | >= 1.30 | FL framework (simulation + distributed) |
| [PyTorch](https://pytorch.org/) | >= 2.2 | Model training |
| [torchvision](https://pytorch.org/) | >= 0.17 | Image model support (DenseNet, ResNet) |
| [NumPy](https://numpy.org/) | >= 1.26 | Numerical computing |
| [scikit-learn](https://scikit-learn.org/) | >= 1.5 | Metrics, preprocessing |
| [pandas](https://pandas.pydata.org/) | >= 2.0 | Data loading and manipulation |
| [Pillow](https://pillow.readthedocs.io/) | >= 10.0 | Image processing |
| [PyYAML](https://pyyaml.org/) | >= 6.0 | Configuration parsing |

### Optional Libraries (PETs & LLM)

Install with `pip install -e ".[pets]"` or `pip install -e ".[all]"`:

| Library | Version | Purpose |
|---------|---------|---------|
| [TenSEAL](https://github.com/OpenMined/TenSEAL) | >= 0.3 | Homomorphic encryption (CKKS/BFV) |
| [Transformers](https://huggingface.co/transformers/) | >= 4.40 | LLM model loading (Mistral, OLMo) |
| [PEFT](https://github.com/huggingface/peft) | >= 0.10 | LoRA/QLoRA adapter fine-tuning |
| [Accelerate](https://github.com/huggingface/accelerate) | >= 0.30 | Distributed training utilities |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | >= 0.43 | 4-bit quantisation (QLoRA) |

### Dev Tools

Install with `pip install -e ".[dev]"`:

| Library | Version | Purpose |
|---------|---------|---------|
| [pytest](https://pytest.org/) | >= 8.0 | Testing |
| [Ruff](https://docs.astral.sh/ruff/) | >= 0.4 | Linting and formatting |

## Quick Start

```bash
# Clone and install
git clone <repository-url> fl-reference
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

See [Tutorial 1: Setup & First Run](tutorials/beginner/01-setup.md) for the full step-by-step guide.

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

### Split Learning — model split across sites

```
Site (bottom)              Cloud (top)
[raw data → LSTM]  ──────> [classifier]
     |                          |
   embeddings             predictions
   (no raw data)          (no raw data)
```

### Federated Transfer Learning — pretrained model fine-tuned across sites

```
                  Pretrained Model (e.g. ImageNet DenseNet-121)
                         |
         +───────────────+───────────────+
         |               |               |
    Site A           Site B           Site C
    [fine-tune on    [fine-tune on    [fine-tune on
     local images]    local images]    local images]
         |               |               |
         +──── FL Server aggregates fine-tuned weights ────+
```

### Federated LoRA — large model, only adapters are federated

```
                  Frozen LLM (e.g. Mistral 7B)
                         |
         +───────────────+───────────────+
         |               |               |
    Site A           Site B           Site C
    [train LoRA      [train LoRA      [train LoRA
     adapter on       adapter on       adapter on
     local docs]      local docs]      local docs]
         |               |               |
         +──── FL Server aggregates LoRA adapters (160MB) ────+
                  (not the full 7B model)
```

Each participant keeps data on-premise. Only encrypted model updates (or adapters) leave each site. The server aggregates without ever seeing raw data.

## Repository Structure

```
fl-reference/
  models/
    hfl/              Horizontal FL (MLP, BiLSTM, DenseNet, CNN1D, ...)
    vfl/              Vertical FL & Split Learning (VFL MLP, Split BiLSTM)
    ftl/              Federated Transfer Learning (DenseNet + ImageNet)
    llm/              LLM fine-tuning (Mistral QLoRA, OLMo LoRA)
  tasks/
    hfl/              HFL data tasks (fraud, sepsis, ECG, chest X-ray, ...)
    llm/              LLM data tasks (gov_doc, gov_llm)
  fl_pets/            PET toolkit (by lifecycle stage)
    dp.py               During training — Opacus DP-SGD + RDP accounting
    secagg.py            During training — Flower SecAgg+ pairwise masking
    psi.py               Pre-training — ECDH-PSI entity alignment
    he.py                Inference — TenSEAL CKKS homomorphic encryption
    mpc.py               Inference — CrypTen multi-party computation
  fl_common/          Core FL library (strategies, data pipeline)
  privacy/            Privacy attack suite (MIA, DLG, model inversion, canary)
  secure_inference/   Secure inference implementations (Paillier, MPC, TEE)
  tools/              CLI tools
    tools/ingest.py            Data ingestion pipeline
    tools/dp_budget.py         DP privacy budget calculator
    tools/validate_manifest.py Data manifest validator
  runners/            FL execution entry points
    runners/run_ec2.py           Server runner (simulation + distributed)
    runners/run_client.py        Client runner (distributed mode)
  tests/              Test and benchmark scripts
    tests/run_tests.py         Strategy test runner
    tests/run_all.py           Full model x task benchmark
  deploy/
    microservices/    Docker Compose (coordinator + clients)
    distributed/      Multi-node Docker Compose (SuperLink + SuperNodes)
    terraform/        AWS infrastructure (VPC, EC2, S3)
  configs/            Environment configs (dev, staging, production)
  scenarios/          Experiment definitions (YAML)
  templates/          Manifest and agreement templates
  runbooks/           Operational runbooks
  tutorials/
    beginner/       Jupyter notebooks (01-setup, 02-first-model, 03-data-pipeline)
    intermediate/   Jupyter notebooks (04-DP, 05-SecAgg, 06-strategies, 07-attacks)
    advanced/       Deployment guides (08-distributed, 09-terraform, 10-VFL, 11-LLM, 12-ops)
    reference/      Configuration, PET reference, production technical reference
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

See [tutorials/reference/configuration.md](tutorials/reference/configuration.md) for all parameters.

## Tutorials

Hands-on tutorials organised by experience level. See [tutorials/](tutorials/README.md) for the full index.

### Beginner — Jupyter Notebooks

| # | Tutorial | Time | Topic |
|---|----------|------|-------|
| 1 | [Setup & First Run](tutorials/beginner/01-setup.ipynb) | 20 min | Install, verify, train, inference |
| 2 | [Your First Model](tutorials/beginner/02-first-model.ipynb) | 25 min | Centralised baseline, FL comparison, VFL |
| 3 | [Data Pipeline](tutorials/beginner/03-data-pipeline.ipynb) | 15 min | Ingest, validate, manifests |

### Intermediate — Jupyter Notebooks

| # | Tutorial | Time | Topic |
|---|----------|------|-------|
| 4 | [Differential Privacy](tutorials/intermediate/04-differential-privacy.ipynb) | 25 min | DP presets, budget, trade-offs |
| 5 | [Secure Aggregation](tutorials/intermediate/05-secure-aggregation.ipynb) | 15 min | SecAgg, pairwise masking |
| 6 | [FL Strategies](tutorials/intermediate/06-strategies.ipynb) | 30 min | FedProx, SCAFFOLD, non-IID |
| 7 | [Privacy Attacks](tutorials/intermediate/07-privacy-attacks.ipynb) | 25 min | MIA, gradient leakage, canary |

### Advanced (multi-node, cloud)

| # | Tutorial | Time | Topic |
|---|----------|------|-------|
| 8 | [Distributed Deployment](tutorials/advanced/08-distributed-deployment.md) | 45 min | EC2, mTLS, Docker |
| 9 | [Terraform](tutorials/advanced/09-terraform.md) | 30 min | AWS provisioning |
| 10 | [Vertical FL & PSI](tutorials/advanced/10-vertical-fl.md) | 25 min | VFL, entity alignment, split learning |
| 11 | [LLM Fine-tuning](tutorials/advanced/11-llm-finetuning.md) | 30 min | Federated LoRA/QLoRA |
| 12 | [Operations](tutorials/advanced/12-operations.md) | 30 min | Monitoring, certs, governance, cost |

### Reference

- [Configuration Reference](tutorials/reference/configuration.md) — All configurable parameters
- [PET Reference](tutorials/reference/PET_Reference.md) — DP, SecAgg, HE, MPC, TEE details
- [Distributed Deployment Guide](tutorials/reference/Distributed_Deployment_Guide.md) — Detailed multi-node setup
- [Operations & Cost](tutorials/advanced/12-operations.md) — Monitoring, governance, cost tracking

## Infrastructure (Terraform)

Provision the full FL cluster on AWS with Terraform:

```bash
cd deploy/terraform/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform init
terraform plan
terraform apply
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `region` | `ap-southeast-1` | AWS region |
| `num_supernodes` | `2` | Number of FL client instances |
| `superlink_instance_type` | `t3.large` | Coordinator instance type |
| `supernode_instance_type` | `g4dn.xlarge` | Client instance type (GPU) |
| `use_gpu` | `false` | Use GPU instances for clients |
| `key_name` | (required) | SSH key pair name |
| `data_s3_bucket` | (required) | S3 bucket with training data |
| `enable_tls` | `true` | Enable mTLS between nodes |

Terraform provisions: VPC, subnets, security groups, EC2 instances (1 coordinator + N clients), S3 access, and TLS certificates. See `deploy/terraform/` for full configuration.

## Results (Validated on EC2)

| Task | Model | Best Strategy | Metric | Notes |
|------|-------|--------------|--------|-------|
| Chest X-ray (NIH 112K) | DenseNet-121 | FedAvg | AUC 0.819 | 14-label multilabel |
| Sepsis (eICU 100K+) | BiLSTM | SCAFFOLD | Acc 0.809 | 5 clients distributed |
| Fraud (50K) | MLP | FedAvg | Acc 0.98 | 11/11 strategies pass |
| Mortality (eICU) | TabNet | FedAvg | Acc 0.876 | 11/11 strategies pass |
| Readmission | LogReg | FedAvg | - | 11/11 strategies pass |
| Anomaly detection | Autoencoder | FedAvg | AUC 0.93 | Unsupervised |
| Clinical NLP | Mistral 7B QLoRA | FL LoRA + DP | MIA 1.0->0.83 | Canary leakage reduced |

### Key Findings

- FedAvg AUC 0.819 on chest X-ray (14-label, NIH CXR-14 dataset)
- DP destroys large models (DenseNet AUC->0.50) but works on small models (BiLSTM 0.65)
- LLM canary leakage: 41.7% without DP -> 16.7% with DP
- FedAdam/FedYogi diverge on pretrained models — FedAvg+SCAFFOLD preferred

## Roadmap

### v0.1 — Foundation (Done)

12 models, 10 tasks, 11 FL strategies, 4 secure inference demos, distributed across 6 EC2 GPU nodes. Production data pipeline with `tools/ingest.py`.

### v0.2 — Synthetic Data Engine & Integrity

- Synthetic data engine: generators for each demo domain (EHR time series, medical images, transactions, ECG waveforms, satellite patches, molecular fingerprints) — for testing data format compatibility, pipeline integration, partitioning strategies, and non-IID distribution scenarios
- Robust aggregation: Krum, Multi-Krum, Trimmed Mean, Bulyan, FLTrust
- Poisoning attack suite: label flipping, gradient scaling, backdoor
- Audit trail: round-level JSONL log, HMAC signing, model provenance, privacy budget ledger

### v0.3 — TEE

- FL server inside AWS Nitro Enclave
- Client-side attestation verification (PCR validation)
- Encrypted client updates via enclave public key

### v0.4 — Advanced PETs

- SecAgg+ (dropout-tolerant), Paillier/CKKS HE aggregation
- DP-FTRL, adaptive clipping, user-level DP
- Federated analytics (aggregate stats without ML)
- Federated GAN: DP-CTGAN for synthetic tabular data, federated DCGAN/StyleGAN for synthetic image generation

### v0.5 — Deployment Hardening

- Air-gapped deployment (offline Docker + pip bundles)
- RBAC (Admin, Operator, Data Owner, Auditor, Model Consumer)
- Data sovereignty policy engine (geo-tagging, region restrictions)
- Cost calculator: estimate compute costs by task, model size, number of clients, rounds, and cloud provider

### v1.0 — Extended Models

- New data types: U-Net (segmentation), YOLO (detection), GNN (graphs), BERT (NER), Whisper (audio), 3D CNN (volumetric)
- New FL paradigms: federated XGBoost, multi-modal (CLIP), federated RL, federated GAN
- Production secure inference:
  - **CrypTen** (Meta) — MPC-based, PyTorch native. Runs DenseNet-121, ResNet-50, BERT, speech models
  - **CrypTFlow2/EzPC** (Microsoft Research) — 2PC with OT. Validated on DenseNet-121 chest X-ray across 7 multi-institution sites
  - **SecretFlow/SPU** (Ant Group) — MPC hybrid (ABY3, Semi2k, Cheetah). Runs LLaMA-7B inference
  - **Concrete ML** (Zama) — FHE-based, single-server. Best for small-medium models with quantization

## Deployment

```bash
# Microservices (Docker Compose)
cd deploy/microservices
docker compose up                          # coordinator + 2 clients
FL_TASK=sepsis docker compose up           # different task

# Multi-node (Terraform + Docker)
cd deploy/terraform
terraform init && terraform apply          # provision AWS infra
```

See [Tutorial 8: Distributed Deployment](tutorials/advanced/08-distributed-deployment.md) and [Tutorial 9: Terraform](tutorials/advanced/09-terraform.md).
