# FL Platform Tutorials

Hands-on tutorials organised by experience level.

## Prerequisites

- **Python 3.10+** required for all tutorials
- Run `pip install -e ".[dev]"` before starting (see Tutorial 1)
- **GPU** is optional — only needed for Tutorial 11 (LLM fine-tuning), and beneficial for Tutorials 8–9
- **Docker** is required only for Tutorials 8–9 and 12
- **Terraform + AWS account** required only for Tutorial 9

## Beginner (Jupyter Notebooks)

Start here. No cloud infrastructure needed — everything runs on a single machine.
Open in Jupyter: `jupyter notebook tutorials/beginner/`

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [1. Setup & First Run](beginner/01-setup.ipynb) | 20 min | Install, verify, train a model, run inference |
| [2. Your First Model](beginner/02-first-model.ipynb) | 25 min | Centralised baseline, FL comparison, VFL example |
| [3. Data Pipeline](beginner/03-data-pipeline.ipynb) | 15 min | Ingest data, validate manifests, understand the data flow |

## Intermediate (Jupyter Notebooks)

Add privacy controls and experiment with FL strategies. Still single-machine.
Open in Jupyter: `jupyter notebook tutorials/intermediate/`

Tutorials 4–6 each require Tutorial 3 but are independent of each other (can be done in any order).
Tutorial 7 requires both Tutorial 4 and Tutorial 5.

| Tutorial | Time | Prerequisites | What You'll Learn |
|----------|------|---------------|-------------------|
| [4. Differential Privacy](intermediate/04-differential-privacy.ipynb) | 25 min | Tutorial 3 | DP presets, privacy budget, accuracy/privacy trade-off |
| [5. Secure Aggregation](intermediate/05-secure-aggregation.ipynb) | 15 min | Tutorial 3 | SecAgg pairwise masking, quorum, combining with DP |
| [6. FL Strategies Deep Dive](intermediate/06-strategies.ipynb) | 30 min | Tutorial 3 | FedProx, SCAFFOLD, non-IID data, choosing the right strategy |
| [7. Privacy Attack Testing](intermediate/07-privacy-attacks.ipynb) | 25 min | Tutorials 4 & 5 | MIA, gradient leakage, model inversion, canary insertion |

## Advanced (Deployment Guides)

Multi-node deployment, infrastructure, and specialised FL paradigms.

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [8. Distributed Deployment](advanced/08-distributed-deployment.md) | 45 min | Docker Compose microservices, multi-node training |
| [9. Infrastructure with Terraform](advanced/09-terraform.md) | 30 min | Provision AWS infra, VPC, security groups |
| [10. Vertical FL & PSA](advanced/10-vertical-fl.md) | 25 min | VFL with split features, PSA entity alignment, split learning |
| [11. LLM Federated Fine-tuning](advanced/11-llm-finetuning.md) | 30 min | Federated LoRA/QLoRA, Mistral, OLMo, adapter aggregation |
| [12. Operations & Production](advanced/12-operations.md) | 30 min | Monitoring, cert rotation, governance, cost |

## All-Paradigms Demo

- [demo_all_paradigms.ipynb](demo_all_paradigms.ipynb) — HFL + VFL + Federated Transfer Learning (FTL) + full PET toolkit in one notebook

## What to Do Next

After completing the tutorials:

- **Production deployments** — see the [reference docs](reference/) for configuration, PET details, and production operations
- **Pre-built experiments** — explore the `scenarios/` directory for ready-to-run experiment configurations
- **Contributing** — open an issue or pull request on GitHub

## Reference

- [Configuration Reference](reference/configuration.md)
- [PET Reference](reference/PET_Reference.md)
- [Distributed Deployment Guide](reference/Distributed_Deployment_Guide.md)
- [Production Technical Reference](reference/FL_Production_Technical_Reference.md)
