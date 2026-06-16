# FL Platform Tutorials

Hands-on tutorials organised by experience level.

## Beginner

Start here. No cloud infrastructure needed — everything runs on a single machine.

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [1. Setup & First Run](beginner/01-setup.md) | 15 min | Install, verify, run your first FL experiment |
| [2. Your First Model](beginner/02-first-model.md) | 20 min | Understand FL training, try different tasks and strategies |
| [3. Data Pipeline](beginner/03-data-pipeline.md) | 15 min | Ingest data, validate manifests, understand the data flow |

## Intermediate

Add privacy controls and experiment with FL strategies. Still single-machine.

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [4. Differential Privacy](intermediate/04-differential-privacy.md) | 25 min | DP presets, privacy budget, measuring the accuracy/privacy trade-off |
| [5. Secure Aggregation](intermediate/05-secure-aggregation.md) | 15 min | Enable SecAgg, understand pairwise masking, test quorum behaviour |
| [6. FL Strategies Deep Dive](intermediate/06-strategies.md) | 30 min | FedProx, SCAFFOLD, non-IID data, choosing the right strategy |
| [7. Privacy Attack Testing](intermediate/07-privacy-attacks.md) | 25 min | Run MIA, gradient leakage, canary insertion, interpret results |

## Advanced

Multi-node deployment, infrastructure, and specialised FL paradigms.

| Tutorial | Time | What You'll Learn |
|----------|------|-------------------|
| [8. Distributed Deployment](advanced/08-distributed-deployment.md) | 45 min | Deploy to EC2, mTLS, Docker, multi-node training |
| [9. Infrastructure with Terraform](advanced/09-terraform.md) | 30 min | Provision AWS infra, VPC, security groups, automated setup |
| [10. Vertical FL & PSI](advanced/10-vertical-fl.md) | 25 min | VFL with split features, PSI entity alignment, split learning |
| [11. LLM Federated Fine-tuning](advanced/11-llm-finetuning.md) | 30 min | Federated LoRA/QLoRA, Mistral, OLMo, adapter aggregation |
| [12. Operations & Production](advanced/12-operations.md) | 30 min | Monitoring, troubleshooting, cert rotation, cost, governance |

## Reference

Detailed parameter tables and supplementary material:

- [Configuration Reference](../configuration.md)
- [PET Reference](../PET_Reference.md)
- [Distributed Deployment Guide](../Distributed_Deployment_Guide.md)
- [Production Technical Reference](../FL_Production_Technical_Reference.md)
- [Cost Reporting](../cost-reporting.md)
