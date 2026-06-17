# Deployment Guide

Per-environment deployment instructions for the FL platform.

## Environments

| Environment | Config | Purpose |
|-------------|--------|---------|
| Development | `configs/dev.yaml` | Local simulation, single machine |
| Staging | `configs/staging.yaml` | Multi-node with synthetic data |
| Production | `configs/production.yaml` | Full-scale with real data |

## Implementation Path

The guide recommends a 5-stage implementation path:

1. **Local simulation** — single machine, synthetic data (`configs/dev.yaml`)
2. **Cloud pilot (synthetic)** — real infrastructure, synthetic data
3. **Representative data test** — real infrastructure, representative data
4. **Governed pilot** — full security controls, limited data
5. **Production** — full deployment with governance sign-off

## Development Deployment

```bash
# Install locally
pip install -e ".[dev]"

# Run smoke test
python runners/run_ec2.py fraud --synthetic

# Run full test suite
python tests/run_tests.py
```

No infrastructure provisioning needed. TLS and SecAgg are disabled in dev.

## Staging / Production Deployment

### Prerequisites

- AWS/GCC accounts in Singapore region (ap-southeast-1)
- EC2 instances provisioned (see Infrastructure below)
- VPC with private subnets configured
- IAM roles with least-privilege access
- KMS keys for encryption

### Infrastructure Provisioning

```bash
# Option 1: Terraform
cd deploy/terraform/
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars
terraform init && terraform apply

# Option 2: Manual (see docs/Distributed_Deployment_Guide.md)
```

### Pre-flight Validation

```bash
# Validate all prerequisites
./scripts/preflight.sh

# Or check specific areas
./scripts/preflight.sh --check landing-zone --check iam --check endpoints
```

### Deployment Sequence

1. **Provision VPC and networking**
2. **Provision encryption keys (KMS)**
3. **Build and sign container images**
   ```bash
   docker build -t healthcare-fl:v1.0.0 .
   ```
4. **Establish CA and issue certificates**
   ```bash
   ./deploy/gen_mtls_certs.sh
   ```
5. **Deploy coordinator**
   ```bash
   cp deploy/cluster.env.template cluster.env
   # Edit cluster.env with production values
   ./deploy/distributed/deploy.sh up
   ```
6. **Deploy clients** — per participant
7. **Distribute certificates**
8. **Configure observability** (CloudWatch, dashboards)
9. **Run smoke test over deployed infrastructure**
   ```bash
   python runners/run_ec2.py fraud --synthetic
   ```
10. **Confirm mTLS connectivity**
    ```bash
    ./deploy/health_check.sh
    ```
11. **Enable SecAgg and validate quorum**
12. **Enable DP and validate budget accounting**
13. **Handover to operations**

### Post-deployment Validation

```bash
# Full health check
./deploy/health_check.sh

# Validate configuration
./deploy/validate_config.sh

# Run diagnostic bundle
./scripts/diagnose.sh --run-id smoke-001 --env production --since 2h
```

### Rolling Upgrades

1. Prepare new release (build + sign new image)
2. Validate in staging
3. Stage clients one at a time (canary deployment)
4. Monitor for regressions
5. Rollback if needed: `./deploy/rollback.sh`

## Detailed Multi-Node Setup

For comprehensive distributed deployment instructions including Docker, TLS, GPU setup, and orchestration, see:

- [Distributed Deployment Guide](Distributed_Deployment_Guide.md)
- [FL Production Technical Reference](FL_Production_Technical_Reference.md)
