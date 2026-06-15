# Cost Reporting

Cost tracking methodology and reporting template for FL platform operations.

## Cost Categories

| # | Category | One-time | Monthly | Notes |
|---|----------|----------|---------|-------|
| 1 | Coordinator compute | EC2 provisioning | Instance hours + GPU | g6.8xlarge in ap-southeast-1 |
| 2 | Client compute | EC2 provisioning | Instance hours + GPU (per participant) | g6.4xlarge per client |
| 3 | Key management | KMS key creation | KMS API calls | Per-key monthly charge |
| 4 | Logging & monitoring | CloudWatch setup | Log ingestion + storage + dashboards | Scales with log volume |
| 5 | Storage | EBS/S3 provisioning | EBS volumes + S3 storage + data transfer | Model checkpoints, run records |
| 6 | PKI / certificates | CA setup | Certificate operations | Minimal ongoing cost |
| 7 | Security assurance | Penetration testing | Vulnerability scanning | Annual or per-release |
| 8 | AI governance | DPIA, model review | Ongoing review cycles | Staff time |
| 9 | Operational support | On-call setup | On-call rotation, incident response | Staff time |

## Cost Estimation

### Compute Cost Formula

```
Monthly coordinator cost = hours_per_month * instance_rate
Monthly client cost = hours_per_month * instance_rate * num_clients
Monthly GPU cost = gpu_hours * gpu_rate

Total compute = coordinator + client + GPU
```

### Storage Cost Formula

```
Model checkpoints = num_models * avg_model_size * retention_months * s3_rate
Run records = num_runs * avg_record_size * retention_months * s3_rate
Audit logs = daily_log_volume * 30 * retention_years * 12 * s3_rate
EBS volumes = total_ebs_gb * ebs_rate * num_instances
```

## Cost Tagging Strategy

Apply these tags to all AWS resources:

| Tag Key | Example Value | Purpose |
|---------|---------------|---------|
| `Project` | `fl-reference` | Cost allocation |
| `Environment` | `production` | Per-environment tracking |
| `Component` | `coordinator` / `client` / `monitoring` | Per-component breakdown |
| `CostCenter` | (your cost center) | Organisational accounting |

## Reporting Template

### Monthly Cost Report

| Category | Budget | Actual | Variance | Notes |
|----------|--------|--------|----------|-------|
| Coordinator compute | | | | |
| Client compute (N clients) | | | | |
| Key management | | | | |
| Logging & monitoring | | | | |
| Storage (S3 + EBS) | | | | |
| Data transfer | | | | |
| **Total** | | | | |

### Cost Optimisation Opportunities

- **Spot instances** for non-production workloads (staging, testing)
- **Reserved instances** for production coordinator (1-year commitment)
- **Log retention tiering** — move old logs to S3 Glacier after 90 days
- **Right-sizing** — monitor actual CPU/GPU utilisation and adjust instance types
- **Training scheduling** — run training during off-peak hours where possible

## FL vs Centralised Cost Comparison

| Factor | Federated | Centralised |
|--------|-----------|-------------|
| Compute | N client instances + 1 coordinator | 1 large instance |
| Communication | gRPC round-trips per round | One-time data transfer |
| Storage | Distributed (each site) | Centralised |
| Privacy overhead | DP noise, SecAgg masks | None |
| Governance | Federation agreements | Data sharing agreements |
| Training time | Longer (communication rounds) | Shorter |
| Data movement | None (key advantage) | Full dataset transfer |
