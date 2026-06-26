# Configuration Reference

All configurable parameters for the FL platform.

## Configuration Files

| File | Format | Purpose |
|------|--------|---------|
| `deploy/env.example.yaml` | YAML | Master configuration template |
| `deploy/cluster.env.template` | Shell env | Legacy deployment configuration |
| `deploy/configs/dev.yaml` | YAML | Development environment |
| `deploy/configs/staging.yaml` | YAML | Staging environment |
| `deploy/configs/production.yaml` | YAML | Production environment |
| `scenarios/*.yaml` | YAML | Experiment definitions |

## Infrastructure Parameters

### Server

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server.host` | (required) | Coordinator public IP or hostname |
| `server.private_ip` | (required) | Coordinator private IP within VPC |
| `server.instance_type` | `g6.8xlarge` | EC2 instance type |
| `server.memory` | `120g` | Container memory limit |
| `server.cpus` | `30` | Container CPU limit |
| `server.shm_size` | `8g` | Shared memory for PyTorch DataLoader |

### Clients

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clients.hosts` | (required) | List of client IPs |
| `clients.num_clients` | `5` | Number of FL clients |
| `clients.instance_type` | `g6.4xlarge` | EC2 instance type |
| `clients.memory` | `56g` | Container memory limit |
| `clients.cpus` | `14` | Container CPU limit |

### TLS

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tls.certs_dir` | `/home/ec2-user/fl-deploy/certs` | Certificate directory |
| `tls.cert_days` | `365` | Certificate validity (days) |
| `tls.cert_cn` | `fl-coordinator` | Certificate common name |

### Network

| Parameter | Default | Description |
|-----------|---------|-------------|
| `network.grpc_port` | `9092` | Flower gRPC server port |
| `network.vpc_id` | (optional) | AWS VPC ID |
| `network.subnet_id` | (optional) | AWS Subnet ID |

## Training Parameters

### Task Configuration

| Parameter | Description | Valid Values |
|-----------|-------------|-------------|
| `task` | Task/dataset name | `fraud`, `sepsis`, `ecg`, `anomaly`, `mortality`, `drug`, `readmission`, `satellite`, `chest_xray`, `gov_llm`, `generic` |
| `num_clients` | Number of FL clients | 2+ |
| `num_rounds` | Training rounds | 1+ |
| `strategies` | FL strategies to run | See strategies table below |
| `synthetic` | Use synthetic data | `true` / `false` |
| `max_samples` | Max samples per client | 0 = unlimited |

### FL Strategies

| Strategy | Description |
|----------|-------------|
| `IID` | FedAvg with IID data partitioning |
| `FedAvg` | Federated Averaging |
| `FedProx` | FedAvg with proximal term (mu=0.01) |
| `SCAFFOLD` | Variance reduction with control variates |
| `FedAdam` | Server-side Adam optimiser |
| `FedYogi` | Server-side Yogi optimiser |
| `SecAgg` | FedAvg + Secure Aggregation |
| `DP-Central` | FedAvg + server-side DP (DP_STRONG) |
| `DP-Local` | FedAvg + client-side DP (DP_MODERATE) |
| `DP-Local-Low` | FedAvg + client-side DP (DP_RELAXED) |
| `OneOwner` | Centralised baseline (single client) |

## Privacy Parameters

### Differential Privacy Presets

| Preset | Noise Multiplier (sigma) | Clipping Norm (C) | ~Epsilon @ 100 rounds (delta=1e-5) |
|--------|--------------------------|--------------------|------------------------------------|
| `DP_STRONG` (default) | 1.5 | 1.0 | ~4 |
| `DP_MODERATE` | 0.8 | 1.0 | ~10 |
| `DP_RELAXED` | 0.5 | 1.0 | ~25 |

**Fail-closed behaviour:** Tasks that do not specify a DP preset default to `DP_STRONG`.

Custom DP configuration:
```yaml
privacy:
  sigma: 1.2        # Custom noise multiplier
  max_norm: 1.0     # Clipping norm
  delta: 1.0e-5     # Delta parameter
```

Compute budget for any configuration:
```bash
python tools/dp_budget.py --sigma 1.2 --rounds 100 --delta 1e-5
python tools/dp_budget.py --all --rounds 100
```

### Secure Aggregation

SecAgg is configured via the `fl_pets/secagg.py` Python API.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scale` | `0.01` | Mask magnitude |
| `min_quorum` | `2` | Minimum clients for SecAgg |
| `max_abort_rate` | `0.20` | Abort rate alarm threshold |
| `dropout_tolerant` | `true` | Handle client dropout |

## Monitoring Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitoring.log_driver` | `json-file` | Docker log driver (`json-file` or `awslogs`) |
| `monitoring.log_max_size` | `200m` | Max log file size |
| `monitoring.log_max_file` | `5` | Max log files to retain |

## Backup Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backup.backup_dir` | `/home/ec2-user/fl-deploy/backups` | Local backup directory |
| `backup.s3_bucket` | (optional) | S3 bucket for remote backups |
| `backup.retention_days` | `30` | Backup retention period |

## Timeout Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.timeout_default` | `3600` (1h) | Default per-task timeout |
| `training.timeout_large` | `54000` (15h) | Large model tasks (chest X-ray, transfer) |
| `training.timeout_medium` | `7200` (2h) | Medium tasks (ECG, satellite) |
| `training.round_timeout` | `120` (2min) | Per-round timeout |
