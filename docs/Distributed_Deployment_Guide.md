# Federated Learning Platform -- Distributed Deployment Guide

**Version:** 2.1
**Last updated:** 2026-05-28
**Platform:** AWS EC2 (configurable region), Amazon Linux 2023, NVIDIA L4 GPU

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Prerequisites and Dependencies](#2-prerequisites-and-dependencies)
3. [Configuration Management](#3-configuration-management)
4. [Infrastructure Setup](#4-infrastructure-setup)
5. [Docker Image](#5-docker-image)
6. [Data Pipeline](#6-data-pipeline)
7. [TLS and PKI](#7-tls-and-pki)
8. [Deployment](#8-deployment)
9. [Tasks and Strategies](#9-tasks-and-strategies)
10. [Monitoring and Observability](#10-monitoring-and-observability)
11. [Security](#11-security)
12. [Backup and Recovery](#12-backup-and-recovery)
13. [Capacity Planning](#13-capacity-planning)
14. [Version Management and Rollback](#14-version-management-and-rollback)
15. [Incident Response Runbook](#15-incident-response-runbook)
16. [Cost Management](#16-cost-management)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Architecture

### 1.1 Overview

```
                      +-------------------------------+
                      |     FL Server (aggregator)     |
                      |     start_server() :9092       |
                      |     TLS (EC P-256)             |
                      |     32 vCPU / 128 GB / L4      |
                      +---------------+----------------+
                                      | gRPC + TLS
              +-----------------------+-----------------------+
              |            |          |          |             |
        +-----+-----+ +---+------+ +-+-------+ ++--------+ +-+--------+
        | Client 0  | | Client 1 | | Client 2| | Client 3| | Client 4 |
        | L4 24GB   | | L4 24GB  | | L4 24GB | | L4 24GB | | L4 24GB  |
        | Site A    | | Site B   | | Site C  | | Site D  | | Site E   |
        +-----------+ +----------+ +---------+ +---------+ +----------+
```

### 1.2 Components

| Component | Role | Key Files |
|-----------|------|-----------|
| **FL Server** | Aggregation, strategy execution, results collection | `run_ec2.py --distributed` |
| **FL Client** | Local training, model updates, pre-flight data check | `run_client.py` |
| **Docker Image** | Unified runtime for server + clients | `Dockerfile` (all deps pinned) |
| **TLS / mTLS** | Server + per-client certificates (EC P-256) | `deploy/gen_mtls_certs.sh` |
| **Orchestrator** | Automated deploy, run, collect | `run_server_side.sh` in `docker:cli` |
| **Data Pipeline** | Client-side ingestion, validation, manifests | `ingest.py`, `fl_common/data.py` |
| **Adapter Framework** | Federated LoRA for any HuggingFace model | `fl_common/federated_adapter.py` |
| **Secure Inference** | CKKS homomorphic encryption (TenSEAL) | `secure_inference/tenseal_inference.py` |

### 1.3 Communication Flow

1. Orchestrator stops existing containers, starts clients (reconnect loop)
2. Orchestrator starts server (`start_server()` on port 9092 with TLS)
3. Clients connect to server via private IP, authenticate with CA cert
4. Server sends strategy config (strategy name, learning rate, epochs, partition type)
5. Clients train locally, send model updates
6. Server aggregates, evaluates, advances to next round
7. After all rounds, server moves to next strategy (clients reconnect)
8. After all strategies, server exits, orchestrator moves to next task

### 1.4 Important: No Separate SuperLink

`run_ec2.py --distributed` calls `start_server()` directly, which binds port 9092 and acts as the gRPC server. **Do not run a separate SuperLink container** -- it will conflict on port 9092 and cause `Port in server address 0.0.0.0:9092 is already in use`.

The SuperLink is only needed for idle monitoring between runs, and must be stopped before any training starts.

### 1.5 Network

| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| 9092 | gRPC + TLS | Client -> Server | FL model updates |
| 22 | SSH | Orchestrator -> All | Deployment, monitoring |

---

## 2. Prerequisites and Dependencies

### 2.1 Software Versions

| Component | Minimum Version | Validated Version |
|-----------|----------------|-------------------|
| Docker | 24.0+ | 25.0.14 |
| NVIDIA Driver | 550+ | 570.124.06 |
| CUDA (container) | 12.4+ | 12.4 (PyTorch cu124) |
| PyTorch | 2.5+ | 2.5.1+cu124 |
| Flower (flwr) | 1.30+ | 1.30.0 |
| Python | 3.12+ | 3.12.13 |
| NumPy | 1.26+ | 1.26.4 |
| Pandas | 3.0+ | 3.0.3 |
| Scikit-learn | 1.8+ | 1.8.0 |
| TenSEAL | 0.3.16 | 0.3.16 |
| OS | Amazon Linux 2023 | Amazon Linux 2023 |
| nvidia-container-toolkit | 1.19+ | 1.19.0 |

### 2.2 AWS Prerequisites

- VPC with private subnets for inter-node communication
- IAM role with EC2 describe/start/stop permissions for cost management
- S3 bucket for result archival (optional but recommended)
- CloudWatch agent installed on all nodes (recommended)
- KMS key for secret encryption (see Section 11)
- SSH key pair created in EC2 console (`.pem` format)

### 2.3 Operator Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.5 (if using IaC, recommended)
- `openssl` >= 3.0 for certificate generation
- Access to a private Docker registry or ECR repository

---

## 3. Configuration Management

### 3.1 Cluster Configuration File

Create `cluster.env` on the operator workstation. **This file must not be committed to version control.**

```bash
# cluster.env -- Cluster configuration
# Copy to cluster.env.local and fill in values

# --- Infrastructure ---
AWS_REGION=ap-southeast-1
VPC_ID=vpc-XXXXXXXXXXXXXXXXX
SUBNET_ID=subnet-XXXXXXXXXXXXXXXXX
KEY_PAIR_NAME=<your-key-pair>
KEY_PATH=~/.ssh/<your-key>.pem
SECURITY_GROUP_ID=sg-XXXXXXXXXXXXXXXXX

# --- Cluster Nodes ---
SERVER_IP=<server-private-ip>
SERVER_PUBLIC_IP=<server-public-ip>
CLIENT_IPS="<client1-ip> <client2-ip> <client3-ip> <client4-ip> <client5-ip>"
NUM_CLIENTS=5

# --- Image ---
FL_IMAGE=healthcare-fl
FL_IMAGE_TAG=latest
REGISTRY=<account-id>.dkr.ecr.<region>.amazonaws.com

# --- TLS ---
CERTS_DIR=~/fl-deploy/certs
CA_CERT=${CERTS_DIR}/ca.pem
SERVER_CERT=${CERTS_DIR}/server.pem
SERVER_KEY=${CERTS_DIR}/server.key

# --- Data ---
DATA_DIR=~/fl-deploy/data
RESULTS_DIR=~/fl-deploy/results

# --- Timeouts (seconds) ---
TIMEOUT_DEFAULT=3600
TIMEOUT_DENSENET=54000
TIMEOUT_MEDIUM=7200
```

### 3.2 Loading Configuration

All scripts source this file:

```bash
source cluster.env

# Validate required variables
for var in SERVER_IP CLIENT_IPS KEY_PATH CERTS_DIR; do
  [ -z "${!var}" ] && echo "ERROR: $var not set in cluster.env" && exit 1
done
```

### 3.3 Per-Environment Overrides

Maintain separate config files per environment:

```
cluster.env.staging
cluster.env.production
```

Symlink the active environment:

```bash
ln -sf cluster.env.production cluster.env
```

---

## 4. Infrastructure Setup

### 4.1 EC2 Instance Sizing

#### Recommended Configuration

| Role | Instance Type | Count | GPU | vCPU | RAM | Storage |
|------|--------------|-------|-----|------|-----|---------|
| Server | g6.8xlarge | 1 | L4 24GB | 32 | 128 GB | 500 GB gp3 |
| Client | g6.4xlarge | N | L4 24GB | 16 | 64 GB | 1 TB gp3 |

#### Budget Configuration (CPU-only clients)

| Role | Instance Type | Count | GPU |
|------|--------------|-------|-----|
| Server | g6.4xlarge | 1 | L4 24GB |
| Client | t3.xlarge | N | None |

### 4.2 Provisioning

```bash
source cluster.env

# Using Terraform (recommended)
cd deploy/terraform
terraform init && terraform apply -var-file=production.tfvars

# Or AWS CLI
aws ec2 run-instances \
  --image-id <ami-id> \
  --instance-type g6.4xlarge \
  --count ${NUM_CLIENTS} \
  --key-name ${KEY_PAIR_NAME} \
  --security-group-ids ${SECURITY_GROUP_ID} \
  --subnet-id ${SUBNET_ID} \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":1000,"VolumeType":"gp3","Encrypted":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fl-client},{Key=Environment,Value=production}]'
```

**Note:** Always enable EBS encryption. Use a KMS CMK for regulated workloads.

### 4.3 GPU Driver Installation

Required on all GPU instances.

```bash
ssh -i ${KEY_PATH} ec2-user@<NODE_IP>

# Add NVIDIA CUDA repo
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo

# Install driver
sudo dnf install -y nvidia-driver nvidia-driver-cuda nvidia-driver-libs

# Load kernel module
sudo modprobe nvidia

# Verify
nvidia-smi
# Expected: NVIDIA L4, Driver 595.x, CUDA 13.x
```

### 4.4 Docker Setup

```bash
# Install Docker
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

---

## 5. Docker Image

### 5.1 Build

```bash
cd /path/to/healthcare-fl
docker build -t ${FL_IMAGE}:${FL_IMAGE_TAG} -f Dockerfile .

# Image includes:
#   - Python 3.12, PyTorch 2.5.1+cu124, Flower 1.29
#   - All models: bilstm, mlp, densenet, autoencoder, logreg, cnn1d,
#     tabnet, resnet_small, vfl_mlp, split_bilstm, generic, mistral
#   - All tasks + generic config-driven pipeline
#   - FL strategies, privacy mechanisms, secure inference, scenarios
#   - run_ec2.py (server), run_client.py (client)
#
# Orchestrator uses docker:cli (Alpine + Docker CLI + SSH)
```

### 5.2 Push to Registry (Recommended)

```bash
# Tag and push to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${REGISTRY}

docker tag ${FL_IMAGE}:${FL_IMAGE_TAG} ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG}
docker push ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG}

# Pull on each node
for ip in ${SERVER_IP} ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "aws ecr get-login-password --region ${AWS_REGION} | \
     docker login --username AWS --password-stdin ${REGISTRY} && \
     docker pull ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG} && \
     docker tag ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG} ${FL_IMAGE}:${FL_IMAGE_TAG}"
done
```

### 5.3 Distribute via SCP (Alternative)

Use this when ECR is unavailable. **Always distribute from the server, not from a local machine.** VPC internal network transfers at gigabit speeds (~30s per client).

```bash
# On server: save image
docker save ${FL_IMAGE}:${FL_IMAGE_TAG} | gzip > /tmp/fl-image.tar.gz
# Size: ~3.2 GB compressed

# Distribute to all clients in parallel
for ip in ${CLIENT_IPS}; do
  (
    scp -i ${KEY_PATH} -o StrictHostKeyChecking=no \
      /tmp/fl-image.tar.gz ec2-user@${ip}:/tmp/
    ssh -i ${KEY_PATH} -o StrictHostKeyChecking=no ec2-user@${ip} \
      "sudo docker load < /tmp/fl-image.tar.gz && rm /tmp/fl-image.tar.gz"
    echo "$ip: DONE"
  ) &
done
wait
```

### 5.4 Container Security

All containers should run with these hardening flags:

```bash
# Production container flags
--read-only \                           # Read-only root filesystem
--tmpfs /tmp:rw,noexec,nosuid,size=2g \ # Writable tmp with limits
--security-opt no-new-privileges \      # Prevent privilege escalation
--cap-drop ALL \                        # Drop all capabilities
--cap-add SYS_NICE \                    # Only add what's needed (GPU scheduling)
--memory 60g \                          # Memory limit (adjust per instance)
--memory-swap 60g \                     # No swap
--cpus 14 \                             # CPU limit
--pids-limit 512 \                      # Prevent fork bombs
--log-opt max-size=100m \               # Log rotation
--log-opt max-file=5 \                  # Keep 5 rotated log files
```

**Note:** `--user` (non-root) is recommended but requires the Dockerfile to support it. If the base image runs as root, add a `USER` directive to the Dockerfile.

---

## 6. Data Pipeline

### 6.1 Data Residency

In federated learning, **data stays at each site**. Each participating organization ingests its own data locally using `ingest.py`. The server never sees raw data -- only metadata manifests.

### 6.2 Data Ingestion

Run on each client node:

```bash
# Ingest local data
python ingest.py --task sepsis --input /mnt/ehr/sepsis_cohort.csv --client-id site_a

# Validate existing ingested data
python ingest.py --task sepsis --validate-only

# View manifest
python ingest.py --show-manifest ~/fl-deploy/data/sepsis
```

### 6.3 What ingest.py Does

1. Reads input data (CSV, NPZ, or image directory)
2. **Validates** -- checks shape, types, NaN/Inf, label distribution. Blocks on errors.
3. Converts to standardized format (`data.npz` with `X` and `y` arrays)
4. Generates `manifest.json` -- schema, sample count, SHA-256 checksum, label distribution
5. Outputs to `~/fl-deploy/data/<task>/`

### 6.4 Data Directory Layout

```
~/fl-deploy/data/<task>/
  manifest.json       -- DataManifest (schema, counts, checksums)
  data.npz            -- features (X) and labels (y)
  OR data.csv         -- tabular data with header row
  OR images/          -- image directory (chest_xray, satellite)
      metadata.csv    -- image paths and labels
```

### 6.5 Data Validation Gates

Before any training run, validate data integrity:

```bash
# Verify manifest checksums match data files
python ingest.py --task <TASK> --validate-only

# Pre-flight check in run_client.py logs:
# Data: npz, 12000 samples, checksum a1b2c3d4e5f6
# Starting FL client: task=sepsis ... data=real
```

`run_client.py` performs these checks at startup:
- Manifest exists and is valid JSON
- SHA-256 checksum of data file matches manifest
- Feature dimensions match task config
- Label values are within expected range
- If any check fails, the client logs the error and falls back to synthetic data

### 6.6 Data Versioning

Tag each data release with a version in the manifest:

```json
{
  "version": "2026-05-01",
  "task": "sepsis",
  "samples": 12000,
  "sha256": "a1b2c3...",
  "schema": {"features": 34, "labels": 2},
  "created": "2026-05-01T00:00:00Z"
}
```

Keep previous data versions in `~/fl-deploy/data/<task>/archive/` for reproducibility.

### 6.7 Data Retention Policy

| Data Type | Retention | Storage |
|-----------|-----------|---------|
| Raw ingested data | Per organization policy | Local EBS |
| Processed NPZ/CSV | Duration of engagement + 90 days | Local EBS |
| Manifests | Indefinite | S3 archive |
| Synthetic data | Ephemeral (regenerated per run) | tmpfs |
| Training results | 1 year minimum | S3 archive |

---

## 7. TLS and PKI

### 7.1 Certificate Authority

For production deployments, use an organizational CA or AWS Private CA (ACM PCA). The self-signed CA procedure below is for environments where a managed PKI is unavailable.

#### Self-Signed CA (when managed PKI is unavailable)

```bash
cd deploy/distributed
mkdir -p certs

# Generate CA key and certificate
openssl ecparam -genkey -name prime256v1 -out certs/ca.key
openssl req -new -x509 -key certs/ca.key -out certs/ca.pem \
  -days 365 -subj "/CN=FL-Platform-CA/O=<YOUR_ORG>"
```

#### AWS Private CA (recommended)

```bash
# Create a subordinate CA in ACM PCA for FL platform use
aws acm-pca create-certificate-authority \
  --certificate-authority-type SUBORDINATE \
  --certificate-authority-configuration \
    "KeyAlgorithm=EC_prime256v1,SigningAlgorithm=SHA256WITHECDSA,Subject={CommonName=FL-Platform-CA,Organization=<YOUR_ORG>}"
```

### 7.2 Server Certificate

```bash
# Generate server cert with SAN
cat > certs/san.cnf << EOF
[req]
distinguished_name=dn
req_extensions=v3
prompt=no
[dn]
CN=fl-server
[v3]
subjectAltName=DNS:fl-server,DNS:localhost,IP:127.0.0.1,IP:${SERVER_IP}
EOF

openssl ecparam -genkey -name prime256v1 -out certs/server.key
openssl req -new -key certs/server.key -out certs/s.csr -config certs/san.cnf
openssl x509 -req -in certs/s.csr -CA certs/ca.pem -CAkey certs/ca.key \
  -CAcreateserial -out certs/server.pem -days 365 \
  -extfile certs/san.cnf -extensions v3

# Clean up signing artifacts
rm -f certs/{s.csr,san.cnf,ca.srl}

# Secure the CA key -- store it offline or in a secrets manager
# Do NOT leave ca.key on any cluster node
chmod 400 certs/ca.key certs/server.key

# Distribution:
#   certs/ca.pem      -> ALL nodes (public, read-only)
#   certs/server.pem  -> server only (read-only)
#   certs/server.key  -> server only (secret, 0400 permissions)
```

### 7.3 mTLS (Per-Client Certificates)

For production, each client should have its own certificate so the server can verify client identity.

```bash
# Generate CA + server cert + per-client certs in one step
./deploy/gen_mtls_certs.sh --full

# Output:
#   certs/ca.pem              -> all nodes
#   certs/server.pem/key      -> server only
#   certs/client_0.pem/key    -> client 0 only
#   certs/client_1.pem/key    -> client 1 only
#   ...

# Add a new client later
./deploy/gen_mtls_certs.sh --add-client 5

# Verify all certs
./deploy/gen_mtls_certs.sh --verify
```

Client certificates are auto-detected by `run_client.py` — if `client_N.pem` and `client_N.key` exist in the certs directory, mTLS is enabled automatically. No configuration needed.

**Limitation:** Flower's deprecated `start_server()` API does not enforce client certificate verification on the server side. The client certs are generated and transmitted but server-side validation requires migrating to Flower's `ServerApp` API.

### 7.4 Certificate Rotation

```bash
# Check certificate expiry
./deploy/rotate_certs.sh --check

# Full rotation: generate new certs, distribute, restart, verify TLS
./deploy/rotate_certs.sh --full

# Generate only (no distribution)
./deploy/rotate_certs.sh --generate
```

Rotation backs up old certs before overwriting. Automate expiry monitoring:

```bash
# Add to cron on operator workstation (check weekly)
0 9 * * 1 /path/to/deploy/rotate_certs.sh --check | grep -q "EXPIRING\|EXPIRED" && \
  echo "FL cert rotation needed" | mail -s "CERT EXPIRY WARNING" ops@example.com
```

---

## 8. Deployment

### 8.1 Server-Side Orchestrator (Recommended)

Run the orchestrator as a Docker container on the server. This ensures the run completes even if the operator's workstation disconnects.

```bash
source cluster.env

# Upload orchestrator script and SSH key to server (one-time setup)
scp -i ${KEY_PATH} run_server_side.sh ec2-user@${SERVER_PUBLIC_IP}:~/
scp -i ${KEY_PATH} ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP}:~/.ssh/$(basename ${KEY_PATH})
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "chmod 600 ~/.ssh/$(basename ${KEY_PATH})"

# Launch orchestrator container
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "
docker run -d --name fl-orchestrator \
  --restart on-failure:3 \
  --network host \
  --memory 4g \
  --cpus 2 \
  --pids-limit 256 \
  --log-opt max-size=200m \
  --log-opt max-file=5 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/.ssh/$(basename ${KEY_PATH}):/keys/deploy.pem:ro \
  -v ~/fl-deploy/certs:${CERTS_DIR}:ro \
  -v ~/fl-deploy/results:${RESULTS_DIR} \
  -v ~/run_server_side.sh:/run.sh:ro \
  docker:cli \
  sh -c 'apk add --no-cache openssh-client bash >/dev/null 2>&1 && bash /run.sh all'
"

# Monitor progress (can disconnect and reconnect anytime)
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-orchestrator --tail 20"
```

**Available targets:**
- `all` -- all tasks (~6-8 hours)
- `failed` -- re-run only previously failed tasks
- `fraud` / `sepsis` / `ecg` / etc. -- single task

**What the orchestrator does per task:**
1. Kills `fl-training`, `fl-superlink`, and all client containers
2. Starts `fl-client` containers on all clients (reconnect loop)
3. Starts `fl-training` on server (`run_ec2.py --distributed <task>`)
4. Monitors server container every 30s, with a per-task timeout
5. On completion or timeout, stops clients, prints summary
6. Moves to next task

**Per-task timeouts:**

| Task | Timeout | Reason |
|------|---------|--------|
| chest, transfer | 15 hours | DenseNet-121 (8M params, image data) |
| ecg, satellite | 2 hours | Medium models with 11 strategies |
| All others | 1 hour | Small models, fast convergence |

### 8.2 Manual Deployment

#### Start Server

```bash
source cluster.env

ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP}

docker run -d --name fl-training --network host \
  --restart on-failure:3 \
  --memory 120g \
  --cpus 30 \
  --log-opt max-size=100m \
  --log-opt max-file=5 \
  --health-cmd "python3 -c 'import socket; s=socket.socket(); s.connect((\"127.0.0.1\",9092)); s.close()'" \
  --health-interval=30s \
  --health-timeout=5s \
  --health-retries=3 \
  -v ~/fl-deploy/certs:/certs:ro \
  -v ~/fl-deploy/results:/app/results \
  -e FL_DISTRIBUTED=1 \
  -e SUPERLINK_ADDRESS=0.0.0.0:9092 \
  -e CERTS_DIR=/certs \
  -e PYTHONUNBUFFERED=1 \
  ${FL_IMAGE}:${FL_IMAGE_TAG} \
  python3 run_ec2.py --distributed fraud
```

#### Start Clients

```bash
source cluster.env
PARTITION=0

for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "
    sudo docker run -d --name fl-client --network host --gpus all \
      --restart on-failure:3 \
      --memory 56g \
      --cpus 14 \
      --log-opt max-size=100m \
      --log-opt max-file=5 \
      --health-cmd 'pgrep -f run_client.py' \
      --health-interval=30s \
      --health-timeout=5s \
      --health-retries=3 \
      -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro \
      -v ~/fl-deploy/data:/data:ro \
      -e PARTITION_ID=${PARTITION} \
      -e NUM_CLIENTS=${NUM_CLIENTS} \
      -e FL_TASK=fraud \
      -e FL_SERVER=${SERVER_IP}:9092 \
      -e CERTS_DIR=/certs \
      -e PYTHONUNBUFFERED=1 \
      ${FL_IMAGE}:${FL_IMAGE_TAG} \
      python3 run_client.py
  "
  PARTITION=$((PARTITION + 1))
done
```

### 8.3 Server-to-Client SSH Key Setup

The server needs SSH access to clients for the orchestrator to manage client containers.

```bash
source cluster.env

# Copy key to server (one-time)
scp -i ${KEY_PATH} ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP}:~/.ssh/deploy.pem
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "chmod 600 ~/.ssh/deploy.pem"

# Verify connectivity to each client
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
    "ssh -i ~/.ssh/deploy.pem -o StrictHostKeyChecking=no ec2-user@${ip} 'echo OK'"
done
```

### 8.4 Environment Variables

#### Server

| Variable | Description | Default |
|----------|-------------|---------|
| `FL_DISTRIBUTED` | Enable distributed mode (1/0) | `0` (simulation) |
| `SUPERLINK_ADDRESS` | gRPC listen address | `0.0.0.0:9092` |
| `CERTS_DIR` | TLS certificates directory | `/certs` |
| `SYNTHETIC` | Use synthetic data (1/0) | `0` |
| `DATA_PATH` | Path to data directory | `/data` |

#### Client

| Variable | Description | Default |
|----------|-------------|---------|
| `PARTITION_ID` | Client index (0 to N-1) | `0` |
| `NUM_CLIENTS` | Total number of clients | `5` |
| `FL_TASK` | Task name | `fraud` |
| `FL_SERVER` | Server address (private_ip:port) | Required |
| `CERTS_DIR` | CA certificate directory | `/certs` |
| `DATA_PATH` | Path to local data | `/data` |
| `MAX_SAMPLES` | Cap dataset size (0=unlimited) | `0` |
| `SYNTHETIC` | Use synthetic data (1/0) | `0` |
| `DATASET_PATH` | Image data directory | `/data/chest_xray` |
| `CSV_PATH` | Image metadata CSV | `Data_Entry_2017.csv` |

---

## 9. Tasks and Strategies

### 9.1 Task Matrix

| Task | Model | Params | Strategies | Distributed Time | Last Verified |
|------|-------|--------|-----------|-----------------|---------------|
| **fraud** | MLP | 50K | 11 | 68s | 2026-05-27, 11/11 PASS |
| **sepsis** | BiLSTM | 500K | 11 | 98s | 2026-05-28, 11/11 PASS |
| **ecg** | BiLSTM | 200K | 11 | ~5 min | 9/11 PASS (DP strategies use CPU fallback) |
| **anomaly** | Autoencoder | 500K | 11 | 69s | 11/11 PASS |
| **mortality** | TabNet | 1M | 11 | 68s | 11/11 PASS |
| **drug** | Generic MLP | 50K | 11 | ~3 min | Needs re-test |
| **readmission** | LogReg | 10K | 11 | 69s | 11/11 PASS |
| **satellite** | ResNet-small | 5M | 7 | 339s | 7/7 PASS |
| **chest** | DenseNet-121 | 8M | 7 | 429s (synthetic) | 7/7 PASS |
| **vfl** | VFL MLP | 50K | 4 | 38s | 4/4 PASS |
| **split** | Split BiLSTM | 500K | 3 | 38s | 3/3 PASS |
| **transfer** | DenseNet-121 | 8M | 2 | 69s (synthetic) | 2/2 PASS |
| **olmo** | OLMo-1B QLoRA | 2.1M (LoRA) | 3 | 96s | 2026-05-28, perplexity 1.13 |
| **privacy** | BiLSTM + MLP | - | 3 attacks | ~10 min | PASS |

#### Federated Adapter Framework

The `olmo` task uses the generic federated adapter framework (`fl_common/federated_adapter.py`). This framework supports any HuggingFace model — change the preset to switch models with no code changes:

| Preset | Model | Adapter/Round | Base Size (4-bit) | Use Case |
|--------|-------|--------------|-------------------|----------|
| `olmo-1b` | allenai/OLMo-1B-hf | 32 MB | 0.5 GB | Gov documents (tested) |
| `llama-3-8b` | meta-llama/Meta-Llama-3-8B | 256 MB | 4.0 GB | General LLM |
| `mistral-7b` | mistralai/Mistral-7B-v0.3 | 224 MB | 3.5 GB | General LLM |
| `phi-3-mini` | microsoft/Phi-3-mini-4k-instruct | 16 MB | 0.2 GB | Lightweight LLM |
| `bert-base` | bert-base-uncased | 8 MB | 2.0 GB | Text classification/NER |
| `biobert` | dmis-lab/biobert-v1.1 | 8 MB | 2.0 GB | Medical NER |
| `vit-base` | google/vit-base-patch16-224 | 8 MB | 2.0 GB | Image classification |
| `whisper-small` | openai/whisper-small | 8 MB | 2.0 GB | Speech-to-text |

```bash
# Switch model by setting environment variable
ADAPTER_PRESET=llama-3-8b FL_TASK=olmo python run_client.py
```

Architecture inspired by FlexOLMo (AI2): base model frozen, only LoRA adapters (0.1-1% of params) are federated. Each agency trains on private data; server aggregates adapter weights.

### 9.2 Strategy Reference

| Strategy Name | Description | Non-IID | Privacy |
|--------------|-------------|---------|---------|
| `IID` | FedAvg with uniform data split | No | No |
| `FedProx_Mu0.1_Alpha_0.5` | FedProx, moderate non-IID | Moderate | No |
| `FedProx_Mu0.1_Alpha_0.1` | FedProx, extreme non-IID | Extreme | No |
| `SCAFFOLD_Alpha_0.5` | SCAFFOLD with control variates | Moderate | No |
| `SCAFFOLD_Alpha_0.1` | SCAFFOLD, extreme non-IID | Extreme | No |
| `SecAgg_Alpha_0.5` | Secure Aggregation (pairwise masks) | Moderate | Server can't see updates |
| `DP_Central_Eps50.0_Alpha_0.5` | Central DP, epsilon=50 | Moderate | Formal guarantee |
| `DP_Central_Eps10.0_Alpha_0.5` | Central DP, epsilon=10 | Moderate | Stronger guarantee |
| `DP_Local_Eps50.0_Alpha_0.5` | Local DP, epsilon=50 | Moderate | Client-side noise |
| `DP_Local_Eps10.0_Alpha_0.5` | Local DP, epsilon=10 | Moderate | Strong client noise |
| `OneOwner_Alpha_0.5` | Single owner, all contribute | Moderate | Access control |

---

## 10. Monitoring and Observability

### 10.1 Health Checks

#### Container Health

```bash
source cluster.env

# Check container health status across cluster
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker inspect --format='{{.Name}}: {{.State.Health.Status}}' fl-training 2>/dev/null || echo 'fl-training: not running'"

for ip in ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "sudo docker inspect --format='{{.Name}}: {{.State.Health.Status}}' fl-client 2>/dev/null || echo 'fl-client: not running'"
done
```

#### Readiness Probe

```bash
# Server readiness: verify gRPC port is listening
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "ss -tlnp | grep 9092 && echo 'READY' || echo 'NOT READY'"
```

### 10.2 Cluster Status

```bash
source cluster.env

# Server status
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker ps --format '{{.Names}}: {{.Status}}'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# All clients
for ip in ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "sudo docker ps --format '{{.Names}}: {{.Status}}'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
done
```

### 10.3 Training Monitoring

```bash
source cluster.env

# Live log stream (server)
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training -f --tail 20"

# Round progress
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training 2>&1 | grep 'Round [0-9]' | tail -5"

# Strategy completion
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training 2>&1 | grep 'Final'"

# Client logs
ssh -i ${KEY_PATH} ec2-user@<CLIENT_IP> "sudo docker logs fl-client --tail 10"
```

### 10.4 GPU Monitoring

```bash
source cluster.env

# GPU usage across cluster
for ip in ${SERVER_PUBLIC_IP} ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
done

# Continuous watch on server
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "watch -n 5 nvidia-smi"
```

### 10.5 Log Aggregation

#### CloudWatch (Recommended)

Configure Docker to ship logs to CloudWatch:

```bash
# /etc/docker/daemon.json on each node
{
  "log-driver": "awslogs",
  "log-opts": {
    "awslogs-region": "<REGION>",
    "awslogs-group": "/fl-platform/containers",
    "awslogs-create-group": "true",
    "tag": "{{.Name}}/{{.ID}}"
  }
}
```

#### File-Based (Alternative)

```bash
# Collect logs from all nodes
source cluster.env
mkdir -p logs/$(date +%Y%m%d)

ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training" > logs/$(date +%Y%m%d)/server.log 2>&1

IDX=0
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "sudo docker logs fl-client" > logs/$(date +%Y%m%d)/client_${IDX}.log 2>&1
  IDX=$((IDX + 1))
done
```

### 10.6 Alerting

Set up CloudWatch alarms for:

| Metric | Threshold | Action |
|--------|-----------|--------|
| GPU memory utilization | > 95% for 5 min | Notify ops |
| Container restart count | > 3 in 10 min | Notify ops, check logs |
| Disk usage | > 85% | Notify ops, prune images |
| Training round duration | > 2x baseline | Investigate straggler |
| gRPC port 9092 unreachable | > 60s | Page on-call |

```bash
# Example: CloudWatch alarm for disk usage
aws cloudwatch put-metric-alarm \
  --alarm-name fl-server-disk-usage \
  --metric-name disk_used_percent \
  --namespace CWAgent \
  --statistic Average \
  --period 300 \
  --threshold 85 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions <SNS_TOPIC_ARN>
```

---

## 11. Security

### 11.1 Network Segmentation

| Zone | Nodes | Allowed Traffic |
|------|-------|----------------|
| **Server subnet** | FL Server, Orchestrator | Inbound 9092 from client subnet only; SSH from bastion only |
| **Client subnet** | FL Clients | Outbound 9092 to server only; SSH from bastion only |
| **Bastion / VPN** | Operator access | SSH to server/client subnets |

Security group rules:

```bash
# Server security group
aws ec2 authorize-security-group-ingress \
  --group-id ${SG_SERVER} \
  --protocol tcp --port 9092 \
  --source-group ${SG_CLIENTS}

aws ec2 authorize-security-group-ingress \
  --group-id ${SG_SERVER} \
  --protocol tcp --port 22 \
  --source-group ${SG_BASTION}

# Client security group
aws ec2 authorize-security-group-ingress \
  --group-id ${SG_CLIENTS} \
  --protocol tcp --port 22 \
  --source-group ${SG_BASTION}
```

### 11.2 Secret Management

**Do not store SSH keys or TLS private keys on disk unprotected.** Use AWS Secrets Manager or SSM Parameter Store:

```bash
# Store SSH key in Secrets Manager
aws secretsmanager create-secret \
  --name fl-platform/ssh-key \
  --secret-binary fileb://${KEY_PATH}

# Store TLS server key
aws secretsmanager create-secret \
  --name fl-platform/server-tls-key \
  --secret-binary fileb://certs/server.key

# Retrieve at deployment time
aws secretsmanager get-secret-value \
  --secret-id fl-platform/ssh-key \
  --query SecretBinary --output text | base64 --decode > /tmp/deploy.pem
chmod 600 /tmp/deploy.pem
# Use /tmp/deploy.pem, then shred it after deployment
shred -u /tmp/deploy.pem
```

### 11.3 Audit Logging

Enable CloudTrail for AWS API calls and Docker audit logging:

```bash
# Docker daemon audit logging
# /etc/docker/daemon.json
{
  "log-level": "info",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  }
}

# Linux audit rules for Docker socket
sudo auditctl -w /var/run/docker.sock -k docker-socket
sudo auditctl -w /etc/docker -k docker-config
```

### 11.4 Security Checklist

#### Pre-Deployment

- [ ] TLS certificates generated with correct SANs (server private IP)
- [ ] SSH keys stored in Secrets Manager, not on local disk
- [ ] Security groups restrict port 9092 to client security group only
- [ ] Security groups restrict SSH (22) to bastion/VPN only
- [ ] No `--insecure` flags in any command
- [ ] EBS volumes encrypted with KMS CMK
- [ ] IMDSv2 enforced on all instances
- [ ] Docker containers run with resource limits

#### Data

- [ ] Patient data only on client machines, never on server
- [ ] Data directories mounted read-only (`:ro`) in Docker
- [ ] Data manifests validated before training

#### Operations

- [ ] Instances stopped when not in use
- [ ] Docker logs have size limits and rotation
- [ ] GPU driver matches CUDA version in Docker image
- [ ] Results collected and backed up before instance termination
- [ ] Certificate expiry monitored

#### Post-Training

- [ ] MIA attack run on final model (check for data leakage)
- [ ] Privacy budget (epsilon) reviewed if DP was used
- [ ] Results JSON archived to S3
- [ ] Temporary files cleaned up (`/tmp/fl-image.tar.gz`)
- [ ] Audit logs reviewed

---

## 12. Backup and Recovery

### 12.1 What to Back Up

| Asset | Location | Frequency | Destination |
|-------|----------|-----------|-------------|
| Training results | `~/fl-deploy/results/*.json` | After each run | S3 bucket |
| TLS certificates | `~/fl-deploy/certs/` | On rotation | Secrets Manager |
| Data manifests | `~/fl-deploy/data/*/manifest.json` | On change | S3 bucket |
| Orchestrator config | `cluster.env`, `run_server_side.sh` | On change | Version control |
| Docker image | `healthcare-fl:latest` | On rebuild | ECR |

### 12.2 Automated Backup

```bash
source cluster.env

# Back up results to S3
aws s3 sync ~/fl-deploy/results/ s3://${BACKUP_BUCKET}/results/$(date +%Y%m%d)/ \
  --sse aws:kms --sse-kms-key-id ${KMS_KEY_ID}

# Back up manifests
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "find ~/fl-deploy/data -name 'manifest.json' -exec cat {} \;" \
    > manifests/client_${ip}.json
done
aws s3 cp manifests/ s3://${BACKUP_BUCKET}/manifests/$(date +%Y%m%d)/ --recursive
```

### 12.3 Recovery Procedures

#### Scenario: Server Instance Lost

1. Launch new server instance (same type, same VPC/subnet)
2. Install GPU driver and Docker (Section 4.3-4.4)
3. Restore TLS certs from Secrets Manager
4. Pull Docker image from ECR
5. Regenerate server TLS cert with new private IP (Section 7.2)
6. Distribute new `ca.pem` if CA was rotated
7. Update `cluster.env` with new server IP
8. Resume training from last checkpoint

#### Scenario: Client Instance Lost

1. Launch replacement instance
2. Install GPU driver and Docker
3. Copy CA cert to new node
4. Re-ingest data from source (data stays at each site)
5. Pull Docker image from ECR
6. Update `cluster.env` with new client IP
7. Client will join on next orchestrator cycle

#### Scenario: Corrupted Training Results

1. Check S3 backup for latest valid results
2. Restore: `aws s3 sync s3://${BACKUP_BUCKET}/results/<date>/ ~/fl-deploy/results/`
3. Re-run affected tasks if backup is stale

---

## 13. Capacity Planning

### 13.1 GPU Memory by Task

| Task | Model | GPU Memory (per client) | Batch Size |
|------|-------|------------------------|------------|
| fraud, drug, readmission | MLP/LogReg | < 1 GB | 256 |
| sepsis, ecg | BiLSTM | ~2 GB | 128 |
| anomaly | Autoencoder | ~2 GB | 128 |
| mortality | TabNet | ~3 GB | 64 |
| satellite | ResNet-small | ~6 GB | 32 |
| chest, transfer | DenseNet-121 | ~12 GB | 16 |

L4 GPU (24 GB) is sufficient for all current models.

### 13.2 Scaling Clients

| Clients | Impact on Training Time | Network Overhead |
|---------|------------------------|------------------|
| 2-5 | Baseline | Negligible |
| 5-10 | ~same per round, more aggregation | Low |
| 10-20 | Aggregation becomes bottleneck | Moderate |
| 20+ | Requires async aggregation or hierarchical FL | High |

When scaling beyond 10 clients:
- Consider asynchronous aggregation strategies
- Use hierarchical FL with regional aggregators
- Monitor server CPU during aggregation (32 vCPU handles ~20 clients)

### 13.3 Storage Planning

| Component | Size | Growth Rate |
|-----------|------|-------------|
| Docker image | ~3.2 GB | Per rebuild |
| Results per task | ~1-5 MB | Per run |
| Full run results | ~50 MB | Per full run |
| Chest X-ray data | ~43 GB | Static |
| EBS root volume | 500 GB (server) / 1 TB (client) | Monitor monthly |

Set up disk usage alerts at 85% (Section 10.6).

---

## 14. Version Management and Rollback

### 14.1 Image Versioning

Tag images with semantic versions, not just `latest`:

```bash
# Build with version tag
VERSION=$(date +%Y%m%d)-$(git rev-parse --short HEAD)
docker build -t ${FL_IMAGE}:${VERSION} -f Dockerfile .
docker tag ${FL_IMAGE}:${VERSION} ${FL_IMAGE}:latest

# Push both tags to registry
docker push ${REGISTRY}/${FL_IMAGE}:${VERSION}
docker push ${REGISTRY}/${FL_IMAGE}:latest
```

### 14.2 Rollback Procedure

```bash
source cluster.env
ROLLBACK_VERSION=<previous-version-tag>

# Pull previous version on all nodes
for ip in ${SERVER_PUBLIC_IP} ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "docker pull ${REGISTRY}/${FL_IMAGE}:${ROLLBACK_VERSION} && \
     docker tag ${REGISTRY}/${FL_IMAGE}:${ROLLBACK_VERSION} ${FL_IMAGE}:latest"
done

# Restart orchestrator to pick up the rolled-back image
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker rm -f fl-orchestrator && docker rm -f fl-training"
# Re-launch orchestrator (Section 8.1)
```

### 14.3 Configuration Rollback

Keep `cluster.env` versions in a private repository. To roll back configuration:

```bash
git log --oneline cluster.env
git checkout <commit> -- cluster.env
```

---

## 15. Incident Response Runbook

### 15.1 Severity Levels

| Level | Definition | Response Time | Example |
|-------|-----------|---------------|---------|
| **P1** | Training halted, data at risk | 15 min | TLS compromise, unauthorized access |
| **P2** | Training degraded or failing | 1 hour | Client crash, GPU OOM, port conflict |
| **P3** | Non-blocking issue | 4 hours | Slow training, disk warning |

### 15.2 P1: Security Incident

1. **Isolate:** Remove affected instances from security group
2. **Preserve:** Snapshot EBS volumes for forensics
3. **Rotate:** Regenerate all TLS certificates immediately
4. **Rotate:** Create new SSH key pair, update all nodes
5. **Audit:** Review CloudTrail and Docker audit logs
6. **Notify:** Inform data owners per incident response policy
7. **Remediate:** Patch vulnerability, redeploy

### 15.3 P2: Training Failure

#### Server container exits unexpectedly

```bash
source cluster.env

# Check exit code and logs
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker inspect fl-training --format='{{.State.ExitCode}}'"
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker logs fl-training --tail 50"

# Common causes:
# Exit 137 -> OOM killed. Increase --memory or reduce model size.
# Exit 1   -> Python exception. Check logs for traceback.
# Exit 0   -> Normal completion. Check if all strategies ran.
```

#### Client can't connect

```bash
# Verify server is listening
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "ss -tlnp | grep 9092"

# Verify TLS cert SANs include server private IP
openssl x509 -in certs/server.pem -noout -text | grep -A1 "Subject Alternative Name"

# Verify network path
ssh -i ${KEY_PATH} ec2-user@<CLIENT_IP> "nc -zv ${SERVER_IP} 9092"
```

#### GPU errors

```bash
# CUDA device-side assert: DP noise corrupted model weights
# All models include prediction clamping (.clamp(1e-7, 1-1e-7)) to prevent this.
# If it recurs, check that the latest image is deployed.

# CUDA OOM: reduce batch size or MAX_SAMPLES
# L4 has 24GB -- sufficient for all current models

# nvidia-smi fails after long runs
sudo reboot  # Wait 60s
nvidia-smi --query-gpu=name --format=csv,noheader
```

### 15.4 P3: Performance Degradation

```bash
# Identify straggler clients
source cluster.env
for ip in ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "sudo docker logs fl-client 2>&1 | grep 'Round' | tail -1"
done

# Check for thermal throttling
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader"
done
```

---

## 16. Cost Management

### 16.1 Running Costs

| Configuration | Hourly | Daily (24h) | Monthly (730h) |
|--------------|--------|-------------|----------------|
| Full GPU cluster (1 server + 5 clients) | ~$9.25 | ~$222 | ~$6,753 |
| Stopped instances (EBS only) | ~$0.21 | ~$5 | ~$150 |

**EC2 charges whether training or not. Always stop instances when idle.**

### 16.2 Instance Lifecycle

```bash
source cluster.env

# Stop all instances
aws ec2 describe-instances \
  --filters "Name=tag:Environment,Values=production" "Name=tag:Project,Values=fl-platform" \
  --query 'Reservations[].Instances[].InstanceId' --output text | \
  xargs aws ec2 stop-instances --instance-ids

# Start all instances
aws ec2 describe-instances \
  --filters "Name=tag:Environment,Values=production" "Name=tag:Project,Values=fl-platform" \
  --query 'Reservations[].Instances[].InstanceId' --output text | \
  xargs aws ec2 start-instances --instance-ids

# Note: Public IPs change on restart unless Elastic IPs are assigned.
# Update cluster.env after restart.
```

### 16.3 Cost Optimization

- **Spot Instances** for clients: ~70% savings, acceptable for fault-tolerant FL (clients can rejoin)
- **Reserved Instances** for predictable workloads: ~40% savings
- **Schedule heavy tasks** (DenseNet) during off-peak hours
- **Right-size instances:** use `t3.xlarge` clients for small models (MLP, LogReg)

### 16.4 Time Estimates

| Scope | Tasks | Est. Time | Est. Cost (full GPU) |
|-------|-------|-----------|---------------------|
| Smoke test | fraud | 5 min | < $1 |
| Light tasks | fraud + sepsis + ecg | 3 hours | ~$28 |
| All except chest | 12 tasks | 5 hours | ~$46 |
| Full run | All 13 tasks | 20 hours | ~$185 |
| Chest X-ray only | chest | 14 hours | ~$130 |

---

## 17. Troubleshooting

### 17.1 Common Issues

| Issue | Symptom | Resolution |
|-------|---------|------------|
| **Port in use** | `Port 0.0.0.0:9092 is already in use` | A SuperLink or previous `fl-training` is still running. Run `docker rm -f fl-superlink fl-training` before starting. The orchestrator does this automatically. |
| **Client can't connect** | `Connection refused` or TLS errors | Check server is running on 9092, certs match, private IP correct in SAN |
| **CUDA device-side assert** | `CUDA error: device-side assert triggered` | DP noise corrupts model weights, causing `BCELoss(log(0))`. All models clamp predictions `.clamp(1e-7, 1-1e-7)`. Verify latest image is deployed. |
| **Server hangs** | Server stuck waiting for clients | Clients hit CUDA errors and exited. Orchestrator enforces per-task timeouts. |
| **GPU not found** | `nvidia-smi: not found` | Install driver: `sudo dnf install -y nvidia-driver` then `sudo modprobe nvidia` |
| **GPU not in Docker** | `CUDA not available` inside container | Install nvidia-container-toolkit, restart Docker (Section 4.4) |
| **Partition KeyError** | `KeyError: 0` in client logs | Extreme non-IID can leave partitions empty. Client code falls back to nearest partition. |
| **Docker needs sudo** | `permission denied` on client | Add user to docker group: `sudo usermod -aG docker ec2-user` |
| **Image not found** | `Unable to find image` | Pull from ECR or distribute from server (Section 5.2-5.3) |
| **TLS handshake fail** | `WRONG_VERSION_NUMBER` | Ensure server cert SAN includes the private IP. Regenerate certs (Section 7.2). |
| **CUDA OOM** | `RuntimeError: CUDA out of memory` | Reduce batch size or `MAX_SAMPLES`. L4 has 24GB -- sufficient for all current models. |

### 17.2 Client Reconnection

`run_client.py` has built-in reconnection logic:
- After each strategy completes, the client waits 2s then reconnects
- On connection failure, retries every 5s
- After 12 consecutive failures (~60s), assumes server is done and exits

To restart a failed client manually:

```bash
source cluster.env

ssh -i ${KEY_PATH} ec2-user@<CLIENT_IP> "
  sudo docker rm -f fl-client
  sudo docker run -d --name fl-client --network host --gpus all \
    --restart on-failure:3 \
    --memory 56g --cpus 14 \
    --log-opt max-size=100m --log-opt max-file=5 \
    -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro \
    -v ~/fl-deploy/data:/data:ro \
    -e PARTITION_ID=<N> -e NUM_CLIENTS=${NUM_CLIENTS} -e FL_TASK=<TASK> \
    -e FL_SERVER=${SERVER_IP}:9092 -e CERTS_DIR=/certs \
    ${FL_IMAGE}:${FL_IMAGE_TAG} python3 run_client.py
"
```

### 17.3 GPU Driver Recovery

```bash
# If nvidia-smi fails after long Docker runs
sudo reboot
# Wait 60s, then verify:
nvidia-smi --query-gpu=name --format=csv,noheader

# If module won't load after reboot
sudo modprobe nvidia
dmesg | tail -20  # check for errors
```

### 17.4 Rebuild and Redeploy

After code changes, rebuild and distribute:

```bash
source cluster.env

# Build on server
cd ~/fl-build
docker build -t ${FL_IMAGE}:${FL_IMAGE_TAG} .

# Distribute to all clients via VPC internal network
docker save ${FL_IMAGE}:${FL_IMAGE_TAG} | gzip > /tmp/fl-image.tar.gz
for ip in ${CLIENT_IPS}; do
  (scp -i ~/.ssh/deploy.pem /tmp/fl-image.tar.gz ec2-user@${ip}:/tmp/ && \
   ssh -i ~/.ssh/deploy.pem ec2-user@${ip} "sudo docker load < /tmp/fl-image.tar.gz && rm /tmp/fl-image.tar.gz") &
done
wait
rm /tmp/fl-image.tar.gz
```

### 17.5 Cleanup

```bash
source cluster.env

# Stop all FL containers across cluster
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker rm -f fl-superlink fl-training fl-orchestrator"
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "sudo docker rm -f fl-supernode fl-client"
done

# Prune old images (reclaim disk)
for ip in ${SERVER_PUBLIC_IP} ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "docker image prune -f"
done
```

---

## Appendix A: Privacy-Enhancing Technologies

For comprehensive PET coverage including DP variants, formal guarantees, SecAgg, TEE platforms, and decision matrices, see **[PET_Reference.md](PET_Reference.md)**.

## Appendix B: File Reference

| File | Purpose |
|------|---------|
| **Core** | |
| `run_ec2.py` | Server-side experiment runner. `--distributed` enables `start_server()` |
| `run_client.py` | Client-side runner with mTLS, pre-flight data check, reconnect loop |
| `ingest.py` | Client-side data ingestion CLI (validation, manifest, checksums) |
| `Dockerfile` | Unified Docker image (all deps pinned to exact versions) |
| **FL Framework** | |
| `fl_common/strategies.py` | All FL strategy implementations (FedAvg, SCAFFOLD, DP, SecAgg, etc.) |
| `fl_common/federated_adapter.py` | Generic federated LoRA framework for any HuggingFace model |
| `fl_common/data.py` | DataConfig, DataManifest, validation gates, partition utilities |
| `fl_common/dp.py` | Differential privacy primitives + RDP accountant |
| `fl_common/secagg.py` | Secure aggregation (pairwise masking) |
| **Models** | |
| `models/olmo/` | OLMo-1B federated LoRA (uses federated_adapter.py) |
| `models/bilstm/` | BiLSTM (sepsis, ECG). CPU fallback for DP strategies |
| `models/mlp/`, `models/densenet/`, etc. | Task-specific models with NaN sanitization |
| `models/*/server_app.py` | Strategy factory per model type |
| `models/*/client_app.py` | NumPyClient implementation per model type |
| **Tasks** | |
| `tasks/*/data.py` | Data pipeline per task (load, validate, clean, normalize, partition) |
| `tasks/gov_doc/data.py` | Government document data (4 domains: healthcare, finance, urban, research) |
| **Security** | |
| `secure_inference/tenseal_inference.py` | CKKS homomorphic encryption inference (MLP, BiLSTM, DenseNet) |
| `privacy/test_privacy.py` | MIA, DLG, canary extraction attacks |
| **Deploy** | |
| `deploy/distributed/deploy.sh` | Main deploy script (build, distribute, health, run, down) |
| `deploy/cluster.env.template` | Cluster configuration template |
| `deploy/validate_config.sh` | Configuration validation |
| `deploy/health_check.sh` | 38-point cluster health check (--quick, --json) |
| `deploy/gen_mtls_certs.sh` | mTLS certificate generation (CA + server + per-client) |
| `deploy/rotate_certs.sh` | Certificate rotation (check, generate, distribute, verify) |
| `deploy/backup.sh` | Backup results, certs, config, manifests (local + S3) |
| `deploy/rollback.sh` | Image version listing and rollback |
| `run_server_side.sh` | Server-side orchestrator (runs inside `docker:cli` container) |
| `scenarios/*.yaml` | Predefined experiment configurations |

## Appendix C: Adding a New Site (Client)

1. Launch a new instance in the same VPC (matching instance type)
2. Install GPU driver + Docker + nvidia-container-toolkit (Section 4.3-4.4)
3. Copy CA cert: `scp certs/ca.pem ec2-user@<NEW_IP>:~/fl-deploy/certs/`
4. Ingest local data: `python ingest.py --task <TASK> --input <DATA_PATH> --client-id <SITE_ID>`
5. Pull Docker image from ECR or load from tarball
6. Update `cluster.env`: add IP to `CLIENT_IPS`, increment `NUM_CLIENTS`
7. Update `PARTITION_ID` range (0 to N-1)

## Appendix D: Adding a New Task

1. Create model: `models/<name>/server_app.py` + `client_app.py`
2. Create data pipeline: `tasks/<name>/data.py`
3. Add to `run_ec2.py`: new `run_<name>()` function + entry in `task_map`
4. Add to `run_client.py`: new case in `make_client()`
5. Rebuild Docker image and distribute (Section 17.4)
6. Add scenario YAML in `scenarios/`
7. Run single-task validation: orchestrator target `<name>`
