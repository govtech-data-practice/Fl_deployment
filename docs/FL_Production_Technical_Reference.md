# Federated Learning Production Technical Reference

**Last updated:** 2026-05-26
**Scope:** Production technical reference for deploying FL+PET systems in government environments.

This document covers the architecture, security model, failure modes, and operational considerations for production federated learning. It is based on a reference implementation validated on AWS EC2 (g6.8xlarge server, 5x g6.4xlarge clients) and documents what works, what the limitations are, and what each deployment decision trades off. Every claim is tagged with its actual implementation status.

---

## 1. Architecture Decisions

### Why Flower

**Status: Implemented, tested.**

Flower (flwr >= 1.13) is the FL framework. It was chosen for:
- Python-native, integrates directly with PyTorch
- Strategy pattern (FedAvg, FedProx, FedAdam, FedYogi) available out of the box
- Simulation mode for local development, SuperLink/SuperNode for distributed

**Limitations encountered:**

1. **Deprecated `start_server()` API.** The codebase uses `start_server()` for distributed mode (`run_ec2.py`, line 199). This API is deprecated in favor of `ServerApp`/`ClientApp`. The deprecated API still works but receives no new features and may be removed. The simulation path already uses the new `ServerApp` API (line 207).

2. **No built-in round timeout enforcement.** `ServerConfig(round_timeout=120)` is passed (line 200) but Flower's `start_server()` does not reliably enforce it. In practice, a stuck client causes the server to hang indefinitely. The workaround is the orchestrator-level task timeout in `deploy/distributed/deploy.sh` (`FL_TIMEOUT_DEFAULT`, `FL_ROUND_TIMEOUT` in `cluster.env.template`, lines 62-64).

3. **No client authentication at the Flower layer.** Flower identifies clients by auto-assigned node IDs, not by verified identity. Any process that can reach port 9092 and present the CA cert can join as a client.

4. **SecAgg is not Flower's built-in SecAgg.** The implementation in `fl_common/secagg.py` is a custom pairwise-mask scheme (deterministic PRG masks that cancel on aggregation). It is not the Flower SecAgg plugin. It requires all clients to participate (line 92-93 of `fl_common/strategies.py`: returns `None` on any failure).

### Why Docker (not Kubernetes)

**Status: Implemented, tested.**

Docker containers are used for all deployment (`Dockerfile`, `deploy/distributed/deploy.sh`). Kubernetes was not used because:
- Cluster is 6 nodes (1 server + 5 clients). K8s overhead is not justified.
- SSH-based orchestration is simpler to debug on a small cluster.
- No auto-scaling requirement (cross-silo FL has fixed participants).

**When to migrate to K8s:**
- More than ~10 client nodes
- Need for GPU node pools with heterogeneous hardware
- Need for automatic container restart with health probes
- Need for service mesh (Istio) for mTLS at the network layer
- Multi-tenant deployments

Docker security flags applied (`deploy/distributed/deploy.sh`, line 85):
```
--security-opt=no-new-privileges --cap-drop ALL --pids-limit 4096
```

**Known issue:** `--read-only` flag was tested and removed. PyTorch writes to `/tmp` for CUDA compilation cache and model loading. The workaround is `--tmpfs /tmp:rw,noexec,nosuid` (line 288).

### Why SSH Orchestration

**Status: Implemented, tested.**

`deploy/distributed/deploy.sh` uses SSH (`ssh -i $FL_SSH_KEY`) to orchestrate all nodes. This means:
- Server has SSH access to all client nodes
- Image distribution is `docker save | gzip | scp | docker load`
- Certificate distribution is `scp`
- No service discovery; client IPs are hardcoded in `cluster.env`

**When to migrate:**
- **ECS/Fargate:** When you need managed container orchestration without K8s complexity
- **Ansible:** When you need idempotent configuration management across >10 nodes
- **Systems Manager (SSM):** When you need agentless command execution without SSH keys

### Single Server SPOF

**Status: Documented, not solved.**

The FL server (SuperLink) is a single point of failure. If the server EC2 instance goes down:
- All training stops
- No automatic failover
- Model state is lost unless checkpointed

The `deploy/rollback.sh` and `deploy/backup.sh` scripts exist for manual recovery but there is no HA setup. Flower does not support multi-server or leader election.

### No Service Discovery

**Status: Documented, not solved.**

Client IPs are hardcoded in `cluster.env` (line 19: `FL_CLIENT_HOSTS=""`). Adding or removing a client requires:
1. Edit `cluster.env`
2. Re-run `deploy.sh distribute`
3. Re-run `deploy.sh restart`

There is no DNS-based discovery, no consul, no ECS service discovery.

---

## 2. Security Model

### Threat Model

**What is protected:**
- Raw data never leaves client nodes. Each client loads data locally (`models/mlp/client_app.py` line 86: `_load()` reads from local partitions). In distributed mode, data is mounted read-only into the container (`-v ~/fl-deploy/data:/data:ro`, `deploy/distributed/deploy.sh` line 325).

**What is NOT protected:**
- **Model updates leak information.** This is proven in this codebase:
  - `privacy/test_privacy.py` lines 72-168: DLG (Deep Leakage from Gradients) attack recovers training input features from plain gradients with cosine similarity > 0.5.
  - `privacy/test_privacy.py` lines 175-291: Membership Inference Attack achieves advantage > 0.1 (better than random) without DP.
  - `privacy/attack_suite.py`: Full LLM attack battery (MIA, verbatim memorization, canary extraction, attribute inference) on fine-tuned Mistral 7B.

- **The server sees all model updates in plaintext** (unless SecAgg is used). In standard FedAvg, the server receives raw weight deltas from each client. A malicious server can run gradient inversion attacks.

- **SecAgg mitigates but current implementation has limitations.** `fl_common/secagg.py` uses deterministic pairwise masks (SHA-256 seed, NumPy PRNG). It requires all N clients to participate (mask cancellation only works with equal 1/N weighting, per comment on line 86-91 of `fl_common/strategies.py`). If any client drops, the round fails. There is no dropout tolerance, no Shamir secret sharing, no threshold decryption.

### TLS

**Status: Implemented, tested. Not mTLS despite task #22 being marked complete.**

Certificate generation: `deploy/gen_certs.sh` and inline in `deploy/distributed/deploy.sh` lines 234-252.

What exists:
- Self-signed CA (ECDSA P-256)
- Server certificate with SAN entries for private IP, public IP, localhost
- CA cert distributed to all clients
- SuperLink started with `--ssl-certfile`, `--ssl-keyfile`, `--ssl-ca-certfile`
- SuperNodes connect with `--root-certificates /certs/ca.pem`
- Certificate rotation script: `deploy/rotate_certs.sh` with expiry checking, backup, and TLS handshake verification

What does NOT exist:
- **Client certificates.** `deploy/gen_certs.sh` generates only server cert + CA. There is no `gen_mtls_certs.sh` in the codebase. Clients present only the CA cert for server verification, not their own client certs. Flower's `start_server()` passes `certificates` as a 3-tuple `(ca, cert, key)` but this is for the server's own TLS, not for verifying client identity.
- **Certificate revocation.** Self-signed CA with no CRL or OCSP.
- **HSM integration.** Private keys are files on disk (`~/fl-deploy/certs/server.key`, `chmod 600`).

### Differential Privacy

**Status: Implemented, tested. Known failure modes documented.**

Implementation: `fl_common/dp.py`

**What works:**
- `clip_update()`: L2 norm clipping of model updates (line 23-30)
- `add_gaussian_noise()`: Calibrated Gaussian noise N(0, (sigma * C)^2) (line 33-49)
- `clip_and_noise()`: Full client-side DP pipeline (line 52-69)
- `PrivacyAccountant`: RDP-based accounting with RDP-to-(epsilon,delta)-DP conversion (line 76-133)
- Central DP: `FedDPAvg` strategy clips each client update on the server, adds noise to aggregate (line 113-202 of `fl_common/strategies.py`)
- Local DP: Clients clip+noise their own updates before sending (line 133-135 of `models/mlp/client_app.py`)

**What breaks:**
- **DP on DenseNet (8M params): AUC drops to ~0.50** (random). The noise required to protect 8M parameters at any reasonable epsilon destroys the signal. DenseNet strategies in `run_ec2.py` lines 100-108 explicitly skip DP strategies.
- **DP noise on LSTM weights can cause NaN.** Fixed with `nan_to_num` sanitization in weight loading. The LSTM hidden state amplifies noise through recurrent connections.
- **Epsilon tracking is per-run only.** `PrivacyAccountant` resets on each experiment (`self.steps = 0` on init, line 88). There is no persistent privacy budget across experiments. Running the same task 10 times with epsilon=10 each time gives total epsilon >> 10, but this is not tracked.

---

## 3. Data Sovereignty

### Data Pipeline

**Status: Implemented, tested.**

`ingest.py` provides a CLI for local data ingestion:
- Supports CSV, NPZ, and image directory formats
- Validates schema, checks for NaN/Inf, verifies class balance
- Generates `manifest.json` with SHA-256 checksum
- Supports synthetic data generation for testing

`fl_common/data.py` provides:
- `DataConfig`: typed configuration per task (line 40-100)
- `DataManifest`: metadata about client datasets (line 105-143)
- `validate_tabular()`: blocks training on invalid data (line 168-225)
- `load_dataset()`: loads from standardized format (line 263-298)
- `partition_local()`: client-side train/val/test split (line 451-478)

### Known Gaps

- **`local_mode` vs `simulation`:** `DataConfig` has a `synthetic` flag (line 56) and `ingest.py` supports `--synthetic`. However, in distributed runs (`deploy/distributed/deploy.sh` line 397), data is mounted from the host filesystem. The `ingest.py` pipeline has never been used in a real distributed deployment; data was manually placed on EC2 instances.

- **No data versioning.** `manifest.json` is overwritten on re-ingest (`generate_manifest()` at line 385-446 of `fl_common/data.py`). There is no version history, no content-addressable storage, no DVC integration.

- **No data lineage.** There is no record of which model checkpoint was trained on which version of which data. The `results/` JSON files record task name and strategy but not data checksums or manifest versions.

- **Manifest checksum is file-level.** `compute_checksum()` (line 376-382 of `fl_common/data.py`) hashes the entire file. There is no per-record provenance.

---

## 4. Network Design

**Status: Implemented. Minimal security segmentation.**

All nodes are in the same AWS VPC and security group. The network topology is:

```
[Server: g6.8xlarge]  <--- gRPC (9092) --->  [Client 0: g6.4xlarge]
                      <--- gRPC (9092) --->  [Client 1: g6.4xlarge]
                      <--- gRPC (9092) --->  [Client 2: g6.4xlarge]
                      <--- gRPC (9092) --->  [Client 3: g6.4xlarge]
                      <--- gRPC (9092) --->  [Client 4: g6.4xlarge]
```

**Issues:**

1. **No network segmentation between server and clients.** All nodes share the same security group. A compromised client can reach any other client directly, not just the server.

2. **Port 9092 is open within VPC.** `cluster.env.template` line 38: `FL_GRPC_PORT=9092`. There are no per-client firewall rules. Any host in the VPC can connect to port 9092.

3. **No egress controls.** Containers use `--network host` (line 281 of `deploy/distributed/deploy.sh`). There is no egress filtering. A compromised container can reach the internet, other AWS services, or the EC2 metadata endpoint (169.254.169.254).

4. **SSH from server to all clients.** `deploy/distributed/deploy.sh` line 190: server SCPs files to clients using the same SSH key. If the server is compromised, the attacker has SSH access to every client.

---

## 5. Failure Modes (All Encountered During Testing)

| Failure | Root Cause | Fix | File |
|---------|-----------|-----|------|
| Client CUDA assert -> server hangs | NaN in model weights after GPU error | `nan_to_num` sanitization on weight load | `models/*/client_app.py` |
| Client synthetic fallback crash | Partition ID not in data loaders | Fallback to first available partition with warning | `models/mlp/client_app.py` line 46 |
| Local network drop -> entire run fails | No client reconnection in `start_server()` | Server-side orchestrator with per-task timeout | `deploy/distributed/deploy.sh`, `cluster.env` timeouts |
| Port 9092 conflict -> silent failure | Previous SuperLink not stopped | `docker rm -f` before starting new container | `deploy/distributed/deploy.sh` line 277 |
| `start_server()` round_timeout ignored | Flower deprecated API limitation | Orchestrator-level `FL_TIMEOUT_DEFAULT` / `FL_ROUND_TIMEOUT` | `cluster.env.template` lines 62-64 |
| Docker `--read-only` breaks PyTorch | PyTorch needs writable `/tmp` for CUDA cache | Removed `--read-only`, added `--tmpfs /tmp:rw,noexec,nosuid` | `deploy/distributed/deploy.sh` line 288 |
| `EarlyStopWrapper.aggregate_fit` exception | Strategy error propagated to Flower server, killing the run | `try/except` in `aggregate_fit` returns `(None, {})` on error | `fl_common/strategies.py` lines 353-358 |
| SecAgg mask cancellation failure | Fewer clients than expected (dropout) | Return `None` and log error; no partial aggregation | `fl_common/strategies.py` lines 92-93 |

---

## 6. Monitoring Gaps

**Status: All gaps are real. No monitoring infrastructure exists.**

| What | Status | Notes |
|------|--------|-------|
| Prometheus metrics | **Not implemented** | No metrics endpoint exposed |
| Grafana dashboards | **Not implemented** | No visualization of training progress |
| CloudWatch | **Documented in `cluster.env.template`** (`FL_LOG_DRIVER=awslogs`) | Not configured. `json-file` driver used in practice |
| Alerting | **Not implemented** | No SNS, no PagerDuty, no email alerts |
| Centralized logging | **Not implemented** | Docker `json-file` logs on each node. `deploy/distributed/deploy.sh` `cmd_logs()` (line 564) does SSH + `docker logs` |
| GPU utilization per round | **Not tracked** | `cmd_status()` runs `nvidia-smi` on-demand but not per-round |
| Training metrics dashboard | **Scaffolded, not tested** | `cmd_dashboard()` in `deploy/distributed/deploy.sh` (line 438) tries to start Streamlit. `dashboard.py` does not exist in the codebase |
| Per-round metric history | **Implemented** | `MetricCapture` class in `run_ec2.py` (line 156) records per-round metrics to JSON |
| Log rotation | **Implemented** | Docker log rotation: `--log-opt max-size=200m --log-opt max-file=5` (line 71 of `deploy/distributed/deploy.sh`) |

---

## 7. Secure Inference -- Honest Assessment

**Status: Implemented as benchmarks. Not production-integrated.**

Implementation: `secure_inference/tenseal_inference.py`

### TenSEAL CKKS (Microsoft SEAL backend)

| Model | Approach | Time/Sample | Accuracy vs Plaintext | Notes |
|-------|----------|-------------|----------------------|-------|
| MLP (30-dim input) | Full encrypted inference | ~0.56s | Mean abs error < 0.01 | Works. 3 encrypted linear layers + square activation. `EncryptedMLP` class, line 115. |
| BiLSTM (sepsis/ECG) | Hybrid: LSTM plaintext, classifier encrypted | ~0.69s | Classification match ~10/10 | Data owner runs LSTM locally (line 182-188), sends encrypted embeddings. Model owner classifies encrypted embeddings. |
| DenseNet-121 (chest X-ray) | Hybrid: feature extractor plaintext, classifier encrypted | ~5.6s | Mean abs error < 0.05 | Data owner runs full DenseNet feature extractor locally (line 397-403), sends encrypted 1024-dim features. |

### Honest problems

1. **"Hybrid" means the data owner has the feature extractor weights.** For BiLSTM, the data owner runs `model.lstm()` locally (line 186). For DenseNet, they run `model.base_model.features()` locally (line 399). The data owner needs partial model weights. This partially defeats the purpose if model confidentiality is a goal.

2. **Full encrypted inference on DenseNet is not feasible.** 8M parameters, convolutions, batch normalization -- the multiplicative depth required exceeds what CKKS can handle with practical parameters. The `poly_modulus_degree=16384` context (line 49) gives ~8 levels of multiplicative depth, enough for a 3-layer MLP but not for a 121-layer DenseNet.

3. **CKKS security parameters.** `create_context()` at line 39 uses `poly_modulus_degree=16384` (labeled "128bit") which is actually conservative. The `coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60]` gives 8 multiplicative levels. This is not independently audited for 128-bit security; TenSEAL/SEAL defaults are trusted.

4. **CrypTen (Meta): archived May 2025.** Not an option for new development.

5. **Other demos in `secure_inference/`:**
   - `demo_paillier.py`: Paillier encryption demo (additive HE only)
   - `demo_secret_sharing.py`: Shamir secret sharing demo
   - `demo_functional_encryption.py`: functional encryption demo
   - `demo_tee.py`: TEE simulation demo
   - These are educational demonstrations, not production implementations.

6. **Production alternatives (not implemented):** SecretFlow (Ant Group), CrypTFlow2 (Microsoft Research), or hardware TEE (Intel SGX/TDX, AWS Nitro Enclaves).

---

## 8. What Would Be Different in Real Production

| Reference Implementation | Scaled Production | Why |
|-------------|----------------|-----|
| SSH + Docker on EC2 | Kubernetes with GPU node pools (EKS) | Auto-scaling, health probes, rolling updates, RBAC |
| `start_server()` deprecated API | Flower `ServerApp`/`ClientApp` via `flwr run` | Active development, better lifecycle management |
| Public/private IPs in `cluster.env` | AWS PrivateLink for cross-account FL | No data traverses public internet, no IP management |
| SSH keys for instance access | IAM roles + SSM Session Manager | No key management, audit trail, temporary credentials |
| TLS key files on disk | AWS Secrets Manager or ACM Private CA | Key rotation, access control, audit logging |
| `docker save \| scp` image distribution | ECR + pull-through cache | Versioned images, vulnerability scanning, no SCP |
| `scp` for model distribution | EFS or S3 for shared model storage | Concurrent access, versioning, lifecycle policies |
| Docker `json-file` logs | CloudWatch Logs + metric filters + SNS alerts | Centralized, searchable, alerting |
| Manual `deploy.sh` | Terraform + CI/CD (GitHub Actions) | Infrastructure as code, reproducible, auditable |
| Self-signed CA, no revocation | ACM Private CA with CRL/OCSP | Certificate lifecycle management |
| `nvidia-smi` on-demand | CloudWatch GPU metrics + Prometheus + Grafana | Continuous monitoring, anomaly detection |
| No audit trail | CloudTrail + application-level audit log | Compliance requirement for HIPAA, 21 CFR Part 11 |

Terraform scaffolding exists at `deploy/terraform/` (main.tf, variables.tf, outputs.tf, userdata templates) but has never been applied.

---

## 9. Monolith vs Microservices for FL+PET

### Current Architecture: Monolith

The reference implementation uses a monolithic architecture: one Docker image (`healthcare-fl:latest`, ~6GB) contains everything — all 12 models, all 10 task pipelines, FL strategies, privacy attacks, secure inference, and the data pipeline. The same image runs on both server and client nodes. The entry point determines the role (`run_ec2.py` for server, `run_client.py` for client).

**Why monolith was chosen:**
- Simplest to deploy — one image, one `docker run` command
- No service discovery needed — server and client are the same binary
- No inter-service communication overhead — everything in-process
- Easy to test — `python run_ec2.py fraud` runs everything locally
- Small team — one developer, no need for service boundaries

**Where monolith breaks down:**

| Problem | Impact | When it matters |
|---------|--------|-----------------|
| 6GB image for every node | Wastes bandwidth and storage. Clients only need one model but get all 12 | >10 clients |
| All models loaded at import time | Memory overhead on small instances | Budget deployments (t3.xlarge) |
| Can't update one model without redeploying everything | Downtime for all tasks when fixing one model | Continuous deployment |
| No independent scaling | Can't add more clients for one task without affecting others | Multi-task parallel runs |
| Single failure domain | Bug in ingest.py crashes the training container | Production reliability |

### Microservices Architecture for Production FL+PET

For government production, FL+PET should decompose into independent services:

```
┌─────────────────────────────────────────────────────────┐
│                    Control Plane                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐            │
│  │ FL        │  │ Task     │  │ Cert       │            │
│  │ Coordinator│  │ Registry │  │ Manager    │            │
│  └──────────┘  └──────────┘  └────────────┘            │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐            │
│  │ Audit    │  │ Config   │  │ Monitoring │            │
│  │ Logger   │  │ Store    │  │ (Prometheus)│            │
│  └──────────┘  └──────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────┘
                         │ gRPC
┌────────────────────────┼────────────────────────────────┐
│                  Aggregation Plane                       │
│  ┌──────────────────────────────────────┐               │
│  │ Strategy Engine                      │               │
│  │ (FedAvg, SCAFFOLD, SecAgg, DP, etc.) │               │
│  └──────────────────────────────────────┘               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐            │
│  │ SecAgg   │  │ DP       │  │ HE         │            │
│  │ Service  │  │ Accountant│  │ Aggregator │            │
│  └──────────┘  └──────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────┘
                         │ gRPC + mTLS
┌────────────────────────┼────────────────────────────────┐
│              Client Plane (per hospital/agency)          │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐            │
│  │ Data     │  │ Training │  │ Secure     │            │
│  │ Ingestor │  │ Worker   │  │ Inference  │            │
│  └──────────┘  └──────────┘  └────────────┘            │
│  ┌──────────┐  ┌──────────┐                             │
│  │ Model    │  │ Privacy  │                             │
│  │ Store    │  │ Guard    │                             │
│  └──────────┘  └──────────┘                             │
└─────────────────────────────────────────────────────────┘
```

### Service Decomposition

| Service | Image Size | Responsibility | Scales Independently |
|---------|-----------|----------------|---------------------|
| **FL Coordinator** | ~200MB | Task scheduling, client registry, status tracking | No (singleton) |
| **Strategy Engine** | ~500MB | Aggregation strategies (FedAvg, SCAFFOLD, etc.) | No (stateful) |
| **SecAgg Service** | ~300MB | Pairwise masking, key exchange, mask aggregation | Yes (per-round) |
| **DP Accountant** | ~100MB | Privacy budget tracking, noise calibration, RDP composition | No (singleton) |
| **HE Aggregator** | ~400MB | Paillier/CKKS encrypted aggregation via TenSEAL | Yes (per-round) |
| **Training Worker** | ~2GB | Model training (one model per image, not all 12) | Yes (per-client) |
| **Data Ingestor** | ~300MB | `ingest.py` as a service — validation, manifest, checksums | Yes (per-client) |
| **Secure Inference** | ~1GB | TenSEAL CKKS inference endpoint | Yes (per-request) |
| **Audit Logger** | ~100MB | Append-only JSONL, HMAC signing, provenance | No (singleton) |
| **Model Store** | ~200MB | Model versioning, checkpoint management, lineage | No (singleton) |
| **Privacy Guard** | ~200MB | MIA/DLG attack testing on model updates per round | Yes (async) |
| **Cert Manager** | ~100MB | Certificate lifecycle, rotation, revocation | No (singleton) |

### When to Use Which

| Factor | Monolith | Microservices |
|--------|----------|---------------|
| Team size | 1-3 engineers | 5+ engineers |
| Number of clients | 2-10 | 10+ |
| Number of concurrent tasks | 1 | Multiple |
| Deployment frequency | Weekly | Daily/continuous |
| Compliance requirements | Informal | Formal audit trail required |
| Budget | Limited | Sufficient for K8s/EKS |
| Time to first deployment | Days | Weeks |
| Operational complexity | Low (SSH + Docker) | High (K8s, service mesh, observability stack) |

### Migration Path (Monolith → Microservices)

Do not start with microservices. The recommended migration path:

**Phase 1 — Modular monolith (current + improvements)**
- Split the 6GB image into task-specific images (`healthcare-fl-fraud:latest`, `healthcare-fl-sepsis:latest`)
- Each image contains only the model and task it needs (~1-2GB instead of 6GB)
- Still deployed with `docker run`, no K8s yet
- **Effort:** 1-2 days. Just build separate Dockerfiles per task.

**Phase 2 — Extract stateless services**
- Pull out `ingest.py` as a standalone data validation service
- Pull out secure inference (`tenseal_inference.py`) as a standalone gRPC service
- Pull out audit logging (not yet implemented) as a sidecar
- Training worker and strategy engine remain coupled
- **Effort:** 1-2 weeks.

**Phase 3 — Full decomposition (requires K8s)**
- FL Coordinator as a separate service managing task lifecycle
- Strategy Engine with pluggable strategy backends
- SecAgg / DP / HE as independent privacy services
- Service mesh (Istio/Linkerd) for mTLS between services
- **Effort:** 1-2 months. Only justified at scale (>10 agencies, continuous operation).

### Anti-Patterns to Avoid

1. **Don't split too early.** A monolith that works is better than microservices that don't. Prove FL+PET works first; decomposition is an optimization.

2. **Don't make the aggregation server stateless.** FL aggregation is inherently stateful (round counter, model weights, client registry). Trying to make it stateless adds complexity with no benefit.

3. **Don't expose PET services externally.** SecAgg, DP, and HE services should only be reachable from the aggregation plane, never from clients directly.

4. **Don't share GPUs across services.** One GPU = one training worker. GPU sharing (MPS/MIG) adds failure modes that aren't worth the savings.

5. **Don't version services independently before you need to.** Coordinated releases are simpler until you have >3 teams deploying independently.

### When Monolith Is Sufficient

**Monolith** (one image, `docker run`, SSH orchestration) is appropriate for:
- Initial deployment and evaluation
- Small-scale production (2-10 agencies)
- Single-task or sequential multi-task training

**Migrate to microservices** when:
- >10 concurrent participants
- 24/7 continuous operation required
- Formal compliance mandates independent audit trail
- Multiple teams deploying independently
- Multi-task parallel training needed

---

## 10. Federated Learning for LLMs — Techniques and Decision Matrix

### 10.1 FlexOLMo: Pros, Cons, and Applicability

FlexOLMo (AI2, July 2025) uses Mixture-of-Experts (MoE) where each expert is trained independently on private data and integrated via nonparametric routing. This is architecturally different from our LoRA-based approach.

#### FlexOLMo Pros

| Advantage | Detail |
|-----------|--------|
| **No data sharing by design** | Each expert trains on its own data. No model updates cross boundaries during training. |
| **Flexible composition** | Experts can be included/excluded at inference time. Add a healthcare expert without retraining finance. |
| **Data attribution** | Can trace which expert (and therefore which agency's data) contributed to a prediction. 41% improvement in attribution vs dense models. |
| **No aggregation overhead** | No FedAvg rounds. Each expert trains independently — no synchronisation, no communication rounds. |
| **Specialisation** | Each expert becomes a domain specialist. Better than diluting all knowledge into one adapter. |

#### FlexOLMo Cons

| Disadvantage | Detail |
|--------------|--------|
| **Model size** | Smallest variant is 2×7B = 14B params. Doesn't fit on L4 (24GB) even with 4-bit quantisation. Needs A100 or multi-GPU. |
| **Inference cost** | MoE routes to multiple experts per token — higher latency and memory than single-model inference. |
| **No cross-agency learning** | Experts don't learn from each other's data. A healthcare expert learns nothing from finance data, even when it would help. |
| **Routing complexity** | Nonparametric routing must be trained or tuned. Misrouted tokens go to the wrong expert. |
| **Limited ecosystem** | Released July 2025, requires transformers≥4.57.0. Small community, few production deployments. |
| **No privacy guarantees** | Experts are shared at inference time — the model weights ARE the data representation. No DP, no SecAgg. |
| **Fixed expert size** | All experts are 7B. Can't mix a 1B expert with a 7B expert. |

#### When to Use FlexOLMo vs Federated LoRA

| Factor | FlexOLMo (MoE) | Federated LoRA (our approach) |
|--------|----------------|-------------------------------|
| Agency trust level | Low — agencies never share anything | Medium — encrypted adapter weights are shared |
| Cross-agency learning | No — each expert is independent | Yes — FedAvg aggregates knowledge across agencies |
| Hardware requirement | A100 80GB+ (14B minimum) | L4 24GB (1B-7B with QLoRA) |
| Number of domains | 2-8 (one expert per domain) | Any (adapters are domain-agnostic) |
| Privacy guarantees | None (weights contain data patterns) | DP noise can be applied to adapter updates |
| Dynamic composition | Yes — add/remove experts at inference | No — single aggregated adapter |
| Training communication | Zero — fully independent | O(rounds × adapter_size) — 32-256 MB/round |
| Model attribution | Built-in (which expert fired) | Not available |
| Production readiness | Research (2025, limited tooling) | Flower + HuggingFace + PEFT (mature tooling) |

### 10.2 Decision Matrix: FL Techniques for LLMs

| Technique | How It Works | Params Transmitted | Privacy | Communication Cost | Accuracy | Hardware | Best For |
|-----------|-------------|-------------------|---------|-------------------|----------|----------|----------|
| **Full FedAvg** | Average all model weights each round | Entire model (7B = 28GB) | None (raw weights) | Prohibitive | Highest | Multi-A100 | Research only |
| **Federated LoRA** | Freeze base, federate only LoRA adapters | Adapter only (8-256MB) | DP applicable | Low (0.1-1% of model) | Good | L4 24GB | **Production default** |
| **Federated QLoRA** | Same as LoRA + 4-bit base quantisation | Same as LoRA | DP applicable | Low | Good (slight degradation) | L4 24GB | **GPU-constrained sites** |
| **FlexOLMo (MoE)** | Independent experts, MoE routing | Zero during training | None (weights = data) | Zero | Varies by routing | A100 80GB+ | Independent domains, no trust |
| **Split Learning** | Model split: bottom at client, top at server | Intermediate embeddings | Embedding leakage risk | Per-batch (latency) | Good | Any | Single agency + cloud |
| **Federated Prompt Tuning** | Freeze everything, federate soft prompts only | Prompt vectors (0.01% of model) | DP easy (tiny params) | Minimal (<1MB/round) | Lower than LoRA | Any GPU | Extreme bandwidth constraints |
| **Federated Head Tuning** | Freeze base, federate only classification head | Head weights (1-10MB) | DP easy | Very low | Lower than LoRA | Any GPU | Simple classification tasks |
| **Distillation-based FL** | Each client trains local model, server distils | Logits or soft labels | Good (no raw weights) | Medium | Good | Any | Heterogeneous client models |
| **FedIT (Instruction Tuning)** | Federated instruction tuning with LoRA | Adapter + instruction stats | DP applicable | Low | Good for instruction tasks | L4+ | Government Q&A systems |

### 10.3 Recommendation by Use Case

| Use Case | Recommended Technique | Why |
|----------|----------------------|-----|
| **Cross-hospital clinical NLP** (5-20 hospitals, similar data) | Federated QLoRA | Hospitals have similar schema, cross-learning helps, L4 GPUs available |
| **Cross-agency document classification** (3-5 agencies, different domains) | FlexOLMo or Federated LoRA | If domains are very different → FlexOLMo. If domains overlap → LoRA. |
| **Multi-bank fraud NLP** (strict compliance) | Federated LoRA + DP | Compliance requires formal privacy guarantee. DP on adapters is feasible. |
| **Government Q&A system** (instruction tuning) | FedIT with QLoRA | Instruction tuning benefits from cross-agency question diversity. |
| **Edge deployment** (mobile, IoT, low bandwidth) | Federated Prompt Tuning | <1MB per round. Works on any hardware. |
| **Sensitive intelligence** (zero trust between parties) | FlexOLMo or Distillation | No weights cross boundaries. Or distil to a shared student model. |
| **Research collaboration** (universities, A*STAR) | Full FedAvg or LoRA | Trust is high, hardware is good, maximise accuracy. |

### 10.4 What We Implemented and Tested

| Technique | Status | Results | Files |
|-----------|--------|---------|-------|
| **Federated QLoRA** | Implemented + tested on EC2 | OLMo-1B, 5 nodes, perplexity 1.13 in 96s | `fl_common/federated_adapter.py`, `models/olmo/` |
| **11 model presets** | Implemented (not all tested on EC2) | OLMo, Llama, Mistral, Phi, BERT, Whisper, ViT, etc. | `fl_common/federated_adapter.py` PRESETS dict |
| **FlexOLMo (actual model)** | Not implemented | Needs A100 + transformers≥4.57.0 | Planned for v1.0 |
| **Federated Prompt Tuning** | Not implemented | — | Planned |
| **Split Learning for LLM** | Not implemented | — | Planned |
| **Full FedAvg for LLM** | Not practical | 28GB per round for 7B model | Not planned |
| **Federated LoRA + DP** | Partially tested | DP noise on adapters causes some instability. Mistral tested in earlier session. | `models/mistral/` (legacy) |

### 10.5 Communication Cost Comparison (7B model, 5 clients, 10 rounds)

| Technique | Per-Round | Total (10 rounds × 5 clients) | Practical? |
|-----------|-----------|-------------------------------|------------|
| Full FedAvg | 28 GB | 1.4 TB | No |
| LoRA rank-8 | 224 MB | 11.2 GB | Yes |
| LoRA rank-4 | 112 MB | 5.6 GB | Yes |
| QLoRA (same adapter size) | 224 MB | 11.2 GB | Yes |
| Prompt Tuning (100 tokens) | 0.4 MB | 20 MB | Yes |
| FlexOLMo | 0 | 0 | Yes (but no cross-learning) |

---

## 11. Compliance Mapping

### PDPA (Singapore Personal Data Protection Act)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Data stays at each site | Implemented | Clients load data locally, server never sees raw data |
| Consent for data use | **Not implemented** | No consent management, no data subject access |
| Model updates may contain PII | **Proven risk** | MIA in `privacy/test_privacy.py` shows membership is detectable. DLG shows partial input reconstruction. DP mitigates but does not eliminate. |
| Data breach notification | **Not implemented** | No breach detection, no notification mechanism |

### HIPAA (US Health Insurance Portability and Accountability Act)

| Requirement | Status | Notes |
|-------------|--------|-------|
| BAA with each participant | **Not implemented** | Legal/contractual, not technical |
| Encrypted transport | Implemented | TLS on gRPC (port 9092) |
| Encrypted at rest | **Not implemented** | Data on EC2 EBS volumes, no EBS encryption configured |
| Access controls | Basic | SSH key-based access, Docker `--cap-drop ALL` |
| Audit logging | **Not implemented** | No application-level audit log. Docker logs only. |
| Minimum necessary | Partial | FL design limits data exposure, but model updates leak info |

### 21 CFR Part 11 (FDA Electronic Records)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Audit trail | **Not implemented** | No record of who accessed what, when |
| Electronic signatures | **Not implemented** | No model approval workflow |
| Model provenance | **Not implemented** | Results JSON records task+strategy but not data version, code version, or operator identity |
| System validation | **Not implemented** | No IQ/OQ/PQ documentation |

### NIST 800-53

| Control Family | Status | Notes |
|----------------|--------|-------|
| AC (Access Control) | Basic SSH | No RBAC, no MFA, single SSH key for cluster |
| AU (Audit) | **Not implemented** | No audit logging |
| SC (System & Communications) | TLS implemented | No network segmentation, no egress filtering |
| IA (Identification & Authentication) | SSH key only | No per-user identity, no certificate-based client auth |
| CM (Configuration Management) | `cluster.env` + `Dockerfile` | No configuration drift detection, no immutable infra |
| CP (Contingency Planning) | `backup.sh`, `rollback.sh` exist | Never tested in a real failure scenario |

---

## 12. Tested vs Claimed

| Feature | Status | Honest Notes |
|---------|--------|-------------|
| **FL Training** | | |
| FedAvg | Tested | Works across all 10 tasks. Core aggregation in `flwr.server.strategy.FedAvg`. |
| FedProx | Tested | Proximal term implemented client-side (`models/mlp/client_app.py` line 108-109). |
| FedAdam | Tested | Server-side adaptive optimizer. `fl_common/strategies.py` line 477-479. |
| FedYogi | Tested | Same as FedAdam but Yogi update rule. Line 483. |
| SCAFFOLD | Tested | Custom implementation in `fl_common/strategies.py` lines 22-61. Server control variate update is simplified (comment on line 53). |
| FedAdaptiveWarmup | Tested | Custom: FedAdam for first N rounds then FedAvg. Lines 209-289. |
| FedOneOwner | Tested | Access control strategy: only designated owner gets final model. Lines 296-327. |
| EarlyStopping | Tested | Wrapper in `fl_common/strategies.py` lines 334-378. |
| Non-IID partitioning (Dirichlet) | Tested | Label skew via alpha parameter. Used across all tasks. |
| **Models** | | |
| MLP (fraud, 30-dim) | Tested | `models/mlp/client_app.py`. 3-layer MLP, dropout 0.3. |
| BiLSTM (sepsis, 14-dim; ECG, 12-dim) | Tested | `models/bilstm/client_app.py`. 2-layer bidirectional LSTM + FC. |
| DenseNet-121 (chest X-ray) | Tested | `models/densenet/client_app.py`. Pretrained ImageNet + fine-tuned. 8M params. |
| VFL MLP (vertical FL) | Tested | `models/vfl_mlp/`. 3 parties, 10 features each. Simulation only. |
| Split BiLSTM | Tested | `models/split_bilstm/`. LSTM private, classifier shared. Simulation only. |
| Autoencoder (anomaly) | Tested | `models/autoencoder/`. Reconstruction-based anomaly detection. |
| CNN1D | Tested | `models/cnn1d/`. 1D convolution for time series. |
| ResNet-small (satellite) | Tested | `models/resnet_small/`. Small ResNet for 64x64 images. |
| TabNet (mortality) | Tested | `models/tabnet_simple/`. Simplified TabNet. |
| LogReg (readmission) | Tested | `models/logreg/`. Logistic regression via PyTorch. |
| Generic MLP (drug) | Tested | `models/generic/`. Configurable via environment variables. |
| Mistral 7B QLoRA | Tested (separate) | `models/mistral/`. Requires A100 GPU. Not part of standard FL run. |
| **Tasks** | | |
| Fraud (synthetic) | Tested | 30-dim tabular, binary classification. |
| Sepsis (real eICU + synthetic) | Tested | 14-dim time series, 48 timesteps. Real data on EC2. |
| ECG (synthetic) | Tested | 12-dim time series, 250 timesteps. |
| Chest X-ray (real NIH + synthetic) | Tested | 224x224 images, 14 multi-label pathologies. Real data (43GB) on server only. |
| Anomaly detection (synthetic) | Tested | 40-dim tabular, autoencoder reconstruction. |
| Mortality (synthetic) | Tested | 25-dim tabular, binary. |
| Drug response (synthetic) | Tested | 200-dim molecular fingerprints, binary. |
| Satellite (synthetic) | Tested | 64x64x3 images, 5-class land use. |
| Readmission (synthetic) | Tested | 20-dim tabular, binary. |
| Gov LLM (clinical notes) | Tested (separate) | Mistral 7B QLoRA fine-tuning. Requires GPU. |
| **Privacy** | | |
| DP-Central (server-side clipping + noise) | Tested | `fl_common/strategies.py` `FedDPAvg`. Works for MLP/BiLSTM. Destroys DenseNet. |
| DP-Local (client-side clipping + noise) | Tested | `fl_common/dp.py` `clip_and_noise()`. Applied in client_app.py. |
| Privacy Accountant (RDP) | Tested | `fl_common/dp.py` `PrivacyAccountant`. Per-run only. |
| DLG attack | Tested | `privacy/test_privacy.py` `test_gradient_inversion()`. Demonstrates gradient leakage. |
| MIA attack (BiLSTM) | Tested | `privacy/test_privacy.py` `test_membership_inference()`. |
| MIA attack (MLP) | Tested | `privacy/test_privacy.py` `test_mia_mlp()`. |
| LLM attack suite (6 attacks) | Documented | `privacy/attack_suite.py`. Requires Mistral 7B + GPU. Not run in standard pipeline. |
| SecAgg+ (pairwise masks) | Tested | `fl_common/secagg.py`. Demo-grade. No dropout tolerance. |
| Persistent privacy budget | **Not implemented** | Accountant resets each run. |
| **Secure Inference** | | |
| TenSEAL CKKS (MLP) | Tested | Full encrypted inference. `secure_inference/tenseal_inference.py`. |
| TenSEAL CKKS (BiLSTM hybrid) | Tested | Encrypted classifier only. |
| TenSEAL CKKS (DenseNet hybrid) | Tested | Encrypted classifier head only. Data owner has feature extractor. |
| Paillier demo | Documented | `secure_inference/demo_paillier.py`. Educational only. |
| Secret sharing demo | Documented | `secure_inference/demo_secret_sharing.py`. Educational only. |
| TEE demo | Documented | `secure_inference/demo_tee.py`. Simulated, no real SGX/TDX. |
| Functional encryption demo | Documented | `secure_inference/demo_functional_encryption.py`. Educational only. |
| **Infrastructure** | | |
| Docker containerization | Tested | `Dockerfile`. Python 3.12, PyTorch 2.5.1+cu124, Flower >= 1.13. |
| TLS (server cert + CA) | Tested | `deploy/gen_certs.sh`, inline in `deploy/distributed/deploy.sh`. |
| mTLS (client certs) | **Not implemented** | No client certificate generation or verification. |
| Certificate rotation | Tested | `deploy/rotate_certs.sh`. Includes expiry check, backup, redistribution, TLS verification. |
| SSH orchestration | Tested | `deploy/distributed/deploy.sh`. Full lifecycle: build, distribute, run, status, logs, down. |
| Data ingestion CLI | Tested | `ingest.py`. CSV/NPZ/image support, validation, manifest generation. |
| Health check | Documented | `deploy/health_check.sh` exists (referenced but not inspected). |
| Backup/rollback | Documented | `deploy/backup.sh`, `deploy/rollback.sh` exist. Never tested in real failure. |
| Terraform | Scaffolded | `deploy/terraform/main.tf` etc. Never applied. |
| Streamlit dashboard | Scaffolded | `cmd_dashboard()` in deploy script. `dashboard.py` does not exist. |
| YAML scenarios | Tested | `run_ec2.py` `load_scenario()`. Supports single-task, multi-experiment, attack-only. |
| Distributed mode (SuperLink) | Tested | `deploy/distributed/deploy.sh` `cmd_up()`. Server + 5 clients on EC2. |
| Simulation mode | Tested | `run_ec2.py` default mode. `flwr.simulation.run_simulation()`. |
| CloudWatch integration | **Not implemented** | `cluster.env.template` has `FL_LOG_DRIVER=awslogs` but never configured. |
| S3 backup | **Not implemented** | `FL_BACKUP_S3_BUCKET` in template, never used. |
| Container resource limits | Tested | Memory, CPU, SHM limits in `cluster.env`. Applied in deploy script. |
| Docker security hardening | Tested | `--cap-drop ALL`, `--security-opt=no-new-privileges`, `--pids-limit`, `--tmpfs` with noexec. |

---

## Key File Paths

| File | Purpose |
|------|---------|
| `run_ec2.py` | Main entry point: simulation and distributed FL runs |
| `fl_common/strategies.py` | All FL strategies: FedAvg, FedProx, SCAFFOLD, SecAgg+, DP, AdaptiveWarmup, OneOwner |
| `fl_common/dp.py` | Differential privacy: clipping, noise, RDP accountant |
| `fl_common/secagg.py` | SecAgg+ pairwise mask scheme |
| `fl_common/data.py` | Data pipeline: config, manifest, validation, loading |
| `ingest.py` | Data ingestion CLI |
| `models/mlp/client_app.py` | MLP client with FedProx, SCAFFOLD, SecAgg, DP support |
| `privacy/test_privacy.py` | DLG + MIA attacks on BiLSTM and MLP |
| `privacy/attack_suite.py` | Full LLM attack battery (6 attacks) |
| `secure_inference/tenseal_inference.py` | CKKS encrypted inference benchmarks |
| `deploy/distributed/deploy.sh` | SSH-based cluster orchestration |
| `deploy/gen_certs.sh` | TLS certificate generation |
| `deploy/rotate_certs.sh` | Certificate rotation with backup and verification |
| `deploy/cluster.env.template` | Cluster configuration template |
| `deploy/terraform/` | Terraform scaffolding (not applied) |
| `Dockerfile` | Container image definition |
