# FL Platform — Microservices Architecture

Containerised microservices design for both Horizontal FL (HFL) and Vertical FL (VFL), built on Flower.

---

## System Overview

```
+=====================================================================+
|                        REST GATEWAY :8080                            |
|  POST /experiments — launch HFL or VFL training                     |
|  GET  /experiments/{id} — status, metrics, privacy budget           |
|  GET  /models — versioned model registry                            |
|  GET  /health — aggregate service health                            |
+========+========================+========================+==========+
         |                        |                        |
   +-----v------+          +-----v------+          +------v------+
   |  HFL PLANE |          |  VFL PLANE |          |   SHARED    |
   |            |          |            |          |  SERVICES   |
   | SuperLink  |          | Server A   |          |             |
   | SuperNodes |          | Parties    |          | CA / TLS    |
   | DP Acct    |          | PSA        |          | Registry    |
   | SecAgg     |          | PSA        |          | Data Pipe   |
   | Model Reg  |          | DP Svc     |          | Monitoring  |
   +------------+          +------------+          +-------------+
```

---

## 1. HFL Microservices

### Service Inventory

| Service | Container | Ports | Role |
|---------|-----------|-------|------|
| **SuperLink** | `fl-superlink` | 9091, 9092, 9093 | Flower coordinator (Fleet API + Control) |
| **SuperNode** x N | `fl-supernode-{i}` | 7070 | Flower client, local data, local training |
| **DP Accountant** | `fl-dp-accountant` | 8081 | Tracks cumulative epsilon per round |
| **SecAgg Orchestrator** | `fl-secagg` | 8082 | SecAgg+ key exchange + mask coordination |
| **Model Registry** | `fl-model-registry` | 8083 | Versioned global models with checksums |
| **Health Monitor** | `fl-monitor` | 9100 | Prometheus metrics for all services |

### Communication Flow

```
                      +------------------+
                      |   Model Registry |
                      |   :8083 (REST)   |
                      +--------+---------+
                               |  save/load models
                               |
+----------+  gRPC :9092  +----v-------+  gRPC :8081  +--------------+
| SuperNode|<============>| SuperLink  |<============>| DP Accountant|
| (client) |    mTLS      | (coord)    |    mTLS      | (budget)     |
+----+-----+              +-----+------+              +--------------+
     |                          |
     |  gRPC :8082              |  Prometheus :9100
     +=========================>+========================>
     |  SecAgg key exchange     |                  +------+------+
     |                          |                  |   Grafana   |
+----v-----+                    |                  |   :3000     |
| SecAgg   |                    |                  +-------------+
| Orch     |                    |
+----------+              +-----v------+
                          | Health Mon |
                          | :9100      |
                          +------------+
```

### How it works

1. **SuperLink** starts on port 9092, waits for SuperNode connections
2. **SuperNodes** connect via gRPC with mTLS, register their data partition
3. Each round: SuperLink sends global model → SuperNodes train locally → send updates back
4. **DP Accountant** receives `(sigma, sample_rate, steps)` after each round, returns cumulative epsilon. Halts training if budget exceeded. Wraps `fl_pets/dp.py`
5. **SecAgg Orchestrator** manages pairwise key exchange before aggregation. Wraps `fl_pets/secagg.py`. Handles client dropout
6. **Model Registry** stores each round's global model (checksum, metrics, round number). S3-backed in production

### Strategies

FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi, DP-Central, DP-Local, SecAgg+, OneOwner — all from `fl_common/strategies.py`

---

## 2. VFL Microservices

### Service Inventory

| Service | Container | Ports | Role |
|---------|-----------|-------|------|
| **Coordinator** | `vfl-coordinator` | 9092 | Orchestrates forward/backward, holds top model + labels |
| **Party** x N | `vfl-party-{i}` | 7070 | Each party holds a feature subset, runs bottom model |
| **PSA Service** | `vfl-psa` | 8084 | Entity alignment before training (anonlink CLK) |
| **DP Service** | `vfl-dp` | 8085 | Noise on embeddings (forward) and gradients (backward) |
| **Audit Logger** | `vfl-audit` | 8086 | Immutable log of all inter-party exchanges |

### Communication Flow

```
                    +-------------+
                    | PSA Service |   (pre-training only)
                    |   :8084     |
                    +------+------+
                           | aligned entity indices
                           v
+----------+  embeddings  +-------------+
| Party 0  |=============>|             |
| (passive)|   mTLS       | Coordinator |
+----------+              |   :9092     |
                          |             |    +------------+
+----------+  embeddings  |  top model  |--->| DP Service |
| Party 1  |=============>|  + labels   |    |   :8085    |
| (passive)|   mTLS       |             |    +------------+
+----------+              |             |
                          |             |    +-------------+
+----------+  embeddings  |             |--->| Audit Logger|
| Party 2  |=============>|             |    |   :8086     |
| (passive)|   mTLS       +------+------+    +-------------+
+----------+                     |
                          DP-noised gradients
                          sent back to parties
```

### How it works

**Pre-training (PSA phase):**
1. Each party submits CLK-encoded quasi-identifiers (name, DOB, address) to the PSA Service
2. PSA Service computes fuzzy alignment using anonlink Dice similarity matching
3. Double PSA (identity + location triangulation) for high precision
4. Output: shared set of aligned row indices sent to all parties
5. Parties slice their data to the aligned subset before training begins

**Training (forward pass):**
1. Each passive party runs its **bottom model** on its feature subset
2. Sends **embeddings** (not raw features) to the Coordinator via mTLS
3. Coordinator concatenates embeddings, runs the **top model**, computes loss

**Training (backward pass):**
1. Coordinator computes gradients w.r.t. each party's embedding
2. DP Service applies calibrated noise to gradients (sigma from DP preset)
3. DP-noised gradients sent back to each party
4. Each party updates its bottom model locally
5. Audit Logger records the exchange (timestamp, party, payload hash, round)

**Privacy guarantees:**
- Parties send **embeddings**, not raw features — Coordinator cannot reconstruct party data
- Gradients are **DP-noised** before returning to parties — limits information leakage
- Raw data **never leaves** the party that holds it

### VFL Models

- `vfl_mlp` — 3-party vertical MLP, 10 features per party, fraud detection
- `split_bilstm` — split BiLSTM, LSTM encoder private, classifier shared, sepsis prediction

---

## 3. Shared Services

| Service | Container | Ports | Used by |
|---------|-----------|-------|---------|
| **REST Gateway** | `fl-gateway` | 8080 | Admin API for both HFL and VFL |
| **Service Registry** | Docker DNS / Consul | — | All services |
| **Certificate Authority** | `fl-ca` | 8200 | Issues/rotates mTLS certs |
| **Data Pipeline** | `fl-data-pipeline` | 8087 | Ingestion, validation, partitioning |
| **Prometheus** | `fl-prometheus` | 9090 | Metrics collection |
| **Grafana** | `fl-grafana` | 3000 | Dashboards |

### Certificate Authority

```
Root CA (offline, generated once)
  |
  +-- Server cert (SuperLink / Server A)
  +-- Client certs x N (SuperNodes / Parties)
  +-- Service certs (DP, SecAgg, PSA, Gateway, etc.)
```

Existing: `deploy/terraform/main.tf` provisions TLS via `tls_private_key` + `tls_locally_signed_cert`, stored in SSM Parameter Store. Rotation: `deploy/runbooks/certificate_rotation.md`.

### Data Pipeline

Wraps existing tools:
- `tools/ingest.py` — CSV/NPZ ingestion with schema validation
- `tools/validate_manifest.py` — checksum + label distribution verification
- HFL: horizontal partitioning (split by rows)
- VFL: vertical partitioning (split by columns) + PSA alignment

---

## 4. Port Map

| Port | Protocol | Service |
|------|----------|---------|
| 3000 | HTTP | Grafana dashboards |
| 7070 | gRPC | SuperNode / Party client |
| 8080 | REST | Gateway (admin API) |
| 8081 | gRPC | DP Accountant |
| 8082 | gRPC | SecAgg Orchestrator |
| 8083 | REST | Model Registry |
| 8084 | gRPC | PSA Service |
| 8085 | gRPC | DP Service (VFL) |
| 8086 | REST | Audit Logger |
| 8087 | REST | Data Pipeline |
| 8200 | REST | Certificate Authority |
| 9090 | HTTP | Prometheus |
| 9091 | HTTP | SuperLink (HTTP) |
| 9092 | gRPC | SuperLink / Server A (Fleet API) |
| 9093 | gRPC | SuperLink (Control) |
| 9100 | HTTP | Health Monitor (Prometheus exporter) |

---

## 5. Deployment Topologies

### Local Development (single machine)

```
docker compose up                    # HFL mode (default)
FL_MODE=vfl docker compose up        # VFL mode
```

All services in one Docker Compose on a bridge network. Docker DNS for discovery. Self-signed certs from `certs/`.

```
+------------------------------------------------------+
|  fl-network (bridge)                                  |
|                                                       |
|  +-------------+  +----------+  +----------+          |
|  | superlink   |  | node-0   |  | node-1   |   HFL   |
|  | :9091-9093  |  | :7070    |  | :7070    |          |
|  +-------------+  +----------+  +----------+          |
|                                                       |
|  +----------+  +----------+  +----------+             |
|  | dp-acct  |  | model-reg|  | gateway  |  Shared    |
|  | :8081    |  | :8083    |  | :8080    |             |
|  +----------+  +----------+  +----------+             |
+------------------------------------------------------+
```

### Multi-Node (EC2, Docker Compose per instance)

```
  EC2: Coordinator (public subnet)    EC2: Client N (private subnet)
  +-----------------------------+     +-----------------------------+
  | superlink  :9091-9093       |     | supernode  :7070            |
  | dp-acct    :8081            |     |   SUPERLINK_IP=10.0.1.x    |
  | model-reg  :8083            |     |   PARTITION_ID=N            |
  | gateway    :8080            |     +-----------------------------+
  | prometheus :9090            |
  +-----------------------------+
```

### Production (Terraform)

```
  VPC 10.0.0.0/16
  +-----------------------------------------------------------+
  |  Public Subnet             Private Subnet(s)              |
  |  10.0.1.0/24               10.0.10.0/24                   |
  |                                                           |
  |  +----------------+        +----------------+             |
  |  | SuperLink EC2  |        | SuperNode EC2  | x N         |
  |  | t3.large       |        | g4dn.xlarge    | (GPU)       |
  |  | SG: 9091-9093  |        | SG: 7070 in    |             |
  |  +----------------+        +----------------+             |
  |                                                           |
  |                                                           |
  |  ECR: healthcare-fl:latest                                |
  |  SSM: /healthcare-fl/tls/{ca,cert,key}                    |
  |  S3: training data (read-only from clients)               |
  +-----------------------------------------------------------+
```

---

## 6. Security Boundaries

```
+---------------------------+     +---------------------------+
|    TRUST DOMAIN 1         |     |    TRUST DOMAIN: SHARED   |
|    (Coordinator)          |     |                           |
|                           |     |  CA (offline root)        |
|  SuperLink / Coordinator  |     |  Prometheus (read-only)   |
|  Model Registry           |     |  Audit Logger (append)    |
|  DP Accountant            |     |                           |
|  Gateway                  |     |                           |
+---------------------------+     +---------------------------+
             |
    mTLS     |
             v
+---------------------------+
|    TRUST DOMAIN 2..N      |
|    (Data Parties)         |
|                           |
|  SuperNode / Party        |
|  Local data never leaves  |
+---------------------------+
```

### Key constraints

- Raw patient data **never leaves** the party that holds it
- All inter-service traffic is **mTLS-encrypted**
- DP defaults to **fail-closed** (`DP_STRONG`, sigma=1.5)
- Audit logger is **append-only** with signed entries
- Certificate rotation every **90 days** (see `deploy/runbooks/certificate_rotation.md`)

---

## 7. Startup Order

### HFL

```
1. CA (generate certs)
2. Prometheus, Grafana
3. DP Accountant, Model Registry
4. SuperLink (healthcheck: TCP 9092)
5. SuperNodes (depend on SuperLink healthy)
6. Gateway
```

### VFL

```
1. CA (generate certs)
2. Prometheus, Grafana
3. PSA Service → run entity alignment (must complete before training)
4. DP Service, Audit Logger
5. Coordinator (healthcheck: TCP 9092)
6. Party services (depend on Coordinator healthy)
7. Gateway
```
