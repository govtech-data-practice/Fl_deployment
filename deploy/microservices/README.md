# Microservices Deployment

Docker Compose configurations for FL microservices architecture.

## Configurations

| File | Architecture | Services |
|------|-------------|----------|
| `docker-compose.yml` | Basic HFL | Coordinator + 2 clients |
| `docker-compose.hfl.yml` | Full HFL | Coordinator + clients + DP Accountant + Model Registry + Prometheus + Grafana |
| `docker-compose.vfl.yml` | Full VFL | Coordinator + 3 parties + PSA + DP Service + Audit Logger + Prometheus + Grafana |

## Quick Start

```bash
# Build the image first (from repo root)
docker build -t healthcare-fl:latest -f deploy/docker/Dockerfile.gpu .

cd deploy/microservices

# Basic HFL (3 containers)
docker compose up -d

# Full HFL microservices (7 containers)
docker compose -f docker-compose.hfl.yml up -d

# Full VFL microservices (9 containers)
docker compose -f docker-compose.vfl.yml up -d

# Custom task and clients
FL_TASK=sepsis NUM_CLIENTS=3 docker compose -f docker-compose.hfl.yml up -d
```

## Service Endpoints

### HFL (`docker-compose.hfl.yml`)

| Service | URL | Purpose |
|---------|-----|---------|
| Coordinator | `localhost:9092` | Flower gRPC |
| DP Accountant | `localhost:8081/budget` | Privacy budget tracking |
| Model Registry | `localhost:8083/models` | Versioned model storage |
| Prometheus | `localhost:9090` | Metrics |
| Grafana | `localhost:3000` | Dashboards (admin/admin) |

### VFL (`docker-compose.vfl.yml`)

| Service | URL | Purpose |
|---------|-----|---------|
| Coordinator | `localhost:9092` | Flower gRPC |
| PSA Service | `localhost:8084/status` | Entity alignment |
| DP Service | `localhost:8085/budget` | Privacy budget |
| Audit Logger | `localhost:8086/logs` | Immutable audit trail |
| Prometheus | `localhost:9090` | Metrics |
| Grafana | `localhost:3000` | Dashboards (admin/admin) |

## PSA Service API

```bash
# Register party records for alignment
curl -X POST localhost:8084/register \
  -H "Content-Type: application/json" \
  -d '{"party_id": "hospital_a", "records": ["id_001", "id_002"]}'

# Run exact alignment (shared IDs)
curl -X POST localhost:8084/align_exact

# Run fuzzy alignment (no shared IDs, CLK matching)
curl -X POST localhost:8084/align_fuzzy \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.7}'
```

## Audit Logger API

```bash
# Log an inter-party exchange
curl -X POST localhost:8086/log \
  -H "Content-Type: application/json" \
  -d '{"source": "party-0", "destination": "coordinator", "action": "send_embedding", "round": 1}'

# View all audit entries
curl localhost:8086/logs

# Count entries
curl localhost:8086/count
```

## Multi-Node Deployment

For deploying across separate EC2 instances, use the distributed compose files in `deploy/distributed/`.

See [`deploy/ARCHITECTURE.md`](../ARCHITECTURE.md) for the full microservices design.
