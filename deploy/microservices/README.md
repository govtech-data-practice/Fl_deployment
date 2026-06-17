# Microservices Deployment

Run the FL platform as containerised microservices using Docker Compose.

## Architecture

```
  docker-compose.yml
  ├── coordinator (fl-coordinator)    port 9092
  │     └── runners/run_ec2.py --distributed
  ├── client_0 (fl-client-0)
  │     └── runners/run_client.py --server coordinator:9092
  └── client_1 (fl-client-1)
        └── runners/run_client.py --server coordinator:9092
```

## Quick Start

```bash
cd deploy/microservices

# HFL: Fraud detection (default)
docker compose up

# VFL: Vertical fraud
FL_TASK=vfl_fraud docker compose up

# FTL: Transfer learning chest X-ray
FL_TASK=transfer docker compose up

# Custom rounds and clients
NUM_ROUNDS=10 NUM_CLIENTS=2 docker compose up
```

## Multi-Node Deployment

For deploying across separate EC2 instances, use the distributed Docker Compose files:

```bash
# On coordinator node:
docker compose -f docker-compose.superlink.yml up -d

# On each client node:
FL_SERVER=<coordinator_private_ip>:9092 docker compose -f docker-compose.supernode.yml up -d
```

See `deploy/distributed/` for the multi-node compose files.
