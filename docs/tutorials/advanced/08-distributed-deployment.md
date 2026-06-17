# Tutorial 8: Distributed Deployment

**Time:** 45 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 7](../intermediate/07-privacy-attacks.md), Docker installed

## What You'll Learn

- Deploy FL as microservices (Docker Compose)
- Run distributed training across multiple containers
- Monitor and troubleshoot a live cluster

## Step 1: Local Microservices (Single Machine)

The quickest way to test distributed FL:

```bash
cd deploy/microservices

# Start coordinator + 2 clients (fraud detection)
docker compose up

# Or a different task
FL_TASK=sepsis docker compose up
```

This starts 3 containers on a shared Docker network:
- `fl-coordinator` — runs `runners/run_ec2.py --distributed` on port 9092
- `fl-client-0` — runs `runners/run_client.py --server coordinator:9092`
- `fl-client-1` — runs `runners/run_client.py --server coordinator:9092`

## Step 2: Build the Docker Image

```bash
# From repo root
docker build -t healthcare-fl:v1.0.0 .

# Verify
docker run --rm healthcare-fl:v1.0.0 python3 -c "import flwr; print('Flower', flwr.__version__)"
```

## Step 3: Multi-Node Deployment (EC2)

For real distributed deployment across separate EC2 instances:

**On the coordinator node:**
```bash
docker compose -f deploy/distributed/docker-compose.superlink.yml up -d
```

**On each client node:**
```bash
FL_SERVER=<coordinator_private_ip>:9092 \
docker compose -f deploy/distributed/docker-compose.supernode.yml up -d
```

## Step 4: Run Distributed Training

```bash
# Run from the coordinator
python3 runners/run_ec2.py fraud --distributed
```

In distributed mode:
- The coordinator manages aggregation (FedAvg, SCAFFOLD, etc.)
- Each client trains locally and sends encrypted model updates
- Communication flows over gRPC on port 9092

## Step 5: Monitor

```bash
# Check container status
docker compose -f deploy/microservices/docker-compose.yml ps

# Follow coordinator logs
docker compose -f deploy/microservices/docker-compose.yml logs -f coordinator

# Follow a client
docker compose -f deploy/microservices/docker-compose.yml logs -f client_0
```

## Step 6: Tear Down

```bash
docker compose -f deploy/microservices/docker-compose.yml down
```

## Checkpoint

After completing this tutorial, you should have:
- [ ] Built the Docker image
- [ ] Run microservices deployment locally
- [ ] Completed a distributed training run
- [ ] Monitored coordinator and client logs

## Next Steps

- [Tutorial 9: Infrastructure with Terraform](09-terraform.md) — automate AWS provisioning
