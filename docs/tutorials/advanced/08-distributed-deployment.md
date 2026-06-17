# Tutorial 8: Distributed Deployment

**Time:** 45 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 7](../intermediate/07-privacy-attacks.md), AWS account

## What You'll Learn

- Deploy FL to real EC2 instances (1 coordinator + N clients)
- Configure mTLS for secure communication
- Run distributed training across multiple nodes
- Monitor and troubleshoot a live cluster

## Step 1: Configure Your Cluster

```bash
cp deploy/cluster.env.template cluster.env
```

Edit `cluster.env` with your values:
```bash
FL_SERVER_HOST=54.x.x.x          # Coordinator public IP
FL_SERVER_PRIVATE=172.31.x.x     # Coordinator private IP
FL_CLIENT_HOSTS="10.0.1.10 10.0.1.11"  # Client IPs
FL_NUM_CLIENTS=2
FL_SSH_KEY=~/.ssh/fl_cluster.pem
```

Validate the configuration:
```bash
./deploy/validate_config.sh
```

## Step 2: Generate mTLS Certificates

```bash
./deploy/gen_mtls_certs.sh
```

This creates a CA and issues certificates for the coordinator and all clients. Certificates are stored in `certs/`.

Verify:
```bash
openssl x509 -in certs/server.pem -noout -subject -dates
```

## Step 3: Build and Deploy

```bash
# Build the Docker image
docker build -t healthcare-fl:v1.0.0 .

# Deploy to all nodes (distributes image, certs, and config)
./deploy/distributed/deploy.sh up
```

This:
1. Copies the Docker image to all nodes
2. Distributes TLS certificates
3. Starts the Flower SuperLink on the coordinator
4. Starts SuperNodes on each client

## Step 4: Pre-flight Check

```bash
./scripts/preflight.sh
```

Verify all checks pass:
- SSH connectivity to all nodes
- Docker running on all nodes
- GPU available (if using GPU instances)
- Certificates valid and distributed
- gRPC port accessible

## Step 5: Run Distributed Training

```bash
# Run fraud detection across the real cluster
python runners/run_ec2.py fraud
```

In distributed mode:
- The coordinator manages the SuperLink (aggregation server)
- Each client runs a SuperNode (local training)
- Communication is encrypted with mTLS
- Model updates flow over gRPC on port 9092

## Step 6: Monitor the Cluster

```bash
# Full health check
./deploy/health_check.sh

# JSON output for monitoring systems
./deploy/health_check.sh --json

# Quick check (coordinator only)
./deploy/health_check.sh --quick
```

## Step 7: Collect Diagnostics

If something goes wrong:

```bash
./scripts/diagnose.sh --run-id fraud-001 --env production --since 2h
```

This creates a `diag-*.tar.gz` bundle with system info, logs, cert status, and configuration.

## Step 8: Tear Down

```bash
./deploy/distributed/deploy.sh down
```

## Checkpoint

After completing this tutorial, you should have:
- [ ] A running FL cluster with mTLS
- [ ] Successfully completed a distributed training run
- [ ] Verified cluster health with `health_check.sh`
- [ ] Collected a diagnostic bundle

## Next Steps

- [Tutorial 9: Infrastructure with Terraform](09-terraform.md) — automate provisioning
