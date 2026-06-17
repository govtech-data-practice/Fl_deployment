# Troubleshooting Guide

## Universal Diagnostic Steps

Before investigating specific symptoms, collect a diagnostic bundle:

```bash
./scripts/diagnose.sh --run-id $RUN_ID --env $ENVIRONMENT --since 2h
```

This collects system info, Docker logs, certificate status, and configuration into a single archive.

## Common Issues

### 1. Clients Cannot Connect to Coordinator

**Symptoms:** Timeout on client startup, "Connection refused" errors.

**Diagnosis:**
```bash
# Check coordinator is running
./deploy/health_check.sh --quick

# Check port accessibility
nc -zv <coordinator_ip> 9092

# Check security groups
aws ec2 describe-security-groups --group-ids <sg-id>
```

**Resolution:**
- Verify `FL_GRPC_PORT` matches in coordinator and client configs
- Check security group allows inbound on port 9092 from client IPs
- Verify VPC peering / PrivateLink if clients are in different VPCs
- Check coordinator container is healthy: `docker ps`

### 2. mTLS Handshake Failures

**Symptoms:** "SSL: CERTIFICATE_VERIFY_FAILED" or "handshake failure" errors.

**Diagnosis:**
```bash
# Verify certificate chain
openssl verify -CAfile certs/ca.pem certs/server.pem
openssl verify -CAfile certs/ca.pem certs/client.pem

# Check certificate expiry
openssl x509 -in certs/server.pem -noout -dates
```

**Resolution:**
- Ensure CA certificate is distributed to all nodes
- Check certificate has not expired (see `runbooks/certificate_rotation.md`)
- Verify CN/SAN matches the coordinator hostname
- Regenerate if needed: `./deploy/gen_mtls_certs.sh`

### 3. Training Diverges or Accuracy Drops

**Symptoms:** Loss increases, accuracy near random, NaN values in metrics.

**Diagnosis:**
- Check if DP is too aggressive: `python tools/dp_budget.py --all --rounds <N>`
- Check for data quality issues: `python tools/validate_manifest.py <manifest.json>`
- Check for non-IID severity (label skew, quantity skew)

**Resolution:**
- Switch to `DP_MODERATE` or `DP_RELAXED` if epsilon is very low
- Use SCAFFOLD or FedProx for non-IID data
- Increase number of local epochs
- Check data validation: `python tools/ingest.py --task <task> --validate-only`

### 4. SecAgg Abort Rate High (>20%)

**Symptoms:** P2 alarm, frequent round restarts.

**Diagnosis:**
- Check client dropout rate
- Check network stability between clients and coordinator
- Review `secagg/config.yaml` min_quorum setting

**Resolution:**
- Increase `round_timeout` if clients are slow
- Enable `dropout_tolerant: true` in SecAgg config
- Reduce `min_quorum` if acceptable for security model
- Investigate network issues on failing clients

### 5. DP Budget Exhausted

**Symptoms:** Training halts automatically, P1 alarm.

**Diagnosis:**
```bash
python tools/dp_budget.py --preset <current_preset> --rounds <rounds_completed>
```

**Resolution:**
- This is expected behaviour (privacy protection)
- Review whether more rounds are needed
- Consider using `DP_MODERATE` for future runs if more rounds are required
- Do NOT bypass the budget check

### 6. Docker Image Not Found on Client

**Symptoms:** Client startup fails with "image not found".

**Resolution:**
```bash
# Distribute image to all clients
./deploy/distributed/deploy.sh distribute

# Or manually on each client
docker load < healthcare-fl-v1.0.0.tar
```

### 7. GPU Not Available

**Symptoms:** Training runs on CPU, very slow.

**Diagnosis:**
```bash
# On the affected node
nvidia-smi
docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Resolution:**
- Install NVIDIA driver: `nvidia-smi` should show GPU
- Install nvidia-container-toolkit for Docker GPU access
- Ensure `--gpus all` flag in Docker run command
- Check that the GPU instance type supports CUDA 12.4+

### 8. Validation Failures

**Symptoms:** "Data validation FAILED" errors before training starts.

**Diagnosis:**
```bash
python tools/validate_manifest.py ~/fl-deploy/data/<task>/manifest.json --task <task>
python tools/ingest.py --task <task> --validate-only
```

**Resolution:**
- Check feature dimension matches task requirements
- Ensure minimum sample count (>= 10)
- Check for excessive NaN values (> 50%)
- Re-ingest data: `python tools/ingest.py --task <task> --input <data>`

## Getting Help

1. Run diagnostics: `./scripts/diagnose.sh --run-id <id> --env <env> --since 4h`
2. Check runbooks: `runbooks/`
3. Review health check: `./deploy/health_check.sh --json`
