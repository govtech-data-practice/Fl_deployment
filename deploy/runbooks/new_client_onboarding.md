# New Client Onboarding Runbook

## Overview

Adding a new participant to the federation requires governance approval, infrastructure provisioning, certificate issuance, and validation.

## Procedure

### 1. Governance Approval

- [ ] Use-case alignment confirmed
- [ ] Federation agreement signed (see `templates/federation_agreement.md`)
- [ ] Data protection impact assessment (DPIA) completed if required
- [ ] Security review passed

### 2. Certificate Issuance

```bash
# Generate client certificate for new participant
openssl req -x509 -newkey rsa:4096 -keyout ca.key -out ca.pem -days 365 -nodes
```

- Issue certificate from subordinate CA
- Set appropriate validity period
- Record in certificate register

### 3. Network Connectivity

- [ ] VPC peering or PrivateLink established
- [ ] Security group rules configured (allow gRPC port from new client IP)
- [ ] DNS records created (if using hostnames)

Verify connectivity:
```bash
# From new client, test connectivity to coordinator
nc -zv <coordinator_ip> 9092
```

### 4. Client Deployment

```bash
# Add new client IP to cluster.env
# FL_CLIENT_HOSTS="<existing_ips> <new_client_ip>"
# FL_NUM_CLIENTS=<updated_count>

# Deploy to new client
docker run -d healthcare-fl:v1.0.0 python3 runners/run_client.py --server <coordinator>:9092
```

Participant-side setup:
```bash
# Ingest and validate data
python tools/ingest.py --task <task> --input <data_path> --client-id <participant_id>
python tools/validate_manifest.py ~/fl-deploy/data/<task>/manifest.json --task <task>
```

### 5. Smoke Test

```bash
# Run smoke test including new participant
python runners/run_ec2.py fraud --synthetic

# Verify in health check
docker compose -f deploy/microservices/docker-compose.yml ps
```

Confirm:
- [ ] mTLS handshake succeeds
- [ ] Client participates in training rounds
- [ ] Aggregation completes with new client count

### 6. Production Onboarding

- [ ] Representative data loaded and validated
- [ ] Privacy budget allocated
- [ ] Participant added to monitoring dashboards
- [ ] On-call rotation updated
- [ ] Runbooks updated with new participant details
- [ ] Notification sent to all existing participants
