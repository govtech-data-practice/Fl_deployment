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
./deploy/gen_mtls_certs.sh --client <participant_id>
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
./deploy/distributed/deploy.sh setup-client <new_client_ip>
```

Participant-side setup:
```bash
# Ingest and validate data
python ingest.py --task <task> --input <data_path> --client-id <participant_id>
python validate_manifest.py ~/fl-deploy/data/<task>/manifest.json --task <task>
```

### 5. Smoke Test

```bash
# Run smoke test including new participant
python run_ec2.py fraud --synthetic

# Verify in health check
./deploy/health_check.sh
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
