# Tutorial 12: Operations & Production

**Time:** 30 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 8](08-distributed-deployment.md)

## What You'll Learn

- Monitor a production FL deployment
- Handle operational tasks (cert management, onboarding, incidents)
- Understand governance requirements
- Track costs

## Step 1: Health Monitoring

```bash
# Check container status
docker compose -f deploy/microservices/docker-compose.yml ps

# Follow coordinator logs
docker compose -f deploy/microservices/docker-compose.yml logs -f coordinator

# Check GPU status on a node
nvidia-smi
```

For multi-node deployments, check each node via SSH:
```bash
ssh ec2-user@<node_ip> "docker ps && nvidia-smi"
```

## Step 2: Certificate Management

Generate mTLS (mutual Transport Layer Security) certificates using OpenSSL:

```bash
# Generate CA
openssl req -x509 -newkey rsa:4096 -keyout ca.key -out ca.pem -days 365 -nodes

# Generate server cert
openssl req -newkey rsa:4096 -keyout server.key -out server.csr -nodes
openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -CAcreateserial -out server.pem -days 365

# Verify
openssl x509 -in server.pem -noout -subject -dates
openssl verify -CAfile ca.pem server.pem
```

See `deploy/runbooks/certificate_rotation.md` for the full rotation procedure.

## Step 3: Adding a New Participant

When a new organisation joins the federation:

1. **Governance** — sign federation agreement (see `deploy/templates/federation_agreement.md`)
2. **Certificate** — issue client certificate from the CA
3. **Network** — ensure gRPC port 9092 is accessible
4. **Deploy** — start client container:
   ```bash
   docker run -d --name fl-client-new \
     healthcare-fl:v1.0.0 \
     python3 runners/run_client.py --server <coordinator>:9092 --partition-id 2
   ```
5. **Validate** — run smoke test to confirm participation

See `deploy/runbooks/new_client_onboarding.md` for the full checklist.

## Step 4: Incident Response

When something goes wrong:

```bash
# 1. Check container logs
docker logs fl-coordinator --tail 50
docker logs fl-client-0 --tail 50

# 2. Check system resources
docker stats --no-stream

# 3. Common fixes
docker compose -f deploy/microservices/docker-compose.yml restart
docker compose -f deploy/microservices/docker-compose.yml down
```

### Severity Levels

| Severity | Response Time | Examples |
|----------|--------------|---------|
| P0 | < 15 min | mTLS regression, data breach |
| P1 | < 1 hour | Coordinator down, DP budget exhausted |
| P2 | < 4 hours | Cert near expiry, SecAgg abort rate high |
| P3 | Next business day | Non-blocking warnings |

See `deploy/runbooks/incident_response.md` for full procedures.

## Step 5: Backup and DR

**Backup scope:**

| Asset | Method | Retention |
|-------|--------|-----------|
| Infrastructure config | Git | Indefinite |
| Certificates | Encrypted store | Duration of federation |
| Model checkpoints | S3 | Per governance policy |
| Run records | S3 | 7 years minimum |
| Audit logs | CloudWatch / S3 | 7 years minimum |

See `deploy/runbooks/disaster_recovery.md` for DR drill procedures.

## Step 6: Governance Checkpoints

Before production, ensure all checkpoints are cleared:

- [ ] Use-case approval
- [ ] DPIA (Data Protection Impact Assessment)
- [ ] Security review
- [ ] Participant agreements signed (see `deploy/templates/federation_agreement.md`)
- [ ] Model governance (model card, privacy testing)
- [ ] Operational readiness (runbooks, on-call, monitoring)
- [ ] Production release approval

## Step 7: Cost Tracking

Key cost drivers:
- Coordinator compute (EC2 instance hours)
- Client compute (per participant)
- Storage (S3 for models, run records, audit logs)
- Key management (KMS)
- Logging (CloudWatch)

Tag all AWS resources with `Project`, `Environment`, and `Component` for cost allocation.
See the [Terraform tutorial](09-terraform.md) for infrastructure provisioning.

## Congratulations

You've completed all 12 tutorials. You now know how to:
- Run FL locally and in the cloud
- Apply DP, SecAgg, and privacy testing
- Choose the right FL strategy for your data
- Deploy and operate a production FL cluster
- Use VFL, split learning, and federated LLM fine-tuning
