# Tutorial 12: Operations & Production

**Time:** 30 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 8](08-distributed-deployment.md)

## What You'll Learn

- Monitor a production FL deployment
- Handle common operational tasks (cert rotation, onboarding, incidents)
- Understand governance requirements
- Track costs

## Step 1: Health Monitoring

```bash
# Full cluster health check
./deploy/health_check.sh

# JSON output (for dashboards / automation)
./deploy/health_check.sh --json

# Quick check (coordinator only)
./deploy/health_check.sh --quick
```

The health check verifies: SSH connectivity, Docker status, GPU availability, disk space, Docker image presence, TLS certificate validity, and port status.

## Step 2: Certificate Rotation

Certificates expire. The platform alerts at 30, 14, and 7 days before expiry.

```bash
# Check current certificate expiry
openssl x509 -in certs/server.pem -noout -dates

# Rotate certificates
./deploy/rotate_certs.sh

# Verify new certs
./deploy/health_check.sh
```

See `runbooks/certificate_rotation.md` for the full procedure.

## Step 3: Adding a New Participant

When a new organisation joins the federation:

```bash
# 1. Generate their certificate
./deploy/gen_mtls_certs.sh --client new_participant

# 2. Add their IP to cluster.env
# FL_CLIENT_HOSTS="<existing> <new_ip>"

# 3. Deploy to the new node
./deploy/distributed/deploy.sh setup-client <new_ip>

# 4. Smoke test
python runners/run_ec2.py fraud --synthetic

# 5. Verify
./deploy/health_check.sh
```

See `runbooks/new_client_onboarding.md` for the full checklist.

## Step 4: Incident Response

When something goes wrong:

```bash
# 1. Collect diagnostics
./scripts/diagnose.sh --run-id <id> --env production --since 4h

# 2. Review the bundle
tar -xzf diag-*.tar.gz
cat diag-*/system_info.txt
cat diag-*/cert_status.txt
cat diag-*/docker_logs.txt

# 3. Common fixes
./deploy/distributed/deploy.sh restart   # Restart services
./deploy/rollback.sh                     # Rollback to previous version
```

### Severity Levels

| Severity | Response Time | Examples |
|----------|--------------|---------|
| P0 | < 15 min | mTLS regression, data breach |
| P1 | < 1 hour | Coordinator down, DP budget exhausted |
| P2 | < 4 hours | Cert near expiry, SecAgg abort rate high |
| P3 | Next business day | Non-blocking warnings |

See `runbooks/incident_response.md` for full procedures.

## Step 5: Backup and DR

```bash
# Manual backup
./deploy/backup.sh

# Verify backup integrity
./deploy/backup.sh --verify
```

**Backup scope:**

| Asset | Method | Retention |
|-------|--------|-----------|
| Infrastructure config | Git | Indefinite |
| Certificates | Encrypted store | Duration of federation |
| Model checkpoints | S3 | Per governance policy |
| Run records | S3 | 7 years minimum |
| Audit logs | CloudWatch / S3 | 7 years minimum |

See `runbooks/disaster_recovery.md` for DR drill procedures.

## Step 6: Governance Checkpoints

Before production, ensure all checkpoints are cleared:

- [ ] Use-case approval
- [ ] Data protection impact assessment (DPIA)
- [ ] Security review
- [ ] Participant agreements signed (see `templates/federation_agreement.md`)
- [ ] Model governance (model card, privacy testing)
- [ ] Operational readiness (runbooks, on-call, monitoring)
- [ ] Production release approval

## Step 7: Cost Tracking

Use the cost reporting template:

```bash
cat docs/cost-reporting.md
```

Key cost drivers:
- Coordinator compute (EC2 instance hours)
- Client compute (per participant)
- Storage (S3 for models, run records, audit logs)
- Key management (KMS)
- Logging (CloudWatch)

Tag all AWS resources with `Project`, `Environment`, and `Component` for cost allocation.

## Step 8: Troubleshooting Quick Reference

| Symptom | First Step |
|---------|-----------|
| Client can't connect | `./deploy/health_check.sh --quick` |
| mTLS failure | `openssl verify -CAfile certs/ca.pem certs/server.pem` |
| Training diverges | `python tools/dp_budget.py --all --rounds <N>` |
| SecAgg aborting | Check `secagg/config.yaml` min_quorum |
| DP budget exhausted | Expected — review round count vs budget |
| Image not found | `./deploy/distributed/deploy.sh distribute` |

See `docs/troubleshooting.md` for detailed resolution steps.

## What You Learned

- Production FL requires monitoring, cert management, and incident procedures
- Runbooks provide step-by-step guides for common operational tasks
- Governance checkpoints must be cleared before production
- Cost tracking uses AWS resource tagging

## Congratulations

You've completed all 12 tutorials. You now know how to:
- Run FL locally and in the cloud
- Apply DP, SecAgg, and privacy testing
- Choose the right FL strategy for your data
- Deploy and operate a production FL cluster
- Use VFL, split learning, and federated LLM fine-tuning

For reference material, see the [docs/](../../) directory.
