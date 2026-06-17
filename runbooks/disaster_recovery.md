# Disaster Recovery Runbook

## Backup Scope

| Asset | Method | Location | Retention |
|-------|--------|----------|-----------|
| Infrastructure config (IaC) | Version control | Git + offsite mirror | Indefinite |
| Certificates & PKI material | Encrypted store | Secrets Manager / offline | Duration of federation |
| Model checkpoints | Versioned object storage | S3 | Per governance policy |
| Run records | Append-only storage | S3 / EBS | 7 years minimum |
| Audit logs | Immutable log storage | CloudWatch / S3 Glacier | 7 years minimum |
| Runbooks & documentation | Version control | Git | Indefinite |

## Backup Commands

```bash
# Manual backup
./deploy/backup.sh

# Verify backup integrity
./deploy/backup.sh --verify

# List available backups
./deploy/backup.sh --list
```

## DR Drill Procedure

Minimum frequency: Annual. Recommended: Quarterly.

### 1. Declare
- Announce DR drill to all participants
- Note start time

### 2. Restore Infrastructure Config
```bash
# From version control
git clone <repository-url> fl-reference-dr
cd fl-reference-dr
cp /path/to/backed-up/cluster.env .
```

### 3. Restore Model Checkpoints
```bash
# From S3
aws s3 sync s3://<backup-bucket>/models/ ./models-restore/
```

### 4. Validate Certificates
- Restore certificates from encrypted backup
- Verify certificate chain: `openssl verify -CAfile ca.pem server.pem`
- Check expiry dates

### 5. Run Smoke Test
```bash
python runners/run_ec2.py fraud --synthetic
```

### 6. Review
- Document recovery time (actual vs RTO target)
- Note any issues encountered
- Update procedures as needed

## Recovery Targets

| Metric | Target |
|--------|--------|
| RTO (Recovery Time Objective) | 4 hours |
| RPO (Recovery Point Objective) | 1 hour (last backup) |
| Recovery environment | Pre-provisioned, same region (ap-southeast-1) |
