# Incident Response Runbook

## Severity Ladder

| Severity | Response Time | Examples |
|----------|--------------|---------|
| P0 — Critical | < 15 min | mTLS regression, coordinator compromise, data breach |
| P1 — High | < 1 hour | Coordinator unavailable, DP budget exhausted |
| P2 — Medium | < 4 hours | Certificate near expiry, SecAgg abort rate high, validation failures |
| P3 — Low | Next business day | Non-blocking warnings, documentation gaps |

## P0/P1 Response Procedure

### 1. Acknowledge
- Confirm receipt within response time window
- Assign incident commander

### 2. Contain
- **mTLS regression:** Immediately halt all training runs
- **Coordinator down:** Check container health, restart if necessary
- **DP budget exhausted:** Halt training (automatic if properly configured)

### 3. Notify
- Alert all federation participants
- Escalate to security team for P0

### 4. Diagnose
```bash
./scripts/diagnose.sh --run-id <affected-run-id> --env production --since 4h
```
Review the diagnostic bundle:
- `system_info.txt` — resource exhaustion?
- `cert_status.txt` — certificate issues?
- `docker_logs.txt` — application errors?
- `health_check.json` — which checks failed?

### 5. Remediate
Apply the fix. Common remediation paths:
- Container restart: `./deploy/distributed/deploy.sh restart`
- Certificate rotation: `./deploy/rotate_certs.sh`
- Rollback: `./deploy/rollback.sh`

### 6. Verify
- Run smoke test: `python run_ec2.py fraud --synthetic`
- Run health check: `./deploy/health_check.sh`

### 7. Resume
- Resume training only after verification passes
- Notify all participants

### 8. Post-incident Review
- Complete within 5 business days
- Document: timeline, root cause, impact, remediation, preventive measures
- Update runbooks if new failure mode discovered

## Dead-Man Heartbeat Probes

| Probe | Interval | Action on Failure |
|-------|----------|-------------------|
| mTLS connectivity | 5 min | P0 alert |
| Certificate expiry | Daily | P2 alert at 14 days, P1 at 7 days |
| CRL refresh | 4 hours | P2 alert |
| Enclave attestation | 15 min | P1 alert |
