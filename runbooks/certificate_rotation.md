# Certificate Rotation Runbook

## Overview

All FL communication is secured with mTLS. Certificates must be rotated before expiry.

**Automated alerts:**
- 30 days before expiry: P3 notification
- 14 days before expiry: P2 alert
- 7 days before expiry: P1 alert

## Pre-rotation Checklist

- [ ] Current certificate expiry date confirmed
- [ ] New certificate validity period agreed
- [ ] All participants notified of rotation window
- [ ] No training runs in progress
- [ ] Backup of current certificates taken

## Rotation Procedure

### 1. Generate New Certificates

```bash
# Generate new mTLS certificates
openssl req -x509 -newkey rsa:4096 -keyout ca.key -out ca.pem -days 365 -nodes

# Verify the new certificates
openssl x509 -in certs/server.pem -noout -subject -dates
openssl x509 -in certs/client.pem -noout -subject -dates
```

### 2. Distribute to Participants

```bash
# Uses deploy.sh to distribute certs to all nodes
# Distribute certs via scp to each node
```

### 3. Restart Services

```bash
# Rolling restart (coordinator first, then clients one at a time)
docker compose -f deploy/microservices/docker-compose.yml restart
```

### 4. Validate

```bash
# Verify mTLS connectivity
docker compose -f deploy/microservices/docker-compose.yml ps

# Run smoke test
python runners/run_ec2.py fraud --synthetic
```

### 5. Update Certificate Register

Record in the certificate register:
- New certificate serial number
- Issue date and expiry date
- CN and SANs
- Issuing CA

### 6. Revoke Old Certificates

- Add old certificates to CRL
- Distribute updated CRL to all participants
- Verify CRL is loaded

## CA Rollover

If the subordinate CA certificate is expiring:

1. Issue new subordinate CA from offline root CA
2. Cross-sign: new subordinate CA signed by both old and new root (if root is changing)
3. Distribute new CA bundle to all participants
4. Issue new end-entity certificates under new CA
5. Follow standard rotation procedure above

## Emergency Rotation

If a certificate is compromised:

1. **Immediately** revoke the compromised certificate
2. Update CRL and distribute to all nodes
3. Generate replacement certificate
4. Follow steps 2-5 of standard rotation
5. File P0 incident report
