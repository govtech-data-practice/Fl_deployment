#!/bin/bash
# TLS Certificate Rotation
# ========================
# Generates new TLS certificates and distributes to all cluster nodes.
# Restarts affected containers and verifies TLS handshake.
#
# Usage:
#   ./deploy/rotate_certs.sh              # generate + distribute + restart
#   ./deploy/rotate_certs.sh --check      # check expiry only
#   ./deploy/rotate_certs.sh --generate   # generate only (no distribute)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../cluster.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: cluster.env not found"
  exit 1
fi
source "$ENV_FILE"

SSH="ssh -i ${FL_SSH_KEY} -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o LogLevel=ERROR"
SCP="scp -i ${FL_SSH_KEY} -o StrictHostKeyChecking=no -o LogLevel=ERROR"
on() { $SSH ${FL_SSH_USER}@"$1" "${@:2}" 2>/dev/null; }

MODE="${1:---full}"
CERTS_DIR="${SCRIPT_DIR}/distributed/certs"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${REPO_ROOT}/backups/certs_$(date -u +%Y%m%d_%H%M%S)"

# ── Check expiry ──────────────────────────────────────────────────
check_expiry() {
  echo "Certificate Expiry Check"
  echo "========================"

  if [ ! -f "$CERTS_DIR/server.pem" ]; then
    echo "  No certificates found at $CERTS_DIR"
    exit 1
  fi

  EXPIRY=$(openssl x509 -enddate -noout -in "$CERTS_DIR/server.pem" | cut -d= -f2)
  DAYS_LEFT=$(( ( $(date -d "$EXPIRY" +%s 2>/dev/null || date -j -f "%b %d %T %Y %Z" "$EXPIRY" +%s 2>/dev/null || echo 0) - $(date +%s) ) / 86400 ))

  echo "  Server cert expires: $EXPIRY"
  echo "  Days remaining: $DAYS_LEFT"

  if [ "$DAYS_LEFT" -lt 0 ]; then
    echo "  STATUS: EXPIRED — rotate immediately"
    return 2
  elif [ "$DAYS_LEFT" -lt 30 ]; then
    echo "  STATUS: EXPIRING SOON — schedule rotation"
    return 1
  else
    echo "  STATUS: OK"
    return 0
  fi
}

# ── Generate new certs ────────────────────────────────────────────
generate_certs() {
  echo "Generating new TLS certificates..."
  echo "  CN: ${FL_CERT_CN:-fl-server}"
  echo "  Validity: ${FL_CERT_DAYS:-365} days"
  echo "  SANs: ${FL_SERVER_PRIVATE}, ${FL_SERVER_HOST}"

  # Backup old certs
  if [ -d "$CERTS_DIR" ] && [ -f "$CERTS_DIR/server.pem" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$CERTS_DIR"/*.pem "$CERTS_DIR"/*.key "$BACKUP_DIR/" 2>/dev/null || true
    echo "  Old certs backed up to $BACKUP_DIR"
  fi

  mkdir -p "$CERTS_DIR"

  # Generate CA
  openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/ca.key" 2>/dev/null
  openssl req -new -x509 -key "$CERTS_DIR/ca.key" -out "$CERTS_DIR/ca.pem" \
    -days "${FL_CERT_DAYS:-365}" -subj "/CN=${FL_CERT_CN:-fl-server}-CA" 2>/dev/null

  # Generate server cert with SAN
  cat > "$CERTS_DIR/san.cnf" << EOF
[req]
distinguished_name=dn
req_extensions=v3
prompt=no
[dn]
CN=${FL_CERT_CN:-fl-server}
[v3]
subjectAltName=DNS:${FL_CERT_CN:-fl-server},DNS:localhost,IP:127.0.0.1,IP:${FL_SERVER_PRIVATE},IP:${FL_SERVER_HOST}
EOF

  openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/server.key" 2>/dev/null
  openssl req -new -key "$CERTS_DIR/server.key" -out "$CERTS_DIR/s.csr" \
    -config "$CERTS_DIR/san.cnf" 2>/dev/null
  openssl x509 -req -in "$CERTS_DIR/s.csr" -CA "$CERTS_DIR/ca.pem" -CAkey "$CERTS_DIR/ca.key" \
    -CAcreateserial -out "$CERTS_DIR/server.pem" -days "${FL_CERT_DAYS:-365}" \
    -extfile "$CERTS_DIR/san.cnf" -extensions v3 2>/dev/null

  # Clean up temp files (keep ca.key for future rotation)
  rm -f "$CERTS_DIR"/{s.csr,san.cnf,ca.srl}

  echo "  Generated:"
  echo "    $CERTS_DIR/ca.pem      (CA certificate — distribute to all nodes)"
  echo "    $CERTS_DIR/ca.key      (CA private key — keep secure)"
  echo "    $CERTS_DIR/server.pem  (Server certificate)"
  echo "    $CERTS_DIR/server.key  (Server private key)"

  # Verify
  openssl x509 -in "$CERTS_DIR/server.pem" -noout -dates -subject 2>/dev/null | sed 's/^/    /'
}

# ── Distribute to cluster ────────────────────────────────────────
distribute_certs() {
  echo ""
  echo "Distributing certificates..."

  # Server
  echo "  Server ($FL_SERVER_HOST):"
  on "$FL_SERVER_HOST" "mkdir -p ${FL_CERTS_DIR}"
  $SCP "$CERTS_DIR/server.pem" "$CERTS_DIR/server.key" "$CERTS_DIR/ca.pem" \
    "${FL_SSH_USER}@${FL_SERVER_HOST}:${FL_CERTS_DIR}/"
  on "$FL_SERVER_HOST" "chmod 600 ${FL_CERTS_DIR}/server.key"
  echo "    server.pem, server.key, ca.pem — OK"

  # Clients (only need CA cert)
  for ip in $FL_CLIENT_HOSTS; do
    echo "  Client ($ip):"
    on "$ip" "mkdir -p ${FL_CERTS_DIR}"
    $SCP "$CERTS_DIR/ca.pem" "${FL_SSH_USER}@${ip}:${FL_CERTS_DIR}/"
    echo "    ca.pem — OK"
  done
}

# ── Restart containers ────────────────────────────────────────────
restart_containers() {
  echo ""
  echo "Restarting containers with new certs..."

  # Stop training containers (certs are mounted read-only, need restart)
  on "$FL_SERVER_HOST" "docker rm -f fl-superlink fl-training 2>/dev/null || true"
  for ip in $FL_CLIENT_HOSTS; do
    on "$ip" "sudo docker rm -f fl-client 2>/dev/null || true" &
  done
  wait

  # Restart SuperLink for idle monitoring
  on "$FL_SERVER_HOST" "
    docker run -d --name fl-superlink --restart unless-stopped --network host \
      -v ${FL_CERTS_DIR}:/certs:ro \
      --log-opt max-size=${FL_LOG_MAX_SIZE:-200m} \
      ${FL_IMAGE}:${FL_IMAGE_TAG} \
      flower-superlink --ssl-certfile /certs/server.pem --ssl-keyfile /certs/server.key --ssl-ca-certfile /certs/ca.pem
  " >/dev/null 2>&1

  echo "  SuperLink restarted with new certs"
}

# ── Verify TLS handshake ─────────────────────────────────────────
verify_tls() {
  echo ""
  echo "Verifying TLS handshake..."

  # Check server cert from each client's perspective
  for ip in $FL_CLIENT_HOSTS; do
    RESULT=$(on "$ip" "openssl s_client -connect ${FL_SERVER_PRIVATE}:${FL_GRPC_PORT} \
      -CAfile ${FL_CERTS_DIR}/ca.pem </dev/null 2>/dev/null | grep 'Verify return code'" 2>/dev/null || echo "FAILED")
    if echo "$RESULT" | grep -q "ok"; then
      echo "  $ip -> server: TLS OK"
    else
      echo "  $ip -> server: TLS FAILED ($RESULT)"
    fi
  done
}

# ── Main ──────────────────────────────────────────────────────────
case "$MODE" in
  --check)
    check_expiry
    ;;
  --generate)
    generate_certs
    ;;
  --full|*)
    generate_certs
    distribute_certs
    restart_containers
    verify_tls
    echo ""
    echo "Certificate rotation complete."
    ;;
esac
