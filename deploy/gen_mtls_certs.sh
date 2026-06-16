#!/bin/bash
# Generate mTLS Certificates for FL Cluster
# ==========================================
# Creates CA, server cert, and per-client certificates.
# Server verifies client identity via client certificate CN.
#
# Output:
#   certs/ca.pem              — CA certificate (distribute to all)
#   certs/ca.key              — CA private key (keep secure, needed for new clients)
#   certs/server.pem          — Server certificate
#   certs/server.key          — Server private key
#   certs/client_<N>.pem      — Client N certificate
#   certs/client_<N>.key      — Client N private key
#
# Usage:
#   ./deploy/gen_mtls_certs.sh                    # generate all
#   ./deploy/gen_mtls_certs.sh --add-client 5     # add a new client cert
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${REPO_ROOT}/cluster.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: cluster.env not found"
  exit 1
fi
source "$ENV_FILE"

CERTS_DIR="${SCRIPT_DIR}/distributed/certs"
DAYS="${FL_CERT_DAYS:-365}"
CN="${FL_CERT_CN:-fl-server}"

mkdir -p "$CERTS_DIR"

generate_ca() {
  echo "Generating CA..."
  openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/ca.key" 2>/dev/null
  openssl req -new -x509 -key "$CERTS_DIR/ca.key" -out "$CERTS_DIR/ca.pem" \
    -days "$DAYS" -subj "/CN=${CN}-CA/O=FL-Platform" 2>/dev/null
  echo "  CA: $CERTS_DIR/ca.pem (valid $DAYS days)"
}

generate_server_cert() {
  echo "Generating server certificate..."
  # SAN includes both private and public IPs
  cat > "$CERTS_DIR/server_san.cnf" << EOF
[req]
distinguished_name=dn
req_extensions=v3
prompt=no
[dn]
CN=${CN}
O=FL-Platform
[v3]
subjectAltName=DNS:${CN},DNS:localhost,IP:127.0.0.1,IP:${FL_SERVER_PRIVATE},IP:${FL_SERVER_HOST}
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
EOF

  openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/server.key" 2>/dev/null
  openssl req -new -key "$CERTS_DIR/server.key" -out "$CERTS_DIR/server.csr" \
    -config "$CERTS_DIR/server_san.cnf" 2>/dev/null
  openssl x509 -req -in "$CERTS_DIR/server.csr" \
    -CA "$CERTS_DIR/ca.pem" -CAkey "$CERTS_DIR/ca.key" -CAcreateserial \
    -out "$CERTS_DIR/server.pem" -days "$DAYS" \
    -extfile "$CERTS_DIR/server_san.cnf" -extensions v3 2>/dev/null
  rm -f "$CERTS_DIR/server.csr" "$CERTS_DIR/server_san.cnf" "$CERTS_DIR/ca.srl"

  echo "  Server: $CERTS_DIR/server.pem"
  echo "  SANs: ${FL_SERVER_PRIVATE}, ${FL_SERVER_HOST}"
}

generate_client_cert() {
  local idx="$1"
  local client_cn="fl-client-${idx}"
  echo "Generating client $idx certificate (CN=$client_cn)..."

  cat > "$CERTS_DIR/client_${idx}_san.cnf" << EOF
[req]
distinguished_name=dn
req_extensions=v3
prompt=no
[dn]
CN=${client_cn}
O=FL-Platform
[v3]
keyUsage=digitalSignature
extendedKeyUsage=clientAuth
EOF

  openssl ecparam -genkey -name prime256v1 -out "$CERTS_DIR/client_${idx}.key" 2>/dev/null
  openssl req -new -key "$CERTS_DIR/client_${idx}.key" -out "$CERTS_DIR/client_${idx}.csr" \
    -config "$CERTS_DIR/client_${idx}_san.cnf" 2>/dev/null
  openssl x509 -req -in "$CERTS_DIR/client_${idx}.csr" \
    -CA "$CERTS_DIR/ca.pem" -CAkey "$CERTS_DIR/ca.key" -CAcreateserial \
    -out "$CERTS_DIR/client_${idx}.pem" -days "$DAYS" \
    -extfile "$CERTS_DIR/client_${idx}_san.cnf" -extensions v3 2>/dev/null
  rm -f "$CERTS_DIR/client_${idx}.csr" "$CERTS_DIR/client_${idx}_san.cnf" "$CERTS_DIR/ca.srl"

  echo "  Client $idx: $CERTS_DIR/client_${idx}.pem"
}

distribute_certs() {
  echo ""
  echo "Distributing certificates..."

  # Server gets: ca.pem, server.pem, server.key
  local SSH="ssh -i ${FL_SSH_KEY} -o StrictHostKeyChecking=no -o LogLevel=ERROR"
  local SCP="scp -i ${FL_SSH_KEY} -o StrictHostKeyChecking=no -o LogLevel=ERROR"

  $SSH ${FL_SSH_USER}@${FL_SERVER_HOST} "mkdir -p ${FL_CERTS_DIR}"
  $SCP "$CERTS_DIR/ca.pem" "$CERTS_DIR/server.pem" "$CERTS_DIR/server.key" \
    ${FL_SSH_USER}@${FL_SERVER_HOST}:${FL_CERTS_DIR}/
  $SSH ${FL_SSH_USER}@${FL_SERVER_HOST} "chmod 600 ${FL_CERTS_DIR}/server.key"
  echo "  Server: ca.pem, server.pem, server.key"

  # Each client gets: ca.pem, client_N.pem, client_N.key
  local idx=0
  for ip in $FL_CLIENT_HOSTS; do
    $SSH ${FL_SSH_USER}@${ip} "mkdir -p ${FL_CERTS_DIR}"
    $SCP "$CERTS_DIR/ca.pem" "$CERTS_DIR/client_${idx}.pem" "$CERTS_DIR/client_${idx}.key" \
      ${FL_SSH_USER}@${ip}:${FL_CERTS_DIR}/
    $SSH ${FL_SSH_USER}@${ip} "chmod 600 ${FL_CERTS_DIR}/client_${idx}.key"
    echo "  Client $idx ($ip): ca.pem, client_${idx}.pem, client_${idx}.key"
    idx=$((idx + 1))
  done
}

verify_certs() {
  echo ""
  echo "Verifying certificates..."
  echo "  CA:"
  openssl x509 -in "$CERTS_DIR/ca.pem" -noout -subject -dates 2>/dev/null | sed 's/^/    /'
  echo "  Server:"
  openssl x509 -in "$CERTS_DIR/server.pem" -noout -subject -dates 2>/dev/null | sed 's/^/    /'
  openssl verify -CAfile "$CERTS_DIR/ca.pem" "$CERTS_DIR/server.pem" 2>/dev/null | sed 's/^/    /'

  for f in "$CERTS_DIR"/client_*.pem; do
    [ -f "$f" ] || continue
    local name=$(basename "$f" .pem)
    echo "  $name:"
    openssl verify -CAfile "$CERTS_DIR/ca.pem" "$f" 2>/dev/null | sed 's/^/    /'
  done
}

# ── Main ──────────────────────────────────────────────────────────────

case "${1:---full}" in
  --full)
    generate_ca
    generate_server_cert
    for i in $(seq 0 $((FL_NUM_CLIENTS - 1))); do
      generate_client_cert "$i"
    done
    verify_certs
    distribute_certs
    echo ""
    echo "mTLS setup complete."
    echo "  Server: authenticates with server.pem"
    echo "  Clients: authenticate with client_N.pem (CN=fl-client-N)"
    echo "  Both sides verify against ca.pem"
    ;;
  --add-client)
    idx="${2:?Usage: --add-client <index>}"
    generate_client_cert "$idx"
    verify_certs
    echo "Distribute manually: scp client_${idx}.pem client_${idx}.key ca.pem to the new client"
    ;;
  --verify)
    verify_certs
    ;;
  *)
    echo "Usage: $0 [--full|--add-client N|--verify]"
    ;;
esac
