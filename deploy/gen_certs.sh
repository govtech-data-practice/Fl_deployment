#!/bin/bash
# Generate self-signed TLS certificates for FL deployment
# Usage: ./gen_certs.sh [output_dir]
#
# Creates:
#   ca.pem        — CA certificate (distribute to all nodes)
#   server.pem    — Server certificate (superlink only)
#   server.key    — Server private key (superlink only)

set -euo pipefail

OUT="${1:-./certs}"
mkdir -p "$OUT"

DAYS=365
CA_SUBJ="/CN=Healthcare-FL-CA"
SERVER_SUBJ="/CN=superlink"

echo "=== Generating CA ==="
openssl ecparam -genkey -name prime256v1 -out "$OUT/ca.key" 2>/dev/null
openssl req -new -x509 -key "$OUT/ca.key" -out "$OUT/ca.pem" \
  -days $DAYS -subj "$CA_SUBJ" 2>/dev/null

echo "=== Generating Server cert ==="
openssl ecparam -genkey -name prime256v1 -out "$OUT/server.key" 2>/dev/null

# SAN config for superlink hostname + localhost
cat > "$OUT/san.cnf" << EOF
[req]
distinguished_name = req_dn
req_extensions = v3_req
prompt = no

[req_dn]
CN = superlink

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = superlink
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF

openssl req -new -key "$OUT/server.key" -out "$OUT/server.csr" \
  -config "$OUT/san.cnf" 2>/dev/null

openssl x509 -req -in "$OUT/server.csr" \
  -CA "$OUT/ca.pem" -CAkey "$OUT/ca.key" -CAcreateserial \
  -out "$OUT/server.pem" -days $DAYS \
  -extfile "$OUT/san.cnf" -extensions v3_req 2>/dev/null

# Cleanup intermediates
rm -f "$OUT/server.csr" "$OUT/san.cnf" "$OUT/ca.key" "$OUT/ca.srl"

echo "=== Certificates generated ==="
echo "  CA cert:     $OUT/ca.pem"
echo "  Server cert: $OUT/server.pem"
echo "  Server key:  $OUT/server.key"

# Verify
echo ""
echo "=== Verification ==="
openssl verify -CAfile "$OUT/ca.pem" "$OUT/server.pem"
openssl x509 -in "$OUT/server.pem" -noout -subject -issuer -dates -ext subjectAltName 2>/dev/null
