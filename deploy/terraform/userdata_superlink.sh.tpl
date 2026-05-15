#!/bin/bash
set -euo pipefail
exec > /var/log/fl-setup.log 2>&1

echo "=== SuperLink setup ==="

# Install Docker
dnf install -y docker
systemctl enable --now docker

# ECR login
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${ecr_repo}

# Pull image
docker pull ${ecr_repo}:${image_tag}

%{ if enable_tls }
# Fetch TLS certs from SSM
mkdir -p /opt/fl/certs
aws ssm get-parameter --name "/${project_name}/tls/ca-cert" --with-decryption --query 'Parameter.Value' --output text --region ${region} > /opt/fl/certs/ca.pem
aws ssm get-parameter --name "/${project_name}/tls/server-cert" --with-decryption --query 'Parameter.Value' --output text --region ${region} > /opt/fl/certs/server.pem
aws ssm get-parameter --name "/${project_name}/tls/server-key" --with-decryption --query 'Parameter.Value' --output text --region ${region} > /opt/fl/certs/server.key
chmod 600 /opt/fl/certs/server.key

TLS_FLAGS="--ssl-certfile /certs/server.pem --ssl-keyfile /certs/server.key --ssl-ca-certfile /certs/ca.pem"
CERT_MOUNT="-v /opt/fl/certs:/certs:ro"
%{ else }
TLS_FLAGS="--insecure"
CERT_MOUNT=""
%{ endif }

# Run SuperLink
docker run -d --restart unless-stopped \
  --name superlink \
  --network host \
  $CERT_MOUNT \
  ${ecr_repo}:${image_tag} \
  flower-superlink $TLS_FLAGS

echo "=== SuperLink running ==="
