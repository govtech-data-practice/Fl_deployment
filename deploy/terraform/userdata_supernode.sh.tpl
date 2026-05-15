#!/bin/bash
set -euo pipefail
exec > /var/log/fl-setup.log 2>&1

echo "=== SuperNode ${partition_id} setup ==="

# Install Docker
if command -v dnf &>/dev/null; then
  dnf install -y docker
  systemctl enable --now docker
elif command -v apt-get &>/dev/null; then
  apt-get update && apt-get install -y docker.io
  systemctl enable --now docker
fi

%{ if use_gpu }
# Nvidia container toolkit (Deep Learning AMI usually has this)
if ! command -v nvidia-container-toolkit &>/dev/null; then
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update && apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi
GPU_FLAG="--gpus all"
%{ else }
GPU_FLAG=""
%{ endif }

# ECR login
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${ecr_repo}

# Pull image
docker pull ${ecr_repo}:${image_tag}

# Download training data from S3
mkdir -p /opt/fl/data
aws s3 sync s3://${data_s3_bucket}/${data_s3_prefix} /opt/fl/data/ --region ${region}
echo "Data downloaded: $(du -sh /opt/fl/data/)"

%{ if enable_tls }
# Fetch CA cert from SSM
mkdir -p /opt/fl/certs
aws ssm get-parameter --name "/${project_name}/tls/ca-cert" --with-decryption --query 'Parameter.Value' --output text --region ${region} > /opt/fl/certs/ca.pem

TLS_FLAGS="--root-certificates /certs/ca.pem"
CERT_MOUNT="-v /opt/fl/certs:/certs:ro"
%{ else }
TLS_FLAGS="--insecure"
CERT_MOUNT=""
%{ endif }

# Wait for SuperLink
echo "Waiting for SuperLink at ${superlink_ip}:9092..."
for i in $(seq 1 60); do
  if timeout 2 bash -c "echo > /dev/tcp/${superlink_ip}/9092" 2>/dev/null; then
    echo "SuperLink reachable"
    break
  fi
  sleep 5
done

# Run SuperNode
docker run -d --restart unless-stopped \
  --name supernode-${partition_id} \
  --network host \
  $GPU_FLAG \
  $CERT_MOUNT \
  -v /opt/fl/data:${data_dir}:ro \
  -e SYNTHETIC=0 \
  ${ecr_repo}:${image_tag} \
  flower-supernode \
    $TLS_FLAGS \
    --superlink ${superlink_ip}:9092 \
    --node-config "partition-id=${partition_id} num-clients=${num_clients} data-path=${data_dir}"

echo "=== SuperNode ${partition_id} running ==="
