#!/bin/bash
# FL Platform Rollback
# ====================
# Rolls back to a previous Docker image version on all cluster nodes.
#
# Usage:
#   ./deploy/rollback.sh --list             # list available versions
#   ./deploy/rollback.sh --to v1.2.3        # rollback to specific tag
#   ./deploy/rollback.sh --to previous      # rollback to the previous image
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../cluster.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: cluster.env not found"
  exit 1
fi
source "$ENV_FILE"

SSH="ssh -i ${FL_SSH_KEY} -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o LogLevel=ERROR"
on() { $SSH ${FL_SSH_USER}@"$1" "${@:2}" 2>/dev/null; }

MODE="${1:---list}"
TARGET_TAG="${2:-}"

# ── List versions ─────────────────────────────────────────────────

list_versions() {
  echo "Available Image Versions (server)"
  echo "================================="
  on "$FL_SERVER_HOST" "
    docker images ${FL_IMAGE} --format 'table {{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}' | head -20
  " 2>/dev/null

  echo ""
  echo "Current: ${FL_IMAGE}:${FL_IMAGE_TAG}"
  CURRENT_ID=$(on "$FL_SERVER_HOST" "docker inspect ${FL_IMAGE}:${FL_IMAGE_TAG} --format '{{.Id}}'" 2>/dev/null | cut -c8-19)
  echo "Image ID: $CURRENT_ID"
}

# ── Rollback ──────────────────────────────────────────────────────

rollback() {
  local NEW_TAG="$1"

  if [ "$NEW_TAG" = "previous" ]; then
    # Find the second most recent image
    NEW_TAG=$(on "$FL_SERVER_HOST" "
      docker images ${FL_IMAGE} --format '{{.Tag}}' | grep -v '<none>' | sed -n '2p'
    " 2>/dev/null)
    if [ -z "$NEW_TAG" ]; then
      echo "ERROR: No previous version found"
      exit 1
    fi
    echo "Previous version: ${FL_IMAGE}:${NEW_TAG}"
  fi

  # Verify target image exists on server
  EXISTS=$(on "$FL_SERVER_HOST" "docker images ${FL_IMAGE}:${NEW_TAG} --format '{{.ID}}'" 2>/dev/null)
  if [ -z "$EXISTS" ]; then
    echo "ERROR: ${FL_IMAGE}:${NEW_TAG} not found on server"
    echo "Available tags:"
    on "$FL_SERVER_HOST" "docker images ${FL_IMAGE} --format '  {{.Tag}} ({{.CreatedAt}})'" 2>/dev/null
    exit 1
  fi

  echo "FL Platform Rollback"
  echo "===================="
  echo "  From: ${FL_IMAGE}:${FL_IMAGE_TAG}"
  echo "  To:   ${FL_IMAGE}:${NEW_TAG}"
  echo ""

  # Stop all running FL containers
  echo "Stopping containers..."
  on "$FL_SERVER_HOST" "docker rm -f fl-orchestrator fl-training fl-superlink 2>/dev/null" || true
  for ip in $FL_CLIENT_HOSTS; do
    on "$ip" "sudo docker rm -f fl-client 2>/dev/null" &
  done
  wait

  # Tag the rollback target as :latest (or current tag)
  echo "Retagging ${FL_IMAGE}:${NEW_TAG} -> ${FL_IMAGE}:${FL_IMAGE_TAG}..."
  # Save current as :rollback-backup
  on "$FL_SERVER_HOST" "docker tag ${FL_IMAGE}:${FL_IMAGE_TAG} ${FL_IMAGE}:rollback-backup-$(date +%Y%m%d)" 2>/dev/null || true
  on "$FL_SERVER_HOST" "docker tag ${FL_IMAGE}:${NEW_TAG} ${FL_IMAGE}:${FL_IMAGE_TAG}"

  # Distribute to clients
  echo "Distributing to clients..."
  on "$FL_SERVER_HOST" "docker save ${FL_IMAGE}:${FL_IMAGE_TAG} | gzip > /tmp/fl-image-rollback.tar.gz"
  for ip in $FL_CLIENT_HOSTS; do
    (
      on "$FL_SERVER_HOST" "scp -i ~/.ssh/$(basename ${FL_SSH_KEY}) -o StrictHostKeyChecking=no /tmp/fl-image-rollback.tar.gz ${FL_SSH_USER}@${ip}:/tmp/" 2>/dev/null
      on "$ip" "sudo docker load < /tmp/fl-image-rollback.tar.gz" 2>/dev/null
      echo "  $ip: done"
    ) &
  done
  wait

  # Verify
  echo ""
  echo "Verifying..."
  SERVER_ID=$(on "$FL_SERVER_HOST" "docker inspect ${FL_IMAGE}:${FL_IMAGE_TAG} --format '{{.Id}}'" | cut -c8-19)
  echo "  Server: $SERVER_ID"
  for ip in $FL_CLIENT_HOSTS; do
    CLIENT_ID=$(on "$ip" "sudo docker inspect ${FL_IMAGE}:${FL_IMAGE_TAG} --format '{{.Id}}'" 2>/dev/null | cut -c8-19)
    if [ "$CLIENT_ID" = "$SERVER_ID" ]; then
      echo "  $ip: $CLIENT_ID (match)"
    else
      echo "  $ip: $CLIENT_ID (MISMATCH)"
    fi
  done

  echo ""
  echo "Rollback complete. Run deploy.sh to restart services."
}

# ── Tag current image ─────────────────────────────────────────────

tag_version() {
  local TAG="$1"
  echo "Tagging ${FL_IMAGE}:${FL_IMAGE_TAG} as ${FL_IMAGE}:${TAG}..."
  on "$FL_SERVER_HOST" "docker tag ${FL_IMAGE}:${FL_IMAGE_TAG} ${FL_IMAGE}:${TAG}"
  echo "Tagged. Push to registry: docker push ${FL_REGISTRY}/${FL_IMAGE}:${TAG}"
}

# ── Main ──────────────────────────────────────────────────────────

case "$MODE" in
  --list)          list_versions ;;
  --to)
    if [ -z "$TARGET_TAG" ]; then
      echo "Usage: ./deploy/rollback.sh --to <tag|previous>"
      exit 1
    fi
    rollback "$TARGET_TAG"
    ;;
  --tag)
    if [ -z "$TARGET_TAG" ]; then
      echo "Usage: ./deploy/rollback.sh --tag v1.2.3"
      exit 1
    fi
    tag_version "$TARGET_TAG"
    ;;
  *)
    echo "Usage: ./deploy/rollback.sh [--list|--to <tag>|--tag <version>]"
    exit 1
    ;;
esac
