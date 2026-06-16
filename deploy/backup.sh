#!/bin/bash
# FL Platform Backup and Recovery
# ================================
# Backs up results, certificates, configurations, and data manifests.
# Supports local and S3 targets.
#
# Usage:
#   ./deploy/backup.sh                   # full backup (local)
#   ./deploy/backup.sh --s3              # full backup to S3
#   ./deploy/backup.sh --results-only    # results only
#   ./deploy/backup.sh --restore <path>  # restore from backup
#   ./deploy/backup.sh --list            # list available backups
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${REPO_ROOT}/cluster.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: cluster.env not found"
  exit 1
fi
source "$ENV_FILE"

SSH="ssh -i ${FL_SSH_KEY} -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o LogLevel=ERROR"
SCP="scp -i ${FL_SSH_KEY} -o StrictHostKeyChecking=no -o LogLevel=ERROR"
on() { $SSH ${FL_SSH_USER}@"$1" "${@:2}" 2>/dev/null; }

MODE="${1:---full}"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
BACKUP_BASE="${REPO_ROOT}/backups"
BACKUP_DIR="${BACKUP_BASE}/backup_${TIMESTAMP}"

# ── Backup functions ──────────────────────────────────────────────

backup_results() {
  echo "Backing up results..."
  mkdir -p "$BACKUP_DIR/results"
  $SCP "${FL_SSH_USER}@${FL_SERVER_HOST}:${FL_RESULTS_DIR}/*" "$BACKUP_DIR/results/" 2>/dev/null || true
  COUNT=$(ls "$BACKUP_DIR/results/" 2>/dev/null | wc -l | tr -d ' ')
  echo "  $COUNT result files"
}

backup_certs() {
  echo "Backing up certificates..."
  mkdir -p "$BACKUP_DIR/certs"
  $SCP "${FL_SSH_USER}@${FL_SERVER_HOST}:${FL_CERTS_DIR}/*.pem" "$BACKUP_DIR/certs/" 2>/dev/null || true
  $SCP "${FL_SSH_USER}@${FL_SERVER_HOST}:${FL_CERTS_DIR}/*.key" "$BACKUP_DIR/certs/" 2>/dev/null || true
  echo "  Certs backed up"
}

backup_config() {
  echo "Backing up configuration..."
  mkdir -p "$BACKUP_DIR/config"
  cp "$ENV_FILE" "$BACKUP_DIR/config/cluster.env" 2>/dev/null || true
  # Save Docker image metadata
  on "$FL_SERVER_HOST" "docker inspect ${FL_IMAGE}:${FL_IMAGE_TAG} --format '{{.Id}} {{.Created}}'" \
    > "$BACKUP_DIR/config/image_info.txt" 2>/dev/null || true
  # Save running container state
  on "$FL_SERVER_HOST" "docker ps --format '{{.Names}} {{.Status}} {{.Image}}'" \
    > "$BACKUP_DIR/config/containers.txt" 2>/dev/null || true
  echo "  Config backed up"
}

backup_manifests() {
  echo "Backing up data manifests..."
  mkdir -p "$BACKUP_DIR/manifests"
  # Collect manifests from all clients
  CLIENT_IDX=0
  for ip in $FL_CLIENT_HOSTS; do
    MANIFEST_DIR="$BACKUP_DIR/manifests/client_${CLIENT_IDX}"
    mkdir -p "$MANIFEST_DIR"
    on "$ip" "find ${FL_DATA_DIR} -name manifest.json" | while read -r f; do
      TASK=$(echo "$f" | grep -oE '[^/]+/manifest.json' | cut -d/ -f1)
      $SCP "${FL_SSH_USER}@${ip}:${f}" "$MANIFEST_DIR/${TASK}_manifest.json" 2>/dev/null || true
    done
    CLIENT_IDX=$((CLIENT_IDX + 1))
  done
  echo "  Manifests collected from $CLIENT_IDX clients"
}

# ── Full backup ───────────────────────────────────────────────────

full_backup() {
  echo "FL Platform Backup"
  echo "=================="
  echo "  Target: $BACKUP_DIR"
  echo "  Server: $FL_SERVER_HOST"
  echo "  Clients: $FL_NUM_CLIENTS"
  echo ""

  backup_results
  backup_certs
  backup_config
  backup_manifests

  # Create archive
  ARCHIVE="${BACKUP_BASE}/backup_${TIMESTAMP}.tar.gz"
  tar czf "$ARCHIVE" -C "$BACKUP_BASE" "backup_${TIMESTAMP}" 2>/dev/null
  rm -rf "$BACKUP_DIR"
  SIZE=$(du -h "$ARCHIVE" | cut -f1)

  echo ""
  echo "Backup complete: $ARCHIVE ($SIZE)"

  # Upload to S3 if requested
  if [ "${1:-}" = "--s3" ] && [ -n "${FL_BACKUP_S3_BUCKET:-}" ]; then
    echo "Uploading to s3://${FL_BACKUP_S3_BUCKET}/backups/..."
    aws s3 cp "$ARCHIVE" "s3://${FL_BACKUP_S3_BUCKET}/backups/" \
      --region "${AWS_REGION:-ap-southeast-1}" 2>/dev/null
    echo "  S3 upload complete"
  fi

  # Cleanup old backups
  if [ -n "${FL_BACKUP_RETENTION_DAYS:-}" ]; then
    find "$BACKUP_BASE" -name "backup_*.tar.gz" -mtime "+${FL_BACKUP_RETENTION_DAYS}" -delete 2>/dev/null
    echo "  Old backups (>${FL_BACKUP_RETENTION_DAYS} days) cleaned up"
  fi
}

# ── Restore ───────────────────────────────────────────────────────

restore_backup() {
  local ARCHIVE="$1"
  if [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: Backup file not found: $ARCHIVE"
    exit 1
  fi

  echo "FL Platform Restore"
  echo "==================="
  echo "  Source: $ARCHIVE"
  echo ""

  # Extract
  RESTORE_DIR=$(mktemp -d)
  tar xzf "$ARCHIVE" -C "$RESTORE_DIR"
  BACKUP_NAME=$(ls "$RESTORE_DIR")
  RESTORE_PATH="$RESTORE_DIR/$BACKUP_NAME"

  # Restore results
  if [ -d "$RESTORE_PATH/results" ]; then
    echo "Restoring results..."
    $SCP "$RESTORE_PATH/results/"* "${FL_SSH_USER}@${FL_SERVER_HOST}:${FL_RESULTS_DIR}/" 2>/dev/null
    echo "  $(ls "$RESTORE_PATH/results/" | wc -l | tr -d ' ') result files restored"
  fi

  # Restore certs
  if [ -d "$RESTORE_PATH/certs" ]; then
    echo "Restoring certificates..."
    on "$FL_SERVER_HOST" "mkdir -p ${FL_CERTS_DIR}"
    $SCP "$RESTORE_PATH/certs/"* "${FL_SSH_USER}@${FL_SERVER_HOST}:${FL_CERTS_DIR}/" 2>/dev/null
    on "$FL_SERVER_HOST" "chmod 600 ${FL_CERTS_DIR}/*.key 2>/dev/null || true"
    echo "  Certs restored to server"

    # Distribute CA to clients
    for ip in $FL_CLIENT_HOSTS; do
      on "$ip" "mkdir -p ${FL_CERTS_DIR}"
      $SCP "$RESTORE_PATH/certs/ca.pem" "${FL_SSH_USER}@${ip}:${FL_CERTS_DIR}/" 2>/dev/null
    done
    echo "  CA cert distributed to clients"
  fi

  rm -rf "$RESTORE_DIR"
  echo ""
  echo "Restore complete. Restart containers to apply restored certs."
}

# ── List backups ──────────────────────────────────────────────────

list_backups() {
  echo "Available Backups"
  echo "================="
  echo ""
  echo "Local ($BACKUP_BASE):"
  ls -lth "$BACKUP_BASE"/backup_*.tar.gz 2>/dev/null | awk '{print "  " $5 " " $6 " " $7 " " $8 " " $NF}' || echo "  (none)"

  if [ -n "${FL_BACKUP_S3_BUCKET:-}" ]; then
    echo ""
    echo "S3 (s3://${FL_BACKUP_S3_BUCKET}/backups/):"
    aws s3 ls "s3://${FL_BACKUP_S3_BUCKET}/backups/" --region "${AWS_REGION:-ap-southeast-1}" 2>/dev/null \
      | awk '{print "  " $3 " " $4}' || echo "  (not configured or no access)"
  fi
}

# ── Main ──────────────────────────────────────────────────────────

case "$MODE" in
  --full)         full_backup ;;
  --s3)           full_backup --s3 ;;
  --results-only)
    mkdir -p "$BACKUP_DIR"
    backup_results
    tar czf "${BACKUP_BASE}/results_${TIMESTAMP}.tar.gz" -C "$BACKUP_BASE" "backup_${TIMESTAMP}"
    rm -rf "$BACKUP_DIR"
    echo "Results backup: ${BACKUP_BASE}/results_${TIMESTAMP}.tar.gz"
    ;;
  --restore)
    if [ -z "${2:-}" ]; then
      echo "Usage: ./deploy/backup.sh --restore <backup.tar.gz>"
      exit 1
    fi
    restore_backup "$2"
    ;;
  --list)         list_backups ;;
  *)
    echo "Usage: ./deploy/backup.sh [--full|--s3|--results-only|--restore <path>|--list]"
    exit 1
    ;;
esac
