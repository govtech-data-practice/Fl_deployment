#!/bin/bash
# Validate cluster.env configuration
# Usage: ./deploy/validate_config.sh [cluster.env path]
set -uo pipefail

ENV_FILE="${1:-cluster.env}"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found"
  echo "  cp deploy/cluster.env.template cluster.env"
  echo "  vim cluster.env  # fill in values"
  exit 1
fi

source "$ENV_FILE"

ERRORS=0
WARNINGS=0

check_required() {
  local var="$1"
  local desc="$2"
  if [ -z "${!var:-}" ]; then
    echo "  ERROR: $var is not set ($desc)"
    ERRORS=$((ERRORS + 1))
  fi
}

check_optional() {
  local var="$1"
  local desc="$2"
  if [ -z "${!var:-}" ]; then
    echo "  WARN:  $var is not set ($desc)"
    WARNINGS=$((WARNINGS + 1))
  fi
}

check_file() {
  local var="$1"
  local path="${!var:-}"
  if [ -n "$path" ] && [ ! -f "$path" ]; then
    echo "  ERROR: $var=$path — file not found"
    ERRORS=$((ERRORS + 1))
  fi
}

check_ip() {
  local var="$1"
  local ip="${!var:-}"
  if [ -n "$ip" ] && ! echo "$ip" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "  ERROR: $var=$ip — not a valid IP address"
    ERRORS=$((ERRORS + 1))
  fi
}

echo "Validating $ENV_FILE..."
echo ""

# Required
echo "Required:"
check_required FL_SERVER_HOST "Server public IP"
check_ip FL_SERVER_HOST
check_required FL_SERVER_PRIVATE "Server private IP"
check_ip FL_SERVER_PRIVATE
check_required FL_CLIENT_HOSTS "Client IPs (space-separated)"
check_required FL_SSH_KEY "SSH private key path"
check_file FL_SSH_KEY
check_required FL_NUM_CLIENTS "Number of clients"

# Validate client count matches client hosts
if [ -n "${FL_CLIENT_HOSTS:-}" ]; then
  ACTUAL=$(echo "$FL_CLIENT_HOSTS" | wc -w | tr -d ' ')
  if [ "$ACTUAL" != "${FL_NUM_CLIENTS:-0}" ]; then
    echo "  ERROR: FL_NUM_CLIENTS=${FL_NUM_CLIENTS} but FL_CLIENT_HOSTS has $ACTUAL IPs"
    ERRORS=$((ERRORS + 1))
  fi
fi

echo ""
echo "Optional:"
check_optional FL_REGISTRY "Private Docker registry"
check_optional FL_BACKUP_S3_BUCKET "S3 backup bucket"
check_optional FL_VPC_ID "VPC ID"

echo ""
echo "Connectivity:"
# Test SSH to server
if [ -n "${FL_SERVER_HOST:-}" ] && [ -f "${FL_SSH_KEY:-/dev/null}" ]; then
  if ssh -i "$FL_SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
    "${FL_SSH_USER:-ec2-user}@$FL_SERVER_HOST" "echo OK" 2>/dev/null; then
    echo "  Server SSH: OK"
  else
    echo "  ERROR: Cannot SSH to server ($FL_SERVER_HOST)"
    ERRORS=$((ERRORS + 1))
  fi
fi

# Test SSH to first client
if [ -n "${FL_CLIENT_HOSTS:-}" ] && [ -f "${FL_SSH_KEY:-/dev/null}" ]; then
  FIRST_CLIENT=$(echo "$FL_CLIENT_HOSTS" | awk '{print $1}')
  if ssh -i "$FL_SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
    "${FL_SSH_USER:-ec2-user}@$FIRST_CLIENT" "echo OK" 2>/dev/null; then
    echo "  Client SSH ($FIRST_CLIENT): OK"
  else
    echo "  ERROR: Cannot SSH to client ($FIRST_CLIENT)"
    ERRORS=$((ERRORS + 1))
  fi
fi

echo ""
echo "================================================"
if [ $ERRORS -gt 0 ]; then
  echo "FAILED: $ERRORS error(s), $WARNINGS warning(s)"
  exit 1
else
  echo "PASSED: 0 errors, $WARNINGS warning(s)"
  exit 0
fi
