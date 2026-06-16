#!/bin/bash
# FL Platform Health Check
# ========================
# Verifies all cluster components are operational.
# Exit code 0 = healthy, 1 = degraded, 2 = critical
#
# Usage:
#   ./deploy/health_check.sh                    # full check
#   ./deploy/health_check.sh --quick            # server only
#   ./deploy/health_check.sh --json             # machine-readable output
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../cluster.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: cluster.env not found. Run: cp deploy/cluster.env.template cluster.env"
  exit 2
fi
source "$ENV_FILE"

SSH="ssh -i ${FL_SSH_KEY} -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o LogLevel=ERROR"
on() { $SSH ${FL_SSH_USER}@"$1" "${@:2}" 2>/dev/null; }

QUICK=false
JSON=false
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=true ;;
    --json)  JSON=true ;;
  esac
done

PASS=0
FAIL=0
WARN=0
CHECKS=()

check() {
  local name="$1"
  local status="$2"  # pass, fail, warn
  local detail="${3:-}"
  CHECKS+=("${name}|${status}|${detail}")
  case "$status" in
    pass) PASS=$((PASS + 1)) ;;
    fail) FAIL=$((FAIL + 1)) ;;
    warn) WARN=$((WARN + 1)) ;;
  esac
  if ! $JSON; then
    local icon="[PASS]"
    [ "$status" = "fail" ] && icon="[FAIL]"
    [ "$status" = "warn" ] && icon="[WARN]"
    printf "  %-8s %-35s %s\n" "$icon" "$name" "$detail"
  fi
}

$JSON || echo "FL Platform Health Check"
$JSON || echo "========================"
$JSON || echo ""

# ── Server checks ──────────────────────────────────────────────────
$JSON || echo "Server ($FL_SERVER_HOST):"

# SSH connectivity
if on "$FL_SERVER_HOST" "echo OK" >/dev/null 2>&1; then
  check "server_ssh" "pass" "SSH OK"
else
  check "server_ssh" "fail" "Cannot connect"
fi

# Docker daemon
if on "$FL_SERVER_HOST" "docker version --format '{{.Server.Version}}'" >/dev/null 2>&1; then
  DOCKER_V=$(on "$FL_SERVER_HOST" "docker version --format '{{.Server.Version}}'")
  check "server_docker" "pass" "Docker $DOCKER_V"
else
  check "server_docker" "fail" "Docker not running"
fi

# GPU
GPU_INFO=$(on "$FL_SERVER_HOST" "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader" 2>/dev/null)
if [ -n "$GPU_INFO" ]; then
  check "server_gpu" "pass" "$GPU_INFO"
else
  check "server_gpu" "warn" "GPU not available or driver not loaded"
fi

# Disk space
DISK_PCT=$(on "$FL_SERVER_HOST" "df / --output=pcent | tail -1 | tr -d ' %'")
if [ -n "$DISK_PCT" ]; then
  if [ "$DISK_PCT" -gt 90 ]; then
    check "server_disk" "fail" "${DISK_PCT}% used (>90%)"
  elif [ "$DISK_PCT" -gt 80 ]; then
    check "server_disk" "warn" "${DISK_PCT}% used (>80%)"
  else
    check "server_disk" "pass" "${DISK_PCT}% used"
  fi
fi

# Docker image
IMG_EXISTS=$(on "$FL_SERVER_HOST" "docker images ${FL_IMAGE}:${FL_IMAGE_TAG} --format '{{.ID}}'" 2>/dev/null)
if [ -n "$IMG_EXISTS" ]; then
  IMG_DATE=$(on "$FL_SERVER_HOST" "docker inspect ${FL_IMAGE}:${FL_IMAGE_TAG} --format '{{.Created}}' | cut -c1-19")
  check "server_image" "pass" "${FL_IMAGE}:${FL_IMAGE_TAG} (built $IMG_DATE)"
else
  check "server_image" "fail" "${FL_IMAGE}:${FL_IMAGE_TAG} not found"
fi

# TLS certs (use $HOME on remote, not ~ which expands locally)
REMOTE_CERTS_DIR=$(echo "$FL_CERTS_DIR" | sed "s|^~|/home/${FL_SSH_USER}|")
CERT_EXPIRY=$(on "$FL_SERVER_HOST" "openssl x509 -enddate -noout -in ${REMOTE_CERTS_DIR}/server.pem 2>/dev/null | cut -d= -f2")
if [ -n "$CERT_EXPIRY" ]; then
  DAYS_LEFT=$(on "$FL_SERVER_HOST" "echo \$(( ( \$(date -d '${CERT_EXPIRY}' +%s) - \$(date +%s) ) / 86400 ))" 2>/dev/null || echo "?")
  if [ "$DAYS_LEFT" != "?" ] && [ "$DAYS_LEFT" -lt 30 ]; then
    check "server_tls" "warn" "Expires in ${DAYS_LEFT} days ($CERT_EXPIRY)"
  else
    check "server_tls" "pass" "Expires: $CERT_EXPIRY"
  fi
else
  check "server_tls" "fail" "Certs not found at ${REMOTE_CERTS_DIR}"
fi

# Port 9092
PORT_FREE=$(on "$FL_SERVER_HOST" "ss -tlnp | grep :${FL_GRPC_PORT} | wc -l")
if [ "${PORT_FREE:-0}" = "0" ]; then
  check "server_port" "pass" "Port ${FL_GRPC_PORT} free (ready for training)"
else
  PROC=$(on "$FL_SERVER_HOST" "ss -tlnp | grep :${FL_GRPC_PORT} | head -1")
  check "server_port" "warn" "Port ${FL_GRPC_PORT} in use: $PROC"
fi

# Results directory
REMOTE_RESULTS_DIR=$(echo "$FL_RESULTS_DIR" | sed "s|^~|/home/${FL_SSH_USER}|")
RESULT_COUNT=$(on "$FL_SERVER_HOST" "ls ${REMOTE_RESULTS_DIR}/*.json 2>/dev/null | wc -l")
check "server_results" "pass" "${RESULT_COUNT:-0} result files"

if ! $QUICK; then
  $JSON || echo ""
  $JSON || echo "Clients:"

  # ── Client checks ──────────────────────────────────────────────────
  CLIENT_IDX=0
  for ip in $FL_CLIENT_HOSTS; do
    # SSH
    if on "$ip" "echo OK" >/dev/null 2>&1; then
      check "client_${CLIENT_IDX}_ssh" "pass" "$ip SSH OK"
    else
      check "client_${CLIENT_IDX}_ssh" "fail" "$ip unreachable"
      CLIENT_IDX=$((CLIENT_IDX + 1))
      continue
    fi

    # Docker
    if on "$ip" "sudo docker version" >/dev/null 2>&1; then
      check "client_${CLIENT_IDX}_docker" "pass" "Docker running"
    else
      check "client_${CLIENT_IDX}_docker" "fail" "Docker not running"
    fi

    # GPU
    CLIENT_GPU=$(on "$ip" "nvidia-smi --query-gpu=name,memory.used --format=csv,noheader" 2>/dev/null)
    if [ -n "$CLIENT_GPU" ]; then
      check "client_${CLIENT_IDX}_gpu" "pass" "$CLIENT_GPU"
    else
      check "client_${CLIENT_IDX}_gpu" "warn" "GPU not available"
    fi

    # Image
    CLIENT_IMG=$(on "$ip" "sudo docker images ${FL_IMAGE}:${FL_IMAGE_TAG} --format '{{.ID}}'" 2>/dev/null)
    if [ -n "$CLIENT_IMG" ]; then
      check "client_${CLIENT_IDX}_image" "pass" "Image present"
    else
      check "client_${CLIENT_IDX}_image" "fail" "Image missing — run deploy.sh distribute"
    fi

    # Disk
    CLIENT_DISK=$(on "$ip" "df / --output=pcent | tail -1 | tr -d ' %'" 2>/dev/null)
    if [ -n "$CLIENT_DISK" ] && [ "$CLIENT_DISK" -gt 90 ]; then
      check "client_${CLIENT_IDX}_disk" "fail" "${CLIENT_DISK}% used"
    else
      check "client_${CLIENT_IDX}_disk" "pass" "${CLIENT_DISK:-?}% used"
    fi

    # CA cert
    CA_EXISTS=$(on "$ip" "test -f ${REMOTE_CERTS_DIR}/ca.pem && echo yes" 2>/dev/null)
    if [ "$CA_EXISTS" = "yes" ]; then
      check "client_${CLIENT_IDX}_cert" "pass" "CA cert present"
    else
      check "client_${CLIENT_IDX}_cert" "fail" "CA cert missing"
    fi

    CLIENT_IDX=$((CLIENT_IDX + 1))
  done
fi

# ── Summary ──────────────────────────────────────────────────────────
$JSON || echo ""
$JSON || echo "================================================"

if $JSON; then
  echo "{"
  echo "  \"pass\": $PASS, \"fail\": $FAIL, \"warn\": $WARN,"
  echo "  \"status\": \"$([ $FAIL -gt 0 ] && echo 'critical' || ([ $WARN -gt 0 ] && echo 'degraded' || echo 'healthy'))\","
  echo "  \"checks\": ["
  for i in "${!CHECKS[@]}"; do
    IFS='|' read -r name status detail <<< "${CHECKS[$i]}"
    echo "    {\"name\": \"$name\", \"status\": \"$status\", \"detail\": \"$detail\"}$([ $i -lt $((${#CHECKS[@]}-1)) ] && echo ',')"
  done
  echo "  ]"
  echo "}"
else
  if [ $FAIL -gt 0 ]; then
    echo "CRITICAL: $PASS pass, $FAIL fail, $WARN warn"
    exit 2
  elif [ $WARN -gt 0 ]; then
    echo "DEGRADED: $PASS pass, $FAIL fail, $WARN warn"
    exit 1
  else
    echo "HEALTHY: $PASS pass, $FAIL fail, $WARN warn"
    exit 0
  fi
fi
