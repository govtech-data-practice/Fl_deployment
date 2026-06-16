#!/bin/bash
# Distributed FL Deployment — Production Grade
# =============================================
# Deploys FL platform (models/, tasks/, fl_common/) to EC2 cluster.
# All configuration is sourced from cluster.env (no hardcoded values).
#
# Usage:
#   ./deploy.sh up            — full deploy (build + distribute + start)
#   ./deploy.sh build         — build Docker image on server only
#   ./deploy.sh distribute    — distribute image + certs to clients only
#   ./deploy.sh run [scenario] — run training
#   ./deploy.sh data          — distribute datasets to clients
#   ./deploy.sh status        — check all nodes
#   ./deploy.sh health        — full health check (health_check.sh)
#   ./deploy.sh logs [target] — tail logs
#   ./deploy.sh results [sub] — view/download results
#   ./deploy.sh dashboard     — start Streamlit dashboard
#   ./deploy.sh list          — list scenarios
#   ./deploy.sh down          — stop all containers
#   ./deploy.sh restart       — restart all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

# ── Load cluster.env ──────────────────────────────────────────────────
ENV_FILE="${REPO_ROOT}/cluster.env"
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: cluster.env not found."
  echo "  cp deploy/cluster.env.template cluster.env && vim cluster.env"
  exit 2
fi
source "$ENV_FILE"

# ── Validate required vars ────────────────────────────────────────────
REQUIRED_VARS=(FL_SERVER_HOST FL_SERVER_PRIVATE FL_SSH_KEY FL_SSH_USER
               FL_IMAGE FL_IMAGE_TAG FL_CLIENT_HOSTS FL_NUM_CLIENTS)
for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var:-}" ]; then
    echo "ERROR: $var is not set in cluster.env"
    exit 2
  fi
done

if [ ! -f "$FL_SSH_KEY" ]; then
  echo "ERROR: SSH key not found: $FL_SSH_KEY"
  exit 2
fi

# ── Derived config ────────────────────────────────────────────────────
read -ra CLIENTS <<< "$FL_CLIENT_HOSTS"
NUM_CLIENTS=${#CLIENTS[@]}
FULL_IMAGE="${FL_IMAGE}:${FL_IMAGE_TAG}"

CONTAINER_SL="fl-superlink"
CONTAINER_SN="fl-supernode"
CONTAINER_RUN="fl-training"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR"
SSH_CMD="ssh ${SSH_OPTS} -i ${FL_SSH_KEY}"
SCP_CMD="scp ${SSH_OPTS} -i ${FL_SSH_KEY}"

# Defaults from cluster.env, with fallbacks
GRPC_PORT="${FL_GRPC_PORT:-9092}"
CERTS_DIR="${FL_CERTS_DIR:-~/fl-deploy/certs}"
DATA_DIR="${FL_DATA_DIR:-~/fl-deploy/data}"
RESULTS_DIR="${FL_RESULTS_DIR:-~/fl-deploy/results}"
LOG_MAX_SIZE="${FL_LOG_MAX_SIZE:-200m}"
LOG_MAX_FILE="${FL_LOG_MAX_FILE:-5}"
CERT_DAYS="${FL_CERT_DAYS:-365}"
CERT_CN="${FL_CERT_CN:-fl-server}"

# Resource limits
SERVER_MEMORY="${FL_SERVER_MEMORY:-120g}"
SERVER_CPUS="${FL_SERVER_CPUS:-30}"
SERVER_SHM="${FL_SERVER_SHM_SIZE:-8g}"
CLIENT_MEMORY="${FL_CLIENT_MEMORY:-56g}"
CLIENT_CPUS="${FL_CLIENT_CPUS:-14}"
ORCH_MEMORY="${FL_ORCH_MEMORY:-4g}"
ORCH_CPUS="${FL_ORCH_CPUS:-2}"

# Docker security flags (shared across all containers)
SECURITY_OPTS="--security-opt=no-new-privileges --cap-drop ALL --pids-limit 4096"
LOG_OPTS="--log-opt max-size=${LOG_MAX_SIZE} --log-opt max-file=${LOG_MAX_FILE}"

# ── Helpers ───────────────────────────────────────────────────────────

# SSH to server (no sudo for docker)
on_server() { $SSH_CMD ${FL_SSH_USER}@"${FL_SERVER_HOST}" "$@"; }

# SSH to client (sudo docker)
on_client() { $SSH_CMD ${FL_SSH_USER}@"$1" "${@:2}"; }

# Docker command: server runs without sudo, clients use sudo
docker_server() { on_server "docker $*"; }
docker_client() { local ip="$1"; shift; on_client "$ip" "sudo docker $*"; }

info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
error() { echo "[ERROR] $*" >&2; }

# Cleanup trap — only used in cmd_up to clean temp files
cleanup() {
  rm -f /tmp/fl-build.tar.gz 2>/dev/null || true
}

# Retry a command up to N times with delay
retry() {
  local max_attempts="${1}" delay="${2}" cmd="${@:3}"
  for attempt in $(seq 1 "$max_attempts"); do
    if eval "$cmd"; then
      return 0
    fi
    [ "$attempt" -lt "$max_attempts" ] && warn "Attempt $attempt/$max_attempts failed, retrying in ${delay}s..." && sleep "$delay"
  done
  error "All $max_attempts attempts failed: $cmd"
  return 1
}

# Graceful container stop: SIGTERM first, then SIGKILL after timeout
graceful_stop() {
  local host="$1" container="$2" use_sudo="${3:-no}" timeout="${4:-10}"
  local docker_prefix=""
  [ "$use_sudo" = "sudo" ] && docker_prefix="sudo "
  $SSH_CMD ${FL_SSH_USER}@"$host" "${docker_prefix}docker stop -t $timeout $container 2>/dev/null; ${docker_prefix}docker rm -f $container 2>/dev/null" || true
}

# Verify image SHA matches across nodes
verify_image_sha() {
  local host="$1" use_sudo="${2:-no}"
  local docker_prefix=""
  [ "$use_sudo" = "sudo" ] && docker_prefix="sudo "
  $SSH_CMD ${FL_SSH_USER}@"$host" "${docker_prefix}docker inspect ${FULL_IMAGE} --format '{{.Id}}'" 2>/dev/null | cut -c8-20
}

# Wait for a container to be running, with timeout
wait_container() {
  local host="$1" container="$2" use_sudo="$3" timeout="${4:-30}"
  local docker_prefix=""
  [ "$use_sudo" = "sudo" ] && docker_prefix="sudo "
  local cmd="${docker_prefix}docker inspect -f '{{.State.Running}}' ${container}"

  for i in $(seq 1 "$timeout"); do
    local status
    status=$($SSH_CMD ${FL_SSH_USER}@"$host" "$cmd" 2>/dev/null || echo "false")
    if [ "$status" = "true" ]; then
      return 0
    fi
    sleep 1
  done
  warn "Container $container on $host did not start within ${timeout}s"
  return 1
}

# ── Commands ──────────────────────────────────────────────────────────

cmd_build() {
  info "Building Docker image on server ($FL_SERVER_HOST)"
  trap cleanup EXIT

  cd "$REPO_ROOT"
  info "Packaging codebase..."
  tar czf /tmp/fl-build.tar.gz \
    Dockerfile \
    fl_common/ \
    models/ \
    tasks/ \
    privacy/ \
    experiments/ \
    scenarios/ \
    secure_inference/ \
    run_tests.py run_all.py run_ec2.py run_client.py ingest.py \
    2>/dev/null

  info "Uploading to server..."
  $SCP_CMD /tmp/fl-build.tar.gz ${FL_SSH_USER}@${FL_SERVER_HOST}:/tmp/

  local BUILD_TAG
  BUILD_TAG=$(date +%Y%m%d_%H%M%S)

  on_server "
    rm -rf ~/fl-build && mkdir -p ~/fl-build && cd ~/fl-build
    tar xzf /tmp/fl-build.tar.gz 2>/dev/null
    echo 'Building Docker image...'
    docker build -t ${FL_IMAGE}:${FL_IMAGE_TAG} . 2>&1 | tail -5
    docker tag ${FL_IMAGE}:${FL_IMAGE_TAG} ${FL_IMAGE}:${BUILD_TAG}
    docker save ${FL_IMAGE}:${FL_IMAGE_TAG} | gzip > /tmp/fl-image.tar.gz
    echo \"Image tagged: ${FL_IMAGE}:${FL_IMAGE_TAG} + ${FL_IMAGE}:${BUILD_TAG}\"
    echo \"Image size: \$(du -h /tmp/fl-image.tar.gz | cut -f1)\"
  "

  # Health check: verify image exists on server
  local img_id
  img_id=$(docker_server "images ${FULL_IMAGE} --format '{{.ID}}'" 2>/dev/null)
  if [ -z "$img_id" ]; then
    error "Build failed: image ${FULL_IMAGE} not found on server"
    return 1
  fi
  info "Build verified: ${FULL_IMAGE} ($img_id)"
}

cmd_distribute() {
  local LOCAL_CERTS="$SCRIPT_DIR/certs"

  if [ ! -f "$LOCAL_CERTS/ca.pem" ]; then
    error "No certs found at $LOCAL_CERTS. Run './deploy.sh up' first or generate certs."
    return 1
  fi

  info "Distributing image + certs to ${NUM_CLIENTS} clients"

  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    local ip="${CLIENTS[$i]}"
    info "  Client $i ($ip)..."
    (
      # Ship image via server's internal network (with retry)
      retry 3 10 "on_server 'scp -i ~/.ssh/\$(basename ${FL_SSH_KEY}) -o StrictHostKeyChecking=no /tmp/fl-image.tar.gz ${FL_SSH_USER}@${ip}:/tmp/' 2>/dev/null"

      # Ship CA cert from local
      $SCP_CMD "$LOCAL_CERTS/ca.pem" ${FL_SSH_USER}@${ip}:/tmp/ca.pem

      on_client "$ip" "
        if ! sudo docker version &>/dev/null; then
          sudo dnf install -y docker 2>&1 | tail -1
          sudo systemctl enable --now docker
        fi
        sudo docker load < /tmp/fl-image.tar.gz 2>&1 | tail -1
        mkdir -p ~/fl-deploy/{certs,data,logs}
        cp /tmp/ca.pem ~/fl-deploy/certs/
      "

      # Health check: verify image loaded
      local img_check
      img_check=$(docker_client "$ip" "images ${FULL_IMAGE} --format '{{.ID}}'" 2>/dev/null)
      if [ -z "$img_check" ]; then
        warn "Client $i ($ip): image not found after load"
      else
        info "  Client $i ($ip): ready ($img_check)"
      fi
    ) &
  done
  wait
  info "Distribution complete"
}

cmd_up() {
  echo "============================================"
  echo "DISTRIBUTED FL — PRODUCTION DEPLOYMENT"
  echo "  Server:  $FL_SERVER_HOST ($FL_SERVER_PRIVATE)"
  echo "  Clients: ${FL_CLIENT_HOSTS} ($NUM_CLIENTS nodes)"
  echo "  Image:   ${FULL_IMAGE}"
  echo "============================================"
  trap cleanup EXIT

  # --- 1. TLS Certs ---
  echo ""
  info "1. TLS Certificates"
  local LOCAL_CERTS="$SCRIPT_DIR/certs"
  mkdir -p "$LOCAL_CERTS"
  if [ ! -f "$LOCAL_CERTS/server.pem" ]; then
    openssl ecparam -genkey -name prime256v1 -out "$LOCAL_CERTS/ca.key" 2>/dev/null
    openssl req -new -x509 -key "$LOCAL_CERTS/ca.key" -out "$LOCAL_CERTS/ca.pem" \
      -days "$CERT_DAYS" -subj "/CN=${CERT_CN}-CA" 2>/dev/null
    openssl ecparam -genkey -name prime256v1 -out "$LOCAL_CERTS/server.key" 2>/dev/null
    cat > "$LOCAL_CERTS/san.cnf" << SANEOF
[req]
distinguished_name=dn
req_extensions=v3
prompt=no
[dn]
CN=${CERT_CN}
[v3]
subjectAltName=DNS:${CERT_CN},DNS:localhost,IP:127.0.0.1,IP:${FL_SERVER_PRIVATE},IP:${FL_SERVER_HOST}
SANEOF
    openssl req -new -key "$LOCAL_CERTS/server.key" -out "$LOCAL_CERTS/s.csr" -config "$LOCAL_CERTS/san.cnf" 2>/dev/null
    openssl x509 -req -in "$LOCAL_CERTS/s.csr" -CA "$LOCAL_CERTS/ca.pem" -CAkey "$LOCAL_CERTS/ca.key" \
      -CAcreateserial -out "$LOCAL_CERTS/server.pem" -days "$CERT_DAYS" \
      -extfile "$LOCAL_CERTS/san.cnf" -extensions v3 2>/dev/null
    rm -f "$LOCAL_CERTS"/{s.csr,san.cnf,ca.key,ca.srl}
    info "  Generated (SAN: ${FL_SERVER_PRIVATE}, ${FL_SERVER_HOST})"
  else
    info "  Certs exist, skipping"
  fi

  # --- 2. Build image on server ---
  echo ""
  info "2. Build Docker image on server"
  cmd_build

  # --- 3. Distribute to clients ---
  echo ""
  info "3. Distribute image + certs to clients"
  cmd_distribute

  # --- 4. Start SuperLink ---
  echo ""
  info "4. Start SuperLink"
  on_server "mkdir -p ~/fl-deploy/certs"
  $SCP_CMD "$LOCAL_CERTS/server.pem" "$LOCAL_CERTS/server.key" "$LOCAL_CERTS/ca.pem" \
    ${FL_SSH_USER}@${FL_SERVER_HOST}:~/fl-deploy/certs/
  on_server "chmod 600 ~/fl-deploy/certs/server.key"

  on_server "
    docker rm -f ${CONTAINER_SL} 2>/dev/null
    docker run -d \
      --name ${CONTAINER_SL} \
      --restart unless-stopped \
      --network host \

      ${SECURITY_OPTS} \
      --memory ${SERVER_MEMORY} \
      --cpus ${SERVER_CPUS} \
      -v ~/fl-deploy/certs:/certs:ro \
      --tmpfs /tmp:rw,noexec,nosuid,size=512m \
      ${LOG_OPTS} \
      ${FULL_IMAGE} \
      flower-superlink \
        --ssl-certfile /certs/server.pem \
        --ssl-keyfile /certs/server.key \
        --ssl-ca-certfile /certs/ca.pem
  "

  if wait_container "$FL_SERVER_HOST" "$CONTAINER_SL" "no" 15; then
    info "SuperLink started"
  else
    error "SuperLink failed to start"
    docker_server "logs ${CONTAINER_SL} --tail 20" 2>/dev/null
    return 1
  fi

  # --- 5. Start SuperNodes ---
  echo ""
  info "5. Start SuperNodes"
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    local ip="${CLIENTS[$i]}"
    echo -n "  Client $i ($ip): "
    on_client "$ip" "
      sudo docker rm -f ${CONTAINER_SN} 2>/dev/null
      sudo docker run -d \
        --name ${CONTAINER_SN} \
        --restart unless-stopped \
        --network host \
  
        ${SECURITY_OPTS} \
        --memory ${CLIENT_MEMORY} \
        --cpus ${CLIENT_CPUS} \
        -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro \
        -v ~/fl-deploy/data:/data:ro \
        -v ~/fl-deploy/logs:/app/logs \
        --tmpfs /tmp:rw,noexec,nosuid,size=1g \
        -e PYTHONUNBUFFERED=1 \
        ${LOG_OPTS} \
        ${FULL_IMAGE} \
        flower-supernode \
          --root-certificates /certs/ca.pem \
          --superlink ${FL_SERVER_PRIVATE}:${GRPC_PORT} \
          --node-config 'partition-id=${i} num-clients=${NUM_CLIENTS}' \
          --clientappio-api-address 0.0.0.0:7070
    " &
  done
  wait

  # Health check: verify all supernodes started
  local ok=0
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    if wait_container "${CLIENTS[$i]}" "$CONTAINER_SN" "sudo" 15; then
      ok=$((ok + 1))
    fi
  done
  info "SuperNodes started: $ok/$NUM_CLIENTS"

  sleep 5
  echo ""
  cmd_status
}
# ── Run production training (simulation mode on server) ───────────────

cmd_run() {
  local SCENARIO="${1:-all}"
  echo "============================================"
  echo "FL + PET SANDBOX — RUN"
  echo "  Scenario: $SCENARIO"
  echo "  Running on server (simulation mode)"
  echo "============================================"

  # Real data paths on EC2
  local DATA_SEPSIS="/home/${FL_SSH_USER}/healthcare-fl/data/sepsis"
  local DATA_CHEST="/home/${FL_SSH_USER}/healthcare-fl/data/chest_xray_real/archive"
  local DATA_CLINICAL="/home/${FL_SSH_USER}/healthcare-fl/data/clinical"

  # Map scenario name to path if not a full path
  local RUN_ARG
  if echo "$SCENARIO" | grep -qE '\.(yaml|yml)$'; then
    RUN_ARG="$SCENARIO"
  elif [ -f "scenarios/${SCENARIO}.yaml" ]; then
    RUN_ARG="scenarios/${SCENARIO}.yaml"
  else
    RUN_ARG="$SCENARIO"
  fi

  on_server "
    docker rm -f ${CONTAINER_RUN} 2>/dev/null
    mkdir -p ~/fl-deploy/results
    docker run -d \
      --name ${CONTAINER_RUN} \
      --gpus all \
      --shm-size=${SERVER_SHM} \
      --network host \
      ${SECURITY_OPTS} \
      --memory ${SERVER_MEMORY} \
      --cpus ${SERVER_CPUS} \
      -v ${DATA_SEPSIS}:/data/sepsis:ro \
      -v ${DATA_CHEST}:/data/chest_xray:ro \
      -v ${DATA_CLINICAL}:/data/clinical:ro \
      -v ~/fl-deploy/results:/app/results \
      --tmpfs /tmp:rw,noexec,nosuid,size=2g \
      -e DATA_PATH=/data/sepsis \
      -e DATASET_PATH=/data/chest_xray \
      -e CSV_PATH=/data/chest_xray/Data_Entry_2017.csv \
      -e PYTHONUNBUFFERED=1 \
      -e SYNTHETIC=0 \
      ${LOG_OPTS} \
      ${FULL_IMAGE} \
      python run_ec2.py ${RUN_ARG}
  "

  if wait_container "$FL_SERVER_HOST" "$CONTAINER_RUN" "no" 15; then
    info "Training started"
    echo "  Monitor: ./deploy.sh logs training"
    echo "  Results: ./deploy.sh results"
  else
    error "Training container failed to start"
    docker_server "logs ${CONTAINER_RUN} --tail 20" 2>/dev/null
    return 1
  fi
}
# ── Download results ──────────────────────────────────────────────────

cmd_results() {
  local TARGET="${1:-download}"

  if [ "$TARGET" = "summary" ]; then
    echo "=== Latest Results Summary ==="
    on_server "
      LATEST=\$(ls -t ~/fl-deploy/results/*.md 2>/dev/null | head -1)
      if [ -n \"\$LATEST\" ]; then
        cat \"\$LATEST\"
      else
        echo 'No results yet. Run a scenario first: ./deploy.sh run quick_fraud'
      fi
    " 2>/dev/null
  elif [ "$TARGET" = "list" ]; then
    echo "=== Available Results ==="
    on_server "ls -lt ~/fl-deploy/results/ 2>/dev/null | head -20" 2>/dev/null
  else
    info "Downloading results from server..."
    mkdir -p "$REPO_ROOT/results"
    $SCP_CMD "${FL_SSH_USER}@${FL_SERVER_HOST}:~/fl-deploy/results/*" "$REPO_ROOT/results/" 2>/dev/null
    info "Results saved to $REPO_ROOT/results/"
    ls -lt "$REPO_ROOT/results/" | head -10
  fi
}
# ── Streamlit dashboard ──────────────────────────────────────────────

cmd_dashboard() {
  echo "============================================"
  echo "FL + PET SANDBOX — DASHBOARD"
  echo "============================================"

  on_server "
    docker rm -f fl-dashboard 2>/dev/null
    docker run -d \
      --name fl-dashboard \
      --gpus all \
      --network host \

      ${SECURITY_OPTS} \
      --memory ${ORCH_MEMORY} \
      --cpus ${ORCH_CPUS} \
      -v ~/fl-deploy/results:/app/results:ro \
      --tmpfs /tmp:rw,noexec,nosuid,size=512m \
      -p 8501:8501 \
      ${LOG_OPTS} \
      ${FULL_IMAGE} \
      streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
  "

  if wait_container "$FL_SERVER_HOST" "fl-dashboard" "no" 15; then
    info "Dashboard running on server:8501"
  else
    error "Dashboard failed to start"
    docker_server "logs fl-dashboard --tail 10" 2>/dev/null
    return 1
  fi

  echo ""
  echo "To access, run this SSH tunnel:"
  echo "  ssh -L 8501:localhost:8501 -i ${FL_SSH_KEY} ${FL_SSH_USER}@${FL_SERVER_HOST}"
  echo ""
  echo "Then open: http://localhost:8501"
}
# ── List available scenarios ─────────────────────────────────────────

cmd_list() {
  echo "============================================"
  echo "FL + PET SANDBOX — AVAILABLE SCENARIOS"
  echo "============================================"
  echo ""
  echo "Quick demos (< 1 min):"
  echo "  ./deploy.sh run quick_fraud      — FedAvg on synthetic fraud"
  echo "  ./deploy.sh run quick_sepsis     — FedAvg on synthetic sepsis"
  echo ""
  echo "Comparisons (5-15 min):"
  echo "  ./deploy.sh run strategy_showdown — FedAvg vs FedProx vs SCAFFOLD vs SecAgg"
  echo "  ./deploy.sh run noniid_impact    — Non-IID severity sweep (alpha 1.0->0.1)"
  echo "  ./deploy.sh run dp_tradeoff      — DP epsilon sweep (50->1)"
  echo ""
  echo "Privacy (5 min):"
  echo "  ./deploy.sh run privacy_attacks  — DLG + MIA with/without DP"
  echo ""
  echo "Full benchmarks (hours):"
  echo "  ./deploy.sh run full_benchmark   — All 4 tasks x key strategies (~2-3h)"
  echo "  ./deploy.sh run chest_production — Real NIH chest X-ray, 7 strategies (~8h)"
  echo "  ./deploy.sh run all              — Everything (~10-12h)"
  echo ""
  echo "Built-in tasks:"
  echo "  ./deploy.sh run sepsis|ecg|fraud|chest|privacy"
  echo ""
  echo "Commands:"
  echo "  ./deploy.sh up                   — Full deploy (build + distribute + start)"
  echo "  ./deploy.sh build                — Build Docker image on server"
  echo "  ./deploy.sh distribute           — Distribute image + certs to clients"
  echo "  ./deploy.sh run [scenario]       — Run training"
  echo "  ./deploy.sh data                 — Distribute datasets to clients"
  echo "  ./deploy.sh status               — Cluster status"
  echo "  ./deploy.sh health               — Full health check"
  echo "  ./deploy.sh logs [target]        — Tail logs (superlink/training/all/N)"
  echo "  ./deploy.sh results [sub]        — View/download results"
  echo "  ./deploy.sh dashboard            — Start Streamlit dashboard"
  echo "  ./deploy.sh list                 — This help"
  echo "  ./deploy.sh down                 — Stop all containers"
  echo "  ./deploy.sh restart              — Restart all"
}
cmd_status() {
  echo "============================================"
  echo "CLUSTER STATUS"
  echo "============================================"

  echo ""
  echo "Server ($FL_SERVER_HOST):"
  on_server "
    docker ps --format '  {{.Names}}: {{.Status}}' 2>/dev/null
    echo '  GPU:'
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    echo '  Disk:'
    df -h / | tail -1 | awk '{print \"    Used: \" \$3 \" / \" \$2 \" (\" \$5 \")\"}'
  " 2>/dev/null

  # Count connected nodes
  local NODES
  NODES=$(on_server "docker logs ${CONTAINER_SL} 2>&1 | grep -oP 'node_id=\d+' | sort -u | wc -l" 2>/dev/null || echo "0")
  echo "  Connected nodes: $NODES/$NUM_CLIENTS"

  echo ""
  echo "SuperNodes:"
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    local ip="${CLIENTS[$i]}"
    local status
    status=$(docker_client "$ip" "ps --filter name=${CONTAINER_SN} --format '{{.Status}}'" 2>/dev/null || echo "unreachable")
    echo "  Client $i ($ip): $status"
  done

  echo ""
  echo "Training container:"
  docker_server "ps --filter name=${CONTAINER_RUN} --format '  {{.Names}}: {{.Status}}'" 2>/dev/null || echo "  Not running"

  echo ""
  echo "Recent results:"
  on_server "ls -lt ~/fl-deploy/results/*.json 2>/dev/null | head -3 | awk '{print \"  \" \$NF}'" 2>/dev/null || echo "  None"
}

cmd_health() {
  local HEALTH_SCRIPT="${DEPLOY_DIR}/health_check.sh"
  if [ ! -f "$HEALTH_SCRIPT" ]; then
    error "health_check.sh not found at $HEALTH_SCRIPT"
    return 1
  fi
  bash "$HEALTH_SCRIPT" "$@"
}

cmd_logs() {
  local TARGET="${1:-superlink}"
  if [ "$TARGET" = "superlink" ]; then
    on_server "docker logs ${CONTAINER_SL} --tail 100 -f" 2>/dev/null
  elif [ "$TARGET" = "training" ]; then
    on_server "docker logs ${CONTAINER_RUN} --tail 200 -f" 2>/dev/null
  elif [ "$TARGET" = "results" ]; then
    on_server "cat \$(ls -t ~/fl-deploy/results/*.json 2>/dev/null | head -1)" 2>/dev/null
  elif [ "$TARGET" = "all" ]; then
    echo "=== SuperLink ==="
    on_server "docker logs ${CONTAINER_SL} --tail 10 2>&1" 2>/dev/null
    echo ""
    echo "=== Training ==="
    on_server "docker logs ${CONTAINER_RUN} --tail 20 2>&1" 2>/dev/null
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
      echo ""
      echo "=== Client $i (${CLIENTS[$i]}) ==="
      docker_client "${CLIENTS[$i]}" "logs ${CONTAINER_SN} --tail 10 2>&1" 2>/dev/null
    done
  else
    docker_client "${CLIENTS[$TARGET]}" "logs ${CONTAINER_SN} --tail 100 -f" 2>/dev/null
  fi
}

cmd_down() {
  info "Stopping all containers (graceful: SIGTERM + 10s wait)..."
  graceful_stop "$FL_SERVER_HOST" "${CONTAINER_SL}" "no" 10 &
  graceful_stop "$FL_SERVER_HOST" "${CONTAINER_RUN}" "no" 10 &
  graceful_stop "$FL_SERVER_HOST" "fl-dashboard" "no" 5 &
  graceful_stop "$FL_SERVER_HOST" "fl-orchestrator" "no" 10 &
  for ip in "${CLIENTS[@]}"; do
    graceful_stop "$ip" "${CONTAINER_SN}" "sudo" 10 &
    graceful_stop "$ip" "fl-client" "sudo" 10 &
  done
  wait
  info "All stopped."
}

cmd_restart() {
  cmd_down
  sleep 3

  info "Starting SuperLink..."
  on_server "
    docker run -d --name ${CONTAINER_SL} --restart unless-stopped --network host \
      ${SECURITY_OPTS} \
      --memory ${SERVER_MEMORY} --cpus ${SERVER_CPUS} \
      -v ~/fl-deploy/certs:/certs:ro \
      --tmpfs /tmp:rw,noexec,nosuid,size=512m \
      ${LOG_OPTS} \
      ${FULL_IMAGE} \
      flower-superlink --ssl-certfile /certs/server.pem --ssl-keyfile /certs/server.key --ssl-ca-certfile /certs/ca.pem
  " 2>/dev/null

  if ! wait_container "$FL_SERVER_HOST" "$CONTAINER_SL" "no" 15; then
    error "SuperLink failed to start"
    return 1
  fi

  info "Starting clients..."
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    on_client "${CLIENTS[$i]}" "
      sudo docker run -d --name ${CONTAINER_SN} --restart unless-stopped --network host \
        ${SECURITY_OPTS} \
        --memory ${CLIENT_MEMORY} --cpus ${CLIENT_CPUS} \
        -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro -v ~/fl-deploy/data:/data:ro \
        -v ~/fl-deploy/logs:/app/logs \
        --tmpfs /tmp:rw,noexec,nosuid,size=1g \
        -e PYTHONUNBUFFERED=1 \
        ${LOG_OPTS} \
        ${FULL_IMAGE} \
        flower-supernode --root-certificates /certs/ca.pem --superlink ${FL_SERVER_PRIVATE}:${GRPC_PORT} \
          --node-config 'partition-id=${i} num-clients=${NUM_CLIENTS}' \
          --clientappio-api-address 0.0.0.0:7070
    " 2>/dev/null &
  done
  wait

  sleep 5
  cmd_status
}
# ── Data distribution ────────────────────────────────────────────────

cmd_data() {
  echo "============================================"
  echo "DATA DISTRIBUTION"
  echo "============================================"

  local DATA_ROOT="/home/${FL_SSH_USER}/healthcare-fl/data"

  echo ""
  echo ">>> Sepsis data (eICU NPZ files)"
  on_server "ls ${DATA_ROOT}/sepsis/*.npz 2>/dev/null | wc -l | xargs echo 'Server has' && echo 'files'" 2>/dev/null

  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    local ip="${CLIENTS[$i]}"
    echo "  Client $i ($ip):"
    on_client "$ip" "mkdir -p ~/fl-deploy/data/flower_data" 2>/dev/null
    # Copy from server via internal network
    on_server "scp -o StrictHostKeyChecking=no -r ${DATA_ROOT}/sepsis/*.npz ${FL_SSH_USER}@${ip}:~/fl-deploy/data/flower_data/" 2>/dev/null &
  done
  wait
  echo "  Sepsis data distributed."

  echo ""
  echo ">>> Clinical data (LLM JSON)"
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    local ip="${CLIENTS[$i]}"
    on_client "$ip" "mkdir -p ~/fl-deploy/data/clinical" 2>/dev/null
    on_server "scp -o StrictHostKeyChecking=no ${DATA_ROOT}/clinical/*.json ${FL_SSH_USER}@${ip}:~/fl-deploy/data/clinical/" 2>/dev/null &
  done
  wait
  echo "  Clinical data distributed."

  echo ""
  echo ">>> Chest X-ray (43GB — too large for client distribution)"
  echo "  Chest X-ray uses synthetic data on clients by default."
  echo "  For real data: copy manually or use NFS/S3."
  echo "  Server has real data at: ${DATA_ROOT}/chest_xray_real/"

  echo ""
  echo ">>> Synthetic tasks (no data transfer needed)"
  echo "  fraud, ecg, anomaly, mortality, drug, satellite, readmission"
  echo "  These generate data at runtime."

  echo ""
  info "Data distribution complete."
}

# ── Command dispatch ─────────────────────────────────────────────────

[ $# -eq 0 ] && { cmd_list; exit 0; }
case "$1" in
  up)         cmd_up ;;
  build)      cmd_build ;;
  distribute) cmd_distribute ;;
  run)        cmd_run "${2:-all}" ;;
  data)       cmd_data ;;
  status)     cmd_status ;;
  health)     shift; cmd_health "$@" ;;
  logs)       cmd_logs "${2:-superlink}" ;;
  results)    cmd_results "${2:-download}" ;;
  dashboard)  cmd_dashboard ;;
  list)       cmd_list ;;
  down)       cmd_down ;;
  restart)    cmd_restart ;;
  *)          error "Unknown command: $1"; cmd_list; exit 1 ;;
esac
