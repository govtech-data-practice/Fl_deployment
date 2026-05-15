#!/bin/bash
# Distributed FL Deployment — Docker containers, production scale
# ================================================================
# Deploys reorganised codebase (models/, tasks/, fl_common/) to EC2.
#
# Usage:
#   ./deploy.sh up        — full deploy (build + distribute + start)
#   ./deploy.sh run       — run production training (run_ec2.py)
#   ./deploy.sh run chest — run single task
#   ./deploy.sh status    — check all nodes
#   ./deploy.sh logs [N]  — tail logs (superlink / client N / all)
#   ./deploy.sh down      — stop all containers
#   ./deploy.sh restart   — restart all

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
KEY="$REPO_ROOT/TEE_FL.pem"

SUPERLINK="54.151.221.104"
SUPERLINK_PRIVATE="172.31.4.42"
CLIENTS=(47.130.0.207 47.129.54.224 52.221.246.101 175.41.152.74 3.0.16.188)
NUM_CLIENTS=${#CLIENTS[@]}
IMAGE="healthcare-fl:latest"
CONTAINER_SL="fl-superlink"
CONTAINER_SN="fl-supernode"
CONTAINER_RUN="fl-training"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $KEY"
SCP="scp -o StrictHostKeyChecking=no -i $KEY"

on() { $SSH ec2-user@"$1" "${@:2}"; }

# ============================================================

cmd_up() {
  echo "============================================"
  echo "DISTRIBUTED FL — PRODUCTION DEPLOYMENT"
  echo "  SuperLink: $SUPERLINK ($SUPERLINK_PRIVATE)"
  echo "  Clients:   ${CLIENTS[*]} ($NUM_CLIENTS nodes)"
  echo "  Image:     $IMAGE"
  echo "  Codebase:  models/ tasks/ fl_common/ (reorganised)"
  echo "============================================"

  # --- 1. TLS Certs ---
  echo ""
  echo ">>> 1. TLS Certificates"
  CERTS="$SCRIPT_DIR/certs"
  mkdir -p "$CERTS"
  if [ ! -f "$CERTS/server.pem" ]; then
    openssl ecparam -genkey -name prime256v1 -out "$CERTS/ca.key" 2>/dev/null
    openssl req -new -x509 -key "$CERTS/ca.key" -out "$CERTS/ca.pem" \
      -days 365 -subj "/CN=Healthcare-FL-CA" 2>/dev/null
    openssl ecparam -genkey -name prime256v1 -out "$CERTS/server.key" 2>/dev/null
    cat > "$CERTS/san.cnf" << EOF
[req]
distinguished_name=dn
req_extensions=v3
prompt=no
[dn]
CN=superlink
[v3]
subjectAltName=DNS:superlink,DNS:localhost,IP:127.0.0.1,IP:$SUPERLINK_PRIVATE,IP:$SUPERLINK
EOF
    openssl req -new -key "$CERTS/server.key" -out "$CERTS/s.csr" -config "$CERTS/san.cnf" 2>/dev/null
    openssl x509 -req -in "$CERTS/s.csr" -CA "$CERTS/ca.pem" -CAkey "$CERTS/ca.key" \
      -CAcreateserial -out "$CERTS/server.pem" -days 365 \
      -extfile "$CERTS/san.cnf" -extensions v3 2>/dev/null
    rm -f "$CERTS"/{s.csr,san.cnf,ca.key,ca.srl}
    echo "  Generated (SAN: $SUPERLINK_PRIVATE, $SUPERLINK)"
  else
    echo "  Exists, skipping"
  fi

  # --- 2. Build image on SuperLink ---
  echo ""
  echo ">>> 2. Build Docker image on SuperLink"
  cd "$REPO_ROOT"
  echo "  Packaging reorganised codebase..."
  tar czf /tmp/fl-build.tar.gz \
    Dockerfile \
    fl_common/ \
    models/bilstm/server_app.py models/bilstm/client_app.py \
    models/densenet/server_app.py models/densenet/client_app.py \
    models/mlp/server_app.py models/mlp/client_app.py \
    models/__init__.py \
    tasks/__init__.py \
    tasks/sepsis/data.py tasks/sepsis/__init__.py \
    tasks/ecg/data.py tasks/ecg/__init__.py \
    tasks/fraud/data.py tasks/fraud/__init__.py \
    tasks/chest_xray/data.py tasks/chest_xray/__init__.py \
    tasks/gov_llm/data.py tasks/gov_llm/__init__.py \
    privacy/test_privacy.py privacy/attack_suite.py privacy/sweep_dp.py \
    experiments/validate_strategies.py \
    run_tests.py run_all.py run_ec2.py \
    2>/dev/null

  echo "  Uploading to SuperLink..."
  $SCP /tmp/fl-build.tar.gz ec2-user@$SUPERLINK:/tmp/

  on $SUPERLINK '
    rm -rf ~/fl-build && mkdir -p ~/fl-build && cd ~/fl-build
    tar xzf /tmp/fl-build.tar.gz 2>/dev/null
    echo "  Building Docker image (this may take a few minutes)..."
    sudo docker build -t healthcare-fl:latest . 2>&1 | tail -5
    sudo docker save healthcare-fl:latest | gzip > /tmp/fl-image.tar.gz
    echo "  Image size: $(du -h /tmp/fl-image.tar.gz | cut -f1)"
  '

  # --- 3. Distribute to clients ---
  echo ""
  echo ">>> 3. Distribute image + certs to clients"

  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    ip="${CLIENTS[$i]}"
    echo "  Client $i ($ip)..."
    (
      # Ship image via SuperLink internal network (fast)
      on $SUPERLINK "scp -o StrictHostKeyChecking=no /tmp/fl-image.tar.gz ec2-user@${ip}:/tmp/" 2>/dev/null

      # Ship certs from local
      $SCP "$CERTS/ca.pem" ec2-user@$ip:/tmp/ca.pem

      on $ip "
        # Docker
        if ! sudo docker version &>/dev/null; then
          sudo dnf install -y docker 2>&1 | tail -1
          sudo systemctl enable --now docker
        fi
        # Load image
        sudo docker load < /tmp/fl-image.tar.gz 2>&1 | tail -1
        # Setup dirs
        mkdir -p ~/fl-deploy/{certs,data,logs}
        cp /tmp/ca.pem ~/fl-deploy/certs/
        echo '  Client $i: ready'
      "
    ) &
  done
  wait

  # --- 4. Start SuperLink ---
  echo ""
  echo ">>> 4. Start SuperLink"
  on $SUPERLINK 'mkdir -p ~/fl-deploy/certs'
  $SCP "$CERTS/server.pem" "$CERTS/server.key" "$CERTS/ca.pem" ec2-user@$SUPERLINK:~/fl-deploy/certs/
  on $SUPERLINK 'chmod 600 ~/fl-deploy/certs/server.key'

  on $SUPERLINK "
    sudo docker rm -f $CONTAINER_SL 2>/dev/null
    sudo docker run -d \
      --name $CONTAINER_SL \
      --restart unless-stopped \
      --network host \
      -v ~/fl-deploy/certs:/certs:ro \
      --log-opt max-size=100m --log-opt max-file=10 \
      $IMAGE \
      flower-superlink \
        --ssl-certfile /certs/server.pem \
        --ssl-keyfile /certs/server.key \
        --ssl-ca-certfile /certs/ca.pem
    sleep 3
    sudo docker ps --filter name=$CONTAINER_SL --format 'SuperLink: {{.Status}}'
  "

  # --- 5. Start SuperNodes ---
  echo ""
  echo ">>> 5. Start SuperNodes"
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    ip="${CLIENTS[$i]}"
    echo -n "  Client $i ($ip): "
    on $ip "
      sudo docker rm -f $CONTAINER_SN 2>/dev/null
      sudo docker run -d \
        --name $CONTAINER_SN \
        --restart unless-stopped \
        --network host \
        -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro \
        -v ~/fl-deploy/data:/data:ro \
        -v ~/fl-deploy/logs:/app/logs \
        -e PYTHONUNBUFFERED=1 \
        --log-opt max-size=100m --log-opt max-file=10 \
        $IMAGE \
        flower-supernode \
          --root-certificates /certs/ca.pem \
          --superlink $SUPERLINK_PRIVATE:9092 \
          --node-config 'partition-id=$i num-clients=$NUM_CLIENTS' \
          --clientappio-api-address 0.0.0.0:7070
      sleep 2
      sudo docker ps --filter name=$CONTAINER_SN --format '{{.Status}}'
    " &
  done
  wait

  sleep 10
  echo ""
  cmd_status
}


# ── Run production training (simulation mode on SuperLink) ───────────

cmd_run() {
  SCENARIO="${1:-all}"
  echo "============================================"
  echo "FL + PET SANDBOX — RUN"
  echo "  Scenario: $SCENARIO"
  echo "  Running on SuperLink (L4 GPU, simulation mode)"
  echo "============================================"

  # Real data paths on EC2
  DATA_SEPSIS="/home/ec2-user/healthcare-fl/data/sepsis"
  DATA_CHEST="/home/ec2-user/healthcare-fl/data/chest_xray_real/archive"
  DATA_CLINICAL="/home/ec2-user/healthcare-fl/data/clinical"

  # Map scenario name to path if not a full path
  if echo "$SCENARIO" | grep -qE '\.(yaml|yml)$'; then
    RUN_ARG="$SCENARIO"
  elif [ -f "scenarios/${SCENARIO}.yaml" ]; then
    RUN_ARG="scenarios/${SCENARIO}.yaml"
  else
    RUN_ARG="$SCENARIO"
  fi

  on $SUPERLINK "
    sudo docker rm -f $CONTAINER_RUN 2>/dev/null
    mkdir -p ~/fl-deploy/results
    sudo docker run -d \
      --name $CONTAINER_RUN \
      --gpus all \
      --shm-size=8g \
      --network host \
      -v $DATA_SEPSIS:/data/sepsis:ro \
      -v $DATA_CHEST:/data/chest_xray:ro \
      -v $DATA_CLINICAL:/data/clinical:ro \
      -v ~/fl-deploy/results:/app/results \
      -e DATA_PATH=/data/sepsis \
      -e DATASET_PATH=/data/chest_xray \
      -e CSV_PATH=/data/chest_xray/Data_Entry_2017.csv \
      -e PYTHONUNBUFFERED=1 \
      -e SYNTHETIC=0 \
      --log-opt max-size=200m --log-opt max-file=10 \
      $IMAGE \
      python run_ec2.py $RUN_ARG
    echo ''
    echo 'Training started.'
    echo 'Monitor: ./deploy.sh logs training'
    echo 'Results: ./deploy.sh results'
  "
}


# ── Download results ─────────────────────────────────────────────────

cmd_results() {
  TARGET="${1:-download}"

  if [ "$TARGET" = "summary" ]; then
    echo "=== Latest Results Summary ==="
    on $SUPERLINK "
      LATEST=\$(ls -t ~/fl-deploy/results/*.md 2>/dev/null | head -1)
      if [ -n \"\$LATEST\" ]; then
        cat \"\$LATEST\"
      else
        echo 'No results yet. Run a scenario first: ./deploy.sh run quick_fraud'
      fi
    " 2>/dev/null
  elif [ "$TARGET" = "list" ]; then
    echo "=== Available Results ==="
    on $SUPERLINK "ls -lt ~/fl-deploy/results/ 2>/dev/null | head -20" 2>/dev/null
  else
    echo "Downloading results from EC2..."
    mkdir -p "$REPO_ROOT/results"
    $SCP "ec2-user@$SUPERLINK:~/fl-deploy/results/*" "$REPO_ROOT/results/" 2>/dev/null
    echo "Results saved to $REPO_ROOT/results/"
    ls -lt "$REPO_ROOT/results/" | head -10
  fi
}


# ── Streamlit dashboard ─────────────────────────────────────────────

cmd_dashboard() {
  echo "============================================"
  echo "FL + PET SANDBOX — DASHBOARD"
  echo "============================================"

  on $SUPERLINK "
    sudo docker rm -f fl-dashboard 2>/dev/null
    sudo docker run -d \
      --name fl-dashboard \
      --gpus all \
      --network host \
      -v ~/fl-deploy/results:/app/results \
      -p 8501:8501 \
      --log-opt max-size=50m \
      $IMAGE \
      streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
    sleep 3
    sudo docker ps --filter name=fl-dashboard --format 'Dashboard: {{.Status}}'
  "

  echo ""
  echo "Dashboard running on EC2:8501"
  echo ""
  echo "To access, run this SSH tunnel:"
  echo "  ssh -L 8501:localhost:8501 -i $KEY ec2-user@$SUPERLINK"
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
  echo "  ./deploy.sh run noniid_impact    — Non-IID severity sweep (alpha 1.0→0.1)"
  echo "  ./deploy.sh run dp_tradeoff      — DP epsilon sweep (50→1)"
  echo ""
  echo "Privacy (5 min):"
  echo "  ./deploy.sh run privacy_attacks  — DLG + MIA with/without DP"
  echo ""
  echo "Full benchmarks (hours):"
  echo "  ./deploy.sh run full_benchmark   — All 4 tasks × key strategies (~2-3h)"
  echo "  ./deploy.sh run chest_production — Real NIH chest X-ray, 7 strategies (~8h)"
  echo "  ./deploy.sh run all              — Everything (~10-12h)"
  echo ""
  echo "Built-in tasks:"
  echo "  ./deploy.sh run sepsis|ecg|fraud|chest|privacy"
  echo ""
  echo "Other commands:"
  echo "  ./deploy.sh logs training        — Monitor running experiment"
  echo "  ./deploy.sh results summary      — View latest results"
  echo "  ./deploy.sh results              — Download results to local"
  echo "  ./deploy.sh dashboard            — Start Streamlit dashboard"
  echo "  ./deploy.sh status               — Cluster health check"
}


cmd_status() {
  echo "============================================"
  echo "CLUSTER STATUS"
  echo "============================================"

  echo ""
  echo "SuperLink ($SUPERLINK):"
  on $SUPERLINK "
    sudo docker ps --format '  {{.Names}}: {{.Status}}' 2>/dev/null
    echo '  GPU:'
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
    echo '  Disk:'
    df -h / | tail -1 | awk '{print \"    Used: \" \$3 \" / \" \$2 \" (\" \$5 \")\"}'
  " 2>/dev/null

  # Count connected nodes
  NODES=$(on $SUPERLINK "sudo docker logs $CONTAINER_SL 2>&1 | grep -oP 'node_id=\d+' | sort -u | wc -l" 2>/dev/null)
  echo "  Connected nodes: $NODES/$NUM_CLIENTS"

  echo ""
  echo "SuperNodes:"
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    ip="${CLIENTS[$i]}"
    status=$(on $ip "sudo docker ps --filter name=$CONTAINER_SN --format '{{.Status}}'" 2>/dev/null || echo "unreachable")
    echo "  Client $i ($ip): $status"
  done

  echo ""
  echo "Training container:"
  on $SUPERLINK "sudo docker ps --filter name=$CONTAINER_RUN --format '  {{.Names}}: {{.Status}}'" 2>/dev/null || echo "  Not running"

  echo ""
  echo "Recent results:"
  on $SUPERLINK "ls -lt ~/fl-deploy/results/*.json 2>/dev/null | head -3 | awk '{print \"  \" \$NF}'" 2>/dev/null || echo "  None"
}

cmd_logs() {
  TARGET="${1:-superlink}"
  if [ "$TARGET" = "superlink" ]; then
    on $SUPERLINK "sudo docker logs $CONTAINER_SL --tail 100 -f" 2>/dev/null
  elif [ "$TARGET" = "training" ]; then
    on $SUPERLINK "sudo docker logs $CONTAINER_RUN --tail 200 -f" 2>/dev/null
  elif [ "$TARGET" = "results" ]; then
    on $SUPERLINK "cat \$(ls -t ~/fl-deploy/results/*.json 2>/dev/null | head -1)" 2>/dev/null
  elif [ "$TARGET" = "all" ]; then
    echo "=== SuperLink ==="
    on $SUPERLINK "sudo docker logs $CONTAINER_SL --tail 10 2>&1" 2>/dev/null
    echo ""
    echo "=== Training ==="
    on $SUPERLINK "sudo docker logs $CONTAINER_RUN --tail 20 2>&1" 2>/dev/null
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
      echo ""
      echo "=== Client $i (${CLIENTS[$i]}) ==="
      on "${CLIENTS[$i]}" "sudo docker logs $CONTAINER_SN --tail 10 2>&1" 2>/dev/null
    done
  else
    on "${CLIENTS[$TARGET]}" "sudo docker logs $CONTAINER_SN --tail 100 -f" 2>/dev/null
  fi
}

cmd_down() {
  echo "Stopping all containers..."
  on $SUPERLINK "sudo docker rm -f $CONTAINER_SL $CONTAINER_RUN 2>/dev/null" &
  for ip in "${CLIENTS[@]}"; do
    on "$ip" "sudo docker rm -f $CONTAINER_SN 2>/dev/null" &
  done
  wait
  echo "All stopped."
}

cmd_restart() {
  cmd_down
  sleep 3

  echo "Starting SuperLink..."
  on $SUPERLINK "
    sudo docker run -d --name $CONTAINER_SL --restart unless-stopped --network host \
      -v ~/fl-deploy/certs:/certs:ro --log-opt max-size=100m $IMAGE \
      flower-superlink --ssl-certfile /certs/server.pem --ssl-keyfile /certs/server.key --ssl-ca-certfile /certs/ca.pem
  " 2>/dev/null
  sleep 5

  echo "Starting clients..."
  for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    on "${CLIENTS[$i]}" "
      sudo docker run -d --name $CONTAINER_SN --restart unless-stopped --network host \
        -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro -v ~/fl-deploy/data:/data:ro \
        -v ~/fl-deploy/logs:/app/logs \
        -e PYTHONUNBUFFERED=1 \
        --log-opt max-size=100m $IMAGE \
        flower-supernode --root-certificates /certs/ca.pem --superlink $SUPERLINK_PRIVATE:9092 \
          --node-config 'partition-id=$i num-clients=$NUM_CLIENTS' \
          --clientappio-api-address 0.0.0.0:7070
    " 2>/dev/null &
  done
  wait
  sleep 10
  cmd_status
}

# ============================================================

[ $# -eq 0 ] && { cmd_list; exit 0; }
case "$1" in
  up)        cmd_up ;;
  run)       cmd_run "${2:-all}" ;;
  status)    cmd_status ;;
  logs)      cmd_logs "${2:-superlink}" ;;
  results)   cmd_results "${2:-download}" ;;
  dashboard) cmd_dashboard ;;
  list)      cmd_list ;;
  down)      cmd_down ;;
  restart)   cmd_restart ;;
  *)         cmd_list; exit 1 ;;
esac
