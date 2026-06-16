#!/bin/bash
# FL + PET Sandbox — Full Distributed Run
# =========================================
# Server runs run_ec2.py (iterates strategies, calls start_server per strategy).
# Clients run run_client.py (loops, reconnects for each strategy).
#
# Usage:
#   ./run_distributed.sh              # all 8 tasks
#   ./run_distributed.sh fraud        # single task
#   ./run_distributed.sh quick        # smoke test (fraud IID only)

set -uo pipefail

KEY="$(cd "$(dirname "$0")" && pwd)/TEE_FL.pem"
SERVER="54.151.221.104"
SERVER_PRIVATE="172.31.4.42"
CLIENTS=(47.130.0.207 47.129.54.224 52.221.246.101 175.41.152.74 3.0.16.188)
NC=${#CLIENTS[@]}
IMAGE="healthcare-fl:latest"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $KEY"
on() { $SSH ec2-user@"$1" "${@:2}"; }

TARGET="${1:-all}"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

echo "============================================"
echo "FL + PET SANDBOX — DISTRIBUTED MODE"
echo "  Target: $TARGET"
echo "  Server: $SERVER ($SERVER_PRIVATE)"
echo "  Clients: ${CLIENTS[*]} ($NC nodes, L4 GPU)"
echo "  Image: $IMAGE"
echo "  Time: $TIMESTAMP"
echo "============================================"

# ── Cleanup ──────────────────────────────────────────────────────────
echo ""
echo ">>> Stopping existing containers..."
on $SERVER "docker rm -f fl-superlink fl-training 2>/dev/null" &
for ip in "${CLIENTS[@]}"; do
  on "$ip" "sudo docker rm -f fl-supernode fl-client 2>/dev/null" &
done
wait
sleep 2

# ── Run a task ───────────────────────────────────────────────────────

run_task() {
  local TASK="$1"
  echo ""
  echo "======================================================================"
  echo "  STARTING: $TASK (distributed, $NC clients)"
  echo "======================================================================"

  # Create results dir
  on $SERVER "mkdir -p ~/fl-deploy/results"

  # Start clients first (they will loop and wait for server)
  echo "  Starting clients..."
  for i in $(seq 0 $((NC - 1))); do
    local ip="${CLIENTS[$i]}"
    on "$ip" "
      sudo docker rm -f fl-client 2>/dev/null
      sudo docker run -d --name fl-client --network host --gpus all \
        -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro \
        -v ~/fl-deploy/data:/data:ro \
        -e PARTITION_ID=$i \
        -e NUM_CLIENTS=$NC \
        -e FL_TASK=$TASK \
        -e FL_SERVER=$SERVER_PRIVATE:9092 \
        -e CERTS_DIR=/certs \
        -e DATA_PATH=/data \
        -e PYTHONUNBUFFERED=1 \
        -e SYNTHETIC=0 \
        --log-opt max-size=100m \
        $IMAGE \
        python3 run_client.py
    " 2>/dev/null &
  done
  wait
  sleep 3

  # Start server (blocks until all strategies for this task complete)
  echo "  Starting server for task: $TASK..."
  on $SERVER "
    docker rm -f fl-training 2>/dev/null
    docker run -d --name fl-training --network host \
      -v ~/fl-deploy/certs:/certs:ro \
      -v ~/fl-deploy/results:/app/results \
      -e FL_DISTRIBUTED=1 \
      -e SUPERLINK_ADDRESS=0.0.0.0:9092 \
      -e CERTS_DIR=/certs \
      -e PYTHONUNBUFFERED=1 \
      --log-opt max-size=200m \
      $IMAGE \
      python3 run_ec2.py --distributed $TASK
  " 2>/dev/null

  # Monitor server until it finishes
  echo "  Running... (monitoring every 30s)"
  while true; do
    STATUS=$(on $SERVER "docker inspect fl-training --format '{{.State.Status}}'" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "exited" ]; then
      EXIT_CODE=$(on $SERVER "docker inspect fl-training --format '{{.State.ExitCode}}'" 2>/dev/null || echo "?")
      echo ""
      echo "  Task $TASK completed (exit code: $EXIT_CODE)"
      # Print summary
      on $SERVER "docker logs fl-training 2>&1 | tail -20" 2>/dev/null | grep -E "Final|TOTAL|ERROR|PASS|FAIL" | head -20 || true
      break
    elif [ "$STATUS" = "running" ]; then
      LAST=$(on $SERVER "docker logs fl-training --tail 3 2>&1" 2>/dev/null | grep -oE "Round [0-9]+.*|Final.*|\[.*\].*rounds" | tail -1 || true)
      [ -n "$LAST" ] && echo "    $LAST" || true
      sleep 30
    else
      echo "  WARNING: unexpected status: $STATUS"
      on $SERVER "docker logs fl-training --tail 10 2>&1" 2>/dev/null
      break
    fi
  done

  # Stop clients
  for ip in "${CLIENTS[@]}"; do
    on "$ip" "sudo docker rm -f fl-client 2>/dev/null" &
  done
  wait
  echo "  Clients stopped."
}

# ── Main ─────────────────────────────────────────────────────────────

T0=$(date +%s)

case "$TARGET" in
  all)
    for task in fraud sepsis ecg anomaly mortality drug readmission satellite chest vfl split transfer privacy; do
      run_task "$task"
    done
    ;;
  quick)
    run_task "fraud"
    ;;
  new)
    for task in anomaly mortality drug readmission satellite; do
      run_task "$task"
    done
    ;;
  *)
    run_task "$TARGET"
    ;;
esac

T1=$(date +%s)
ELAPSED=$((T1 - T0))

echo ""
echo "============================================"
echo "DISTRIBUTED RUN COMPLETE"
echo "  Total time: ${ELAPSED}s ($((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m)"
echo "============================================"

# Download results
echo ""
echo ">>> Downloading results..."
mkdir -p "$(dirname "$0")/results"
scp -o StrictHostKeyChecking=no -i "$KEY" \
  "ec2-user@$SERVER:~/fl-deploy/results/*" \
  "$(dirname "$0")/results/" 2>/dev/null
echo "Results saved to results/"

# Restart SuperLink for idle monitoring
on $SERVER "
  docker run -d --name fl-superlink --restart unless-stopped --network host \
    -v ~/fl-deploy/certs:/certs:ro --log-opt max-size=100m \
    $IMAGE \
    flower-superlink --ssl-certfile /certs/server.pem --ssl-keyfile /certs/server.key --ssl-ca-certfile /certs/ca.pem
" 2>/dev/null
echo "SuperLink restarted."
