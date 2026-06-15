#!/bin/bash
# FL Platform Diagnostic Bundle
# ===============================
# Collects system state, logs, and configuration into a single archive
# for troubleshooting.
#
# Usage:
#   ./scripts/diagnose.sh --run-id <id> --env <env> --since 2h
#   ./scripts/diagnose.sh --run-id smoke-001 --env production --since 4h
#   ./scripts/diagnose.sh --help
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────
RUN_ID="unknown"
ENVIRONMENT="local"
SINCE="2h"

# ── Parse arguments ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)   RUN_ID="$2"; shift 2 ;;
        --env)      ENVIRONMENT="$2"; shift 2 ;;
        --since)    SINCE="$2"; shift 2 ;;
        --help|-h)
            head -10 "$0" | tail -8
            echo ""
            echo "Options:"
            echo "  --run-id <id>      Run identifier"
            echo "  --env <env>        Environment (local, dev, staging, production)"
            echo "  --since <duration> How far back to collect logs (e.g. 2h, 30m, 1d)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BUNDLE_DIR="/tmp/fl-diag-${RUN_ID}-${TIMESTAMP}"
BUNDLE_TAR="diag-${RUN_ID}-${TIMESTAMP}.tar.gz"

mkdir -p "$BUNDLE_DIR"

echo "FL Diagnostic Bundle"
echo "===================="
echo "Run ID:      $RUN_ID"
echo "Environment: $ENVIRONMENT"
echo "Since:       $SINCE"
echo "Output:      $BUNDLE_TAR"
echo ""

# ── System info ───────────────────────────────────────────────────
echo "Collecting system info..."
{
    echo "=== System ==="
    uname -a
    echo ""
    echo "=== Disk ==="
    df -h 2>/dev/null || df
    echo ""
    echo "=== Memory ==="
    free -h 2>/dev/null || vm_stat 2>/dev/null || echo "(not available)"
    echo ""
    echo "=== GPU ==="
    nvidia-smi 2>/dev/null || echo "No GPU detected"
    echo ""
    echo "=== Python ==="
    python3 --version 2>&1
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not available"
    python3 -c "import flwr; print(f'Flower {flwr.__version__}')" 2>/dev/null || echo "Flower not available"
    echo ""
    echo "=== Docker ==="
    docker version 2>/dev/null || echo "Docker not available"
} > "$BUNDLE_DIR/system_info.txt" 2>&1

# ── Configuration snapshot (sanitised) ────────────────────────────
echo "Collecting configuration..."
{
    echo "=== Environment: $ENVIRONMENT ==="
    echo ""

    # Sanitise cluster.env — redact IPs, keys, passwords
    if [ -f "$REPO_ROOT/cluster.env" ]; then
        echo "=== cluster.env (sanitised) ==="
        sed -E \
            -e 's/([0-9]+\.[0-9]+\.[0-9]+\.)[0-9]+/\1***/g' \
            -e 's/(KEY|SECRET|PASSWORD|TOKEN)=.*/\1=<REDACTED>/gi' \
            -e 's|(SSH_KEY=).*|\1<REDACTED>|' \
            "$REPO_ROOT/cluster.env"
    else
        echo "cluster.env not found"
    fi

    echo ""
    echo "=== env.example.yaml ==="
    cat "$REPO_ROOT/env.example.yaml" 2>/dev/null || echo "Not found"
} > "$BUNDLE_DIR/config_snapshot.txt" 2>&1

# ── Certificate status ────────────────────────────────────────────
echo "Collecting certificate status..."
{
    echo "=== TLS Certificates ==="
    CERTS_DIR="${FL_CERTS_DIR:-$REPO_ROOT/certs}"
    if [ -d "$CERTS_DIR" ]; then
        for cert in "$CERTS_DIR"/*.pem "$CERTS_DIR"/*.crt; do
            [ -f "$cert" ] || continue
            echo ""
            echo "--- $(basename "$cert") ---"
            openssl x509 -in "$cert" -noout -subject -issuer -dates 2>/dev/null || echo "Cannot parse: $cert"
        done
    else
        echo "Certs directory not found: $CERTS_DIR"
    fi
} > "$BUNDLE_DIR/cert_status.txt" 2>&1

# ── Docker logs ───────────────────────────────────────────────────
echo "Collecting Docker logs..."
{
    echo "=== Running containers ==="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Docker not available"
    echo ""

    # Collect logs from FL-related containers
    for container in $(docker ps -a --filter "name=fl-" --filter "name=superlink" --filter "name=supernode" --format "{{.Names}}" 2>/dev/null); do
        echo ""
        echo "=== Container: $container ==="
        docker logs --since "$SINCE" "$container" 2>&1 | tail -200
    done
} > "$BUNDLE_DIR/docker_logs.txt" 2>&1

# ── Run records ───────────────────────────────────────────────────
echo "Collecting run records..."
{
    # Check runs/ and results/ directories
    for dir in "$REPO_ROOT/runs" "$REPO_ROOT/results"; do
        if [ -d "$dir" ]; then
            echo "=== $dir ==="
            ls -la "$dir" 2>/dev/null
            echo ""
            # Copy relevant run files
            if [ "$RUN_ID" != "unknown" ]; then
                find "$dir" -name "*${RUN_ID}*" -exec cp {} "$BUNDLE_DIR/" \; 2>/dev/null
            fi
        fi
    done
} > "$BUNDLE_DIR/run_records.txt" 2>&1

# ── Health check (if available) ───────────────────────────────────
if [ -x "$REPO_ROOT/deploy/health_check.sh" ]; then
    echo "Running health check..."
    "$REPO_ROOT/deploy/health_check.sh" --json > "$BUNDLE_DIR/health_check.json" 2>&1 || true
fi

# ── Flower logs ───────────────────────────────────────────────────
echo "Collecting Flower logs..."
{
    # Look for log files in common locations
    for logdir in "$REPO_ROOT/logs" /tmp/flwr* /var/log/fl*; do
        if [ -d "$logdir" ]; then
            echo "=== $logdir ==="
            find "$logdir" -name "*.log" -newer /dev/null -exec echo "--- {} ---" \; -exec tail -100 {} \; 2>/dev/null
        fi
    done
} > "$BUNDLE_DIR/flower_logs.txt" 2>&1

# ── Create archive ────────────────────────────────────────────────
echo ""
echo "Creating archive..."
tar -czf "$BUNDLE_TAR" -C /tmp "$(basename "$BUNDLE_DIR")" 2>/dev/null

# Clean up temp directory
rm -rf "$BUNDLE_DIR"

echo ""
echo "================================================"
echo "Diagnostic bundle: $BUNDLE_TAR"
echo "Share this file with the operations team for troubleshooting."
ls -lh "$BUNDLE_TAR"
