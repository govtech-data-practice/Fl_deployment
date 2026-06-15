#!/bin/bash
# FL Platform Pre-flight Validation
# ==================================
# Validates infrastructure readiness before deployment or training.
#
# Usage:
#   ./scripts/preflight.sh                           # run all checks
#   ./scripts/preflight.sh --check landing-zone      # single check
#   ./scripts/preflight.sh --check iam --check dns   # multiple checks
#
# Checks:
#   landing-zone  — data directories, disk space, file permissions
#   iam           — SSH keys, AWS credentials, user permissions
#   endpoints     — gRPC port, server/client reachability
#   dns           — hostname resolution for coordinator and clients
#   tooling       — docker, python, flwr, openssl versions
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parse arguments ────────────────────────────────────────────────
CHECKS=()
for arg in "$@"; do
    case "$arg" in
        --check) ;;  # next arg is the check name
        landing-zone|iam|endpoints|dns|tooling)
            CHECKS+=("$arg") ;;
        --help|-h)
            head -14 "$0" | tail -13
            exit 0 ;;
        *)
            echo "Unknown argument: $arg"
            echo "Valid checks: landing-zone, iam, endpoints, dns, tooling"
            exit 1 ;;
    esac
done

# Default: run all checks
if [ ${#CHECKS[@]} -eq 0 ]; then
    CHECKS=(landing-zone iam endpoints dns tooling)
fi

PASS=0
FAIL=0
WARN=0

check() {
    local status="$1"  # pass, fail, warn
    local msg="$2"
    case "$status" in
        pass) PASS=$((PASS + 1)); printf "  [PASS] %s\n" "$msg" ;;
        fail) FAIL=$((FAIL + 1)); printf "  [FAIL] %s\n" "$msg" ;;
        warn) WARN=$((WARN + 1)); printf "  [WARN] %s\n" "$msg" ;;
    esac
}

# ── Load cluster config if available ───────────────────────────────
ENV_FILE="${REPO_ROOT}/cluster.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# ── Check: landing-zone ───────────────────────────────────────────
check_landing_zone() {
    echo ""
    echo "Landing Zone:"

    # Data directory
    DATA_DIR="${FL_DATA_DIR:-/home/ec2-user/fl-deploy/data}"
    if [ -d "$DATA_DIR" ]; then
        check pass "Data directory exists: $DATA_DIR"
    else
        check warn "Data directory not found: $DATA_DIR (will be created on first ingest)"
    fi

    # Results directory
    RESULTS_DIR="${FL_RESULTS_DIR:-/home/ec2-user/fl-deploy/results}"
    if [ -d "$RESULTS_DIR" ]; then
        check pass "Results directory exists: $RESULTS_DIR"
    else
        check warn "Results directory not found: $RESULTS_DIR"
    fi

    # Disk space (local)
    DISK_PCT=$(df / --output=pcent 2>/dev/null | tail -1 | tr -d ' %' || df / | tail -1 | awk '{print $5}' | tr -d '%')
    if [ -n "$DISK_PCT" ]; then
        if [ "$DISK_PCT" -gt 90 ]; then
            check fail "Disk usage ${DISK_PCT}% (>90%)"
        elif [ "$DISK_PCT" -gt 80 ]; then
            check warn "Disk usage ${DISK_PCT}% (>80%)"
        else
            check pass "Disk usage ${DISK_PCT}%"
        fi
    fi

    # Repo structure
    for dir in fl_common models tasks privacy; do
        if [ -d "$REPO_ROOT/$dir" ]; then
            check pass "Repository directory: $dir/"
        else
            check fail "Missing repository directory: $dir/"
        fi
    done
}

# ── Check: iam ────────────────────────────────────────────────────
check_iam() {
    echo ""
    echo "IAM & Credentials:"

    # SSH key
    SSH_KEY="${FL_SSH_KEY:-}"
    if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
        PERMS=$(stat -c %a "$SSH_KEY" 2>/dev/null || stat -f %Lp "$SSH_KEY" 2>/dev/null)
        if [ "$PERMS" = "400" ] || [ "$PERMS" = "600" ]; then
            check pass "SSH key: $SSH_KEY (permissions $PERMS)"
        else
            check warn "SSH key permissions are $PERMS (expected 400 or 600)"
        fi
    elif [ -n "$SSH_KEY" ]; then
        check fail "SSH key not found: $SSH_KEY"
    else
        check warn "FL_SSH_KEY not set in cluster.env"
    fi

    # AWS credentials
    if command -v aws &>/dev/null; then
        if aws sts get-caller-identity &>/dev/null; then
            ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
            check pass "AWS credentials valid (account: $ACCOUNT)"
        else
            check warn "AWS credentials not configured or expired"
        fi
    else
        check warn "AWS CLI not installed"
    fi
}

# ── Check: endpoints ──────────────────────────────────────────────
check_endpoints() {
    echo ""
    echo "Endpoints:"

    SERVER="${FL_SERVER_HOST:-}"
    PORT="${FL_GRPC_PORT:-9092}"

    if [ -z "$SERVER" ]; then
        check warn "FL_SERVER_HOST not set — skipping endpoint checks"
        return
    fi

    # Server SSH
    SSH_KEY="${FL_SSH_KEY:-}"
    SSH_USER="${FL_SSH_USER:-ec2-user}"
    if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
        if ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
            "${SSH_USER}@${SERVER}" "echo OK" 2>/dev/null; then
            check pass "Server SSH: $SERVER"
        else
            check fail "Cannot SSH to server: $SERVER"
        fi
    fi

    # gRPC port
    if command -v nc &>/dev/null; then
        if nc -z -w 3 "$SERVER" "$PORT" 2>/dev/null; then
            check warn "Port $PORT already in use on $SERVER (training may be running)"
        else
            check pass "Port $PORT available on $SERVER"
        fi
    fi

    # Delegate to existing health check for full cluster checks
    if [ -x "$REPO_ROOT/deploy/health_check.sh" ]; then
        check pass "Full health check available: deploy/health_check.sh"
    fi
}

# ── Check: dns ────────────────────────────────────────────────────
check_dns() {
    echo ""
    echo "DNS:"

    SERVER="${FL_SERVER_HOST:-}"
    if [ -z "$SERVER" ]; then
        check warn "FL_SERVER_HOST not set — skipping DNS checks"
        return
    fi

    # Check if server host resolves (if it's a hostname, not IP)
    if echo "$SERVER" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
        check pass "Server is an IP address: $SERVER (no DNS needed)"
    else
        if host "$SERVER" &>/dev/null || nslookup "$SERVER" &>/dev/null; then
            check pass "Server hostname resolves: $SERVER"
        else
            check fail "Server hostname does not resolve: $SERVER"
        fi
    fi

    # Check client hosts
    for ip in ${FL_CLIENT_HOSTS:-}; do
        if echo "$ip" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'; then
            check pass "Client IP: $ip"
        else
            if host "$ip" &>/dev/null 2>&1; then
                check pass "Client hostname resolves: $ip"
            else
                check fail "Client hostname does not resolve: $ip"
            fi
        fi
    done
}

# ── Check: tooling ────────────────────────────────────────────────
check_tooling() {
    echo ""
    echo "Tooling:"

    # Python
    if command -v python3 &>/dev/null; then
        PY_VER=$(python3 --version 2>&1)
        check pass "$PY_VER"
    else
        check fail "python3 not found"
    fi

    # Docker
    if command -v docker &>/dev/null; then
        DOCKER_VER=$(docker --version 2>&1)
        check pass "$DOCKER_VER"
    else
        check fail "Docker not installed"
    fi

    # Flower
    if python3 -c "import flwr; print(f'Flower {flwr.__version__}')" 2>/dev/null; then
        FLWR_VER=$(python3 -c "import flwr; print(flwr.__version__)")
        check pass "Flower $FLWR_VER"
    else
        check fail "Flower not installed (pip install flwr)"
    fi

    # PyTorch
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
        CUDA_AVAIL=$(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')")
        check pass "PyTorch $TORCH_VER ($CUDA_AVAIL)"
    else
        check fail "PyTorch not installed"
    fi

    # OpenSSL
    if command -v openssl &>/dev/null; then
        SSL_VER=$(openssl version 2>&1)
        check pass "$SSL_VER"
    else
        check warn "openssl not found (needed for certificate operations)"
    fi

    # Terraform (optional)
    if command -v terraform &>/dev/null; then
        TF_VER=$(terraform version -json 2>/dev/null | python3 -c "import sys,json;print(json.load(sys.stdin)['terraform_version'])" 2>/dev/null || terraform version | head -1)
        check pass "Terraform $TF_VER"
    else
        check warn "Terraform not installed (optional, needed for infra provisioning)"
    fi
}

# ── Run selected checks ───────────────────────────────────────────
echo "FL Platform Pre-flight Checks"
echo "=============================="

for c in "${CHECKS[@]}"; do
    case "$c" in
        landing-zone) check_landing_zone ;;
        iam)          check_iam ;;
        endpoints)    check_endpoints ;;
        dns)          check_dns ;;
        tooling)      check_tooling ;;
    esac
done

# ── Summary ────────────────────────────────────────────────────────
echo ""
echo "================================================"
if [ $FAIL -gt 0 ]; then
    echo "FAILED: $PASS pass, $FAIL fail, $WARN warn"
    exit 1
elif [ $WARN -gt 0 ]; then
    echo "READY (with warnings): $PASS pass, $FAIL fail, $WARN warn"
    exit 0
else
    echo "READY: $PASS pass, $FAIL fail, $WARN warn"
    exit 0
fi
