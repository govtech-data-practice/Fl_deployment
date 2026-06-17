# Tutorial 1: Setup & First Run

**Time:** 15 minutes | **Level:** Beginner | **Requirements:** Python 3.10+, 4 GB RAM

## What You'll Learn

- Install the FL platform and its dependencies
- Verify your environment is ready
- Run your first federated learning experiment

## Step 1: Clone and Install

```bash
git clone <repository-url> fl-reference
cd fl-reference

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install core + dev dependencies
pip install -e ".[dev]"
```

**Checkpoint:** `pip list | grep flwr` should show Flower installed.

## Step 2: Verify Your Environment

Run the pre-flight check:

```bash
./scripts/preflight.sh --check tooling
```

**Expected output:**
```
FL Platform Pre-flight Checks
==============================

Tooling:
  [PASS] Python 3.12.x
  [PASS] Docker version 28.x.x
  [PASS] Flower 1.30.0
  [PASS] PyTorch 2.x.x (CPU or CUDA)
  [PASS] OpenSSL 3.x.x
```

If PyTorch shows `[FAIL]`, ensure your virtual environment is activated.

## Step 3: Run the Smoke Test

Run FedAvg with 2 simulated clients on synthetic fraud data:

```bash
python runners/run_ec2.py fraud --synthetic
```

**What's happening:**
1. Synthetic data is generated for 2 virtual clients (each gets ~1000 samples)
2. A Flower simulation runs 3 rounds of FedAvg
3. Each round: clients train locally, send model updates, server aggregates
4. Final accuracy is reported

**Expected output (last few lines):**
```
Round 3/3 — accuracy: ~0.95-0.98
Training complete. Results saved.
```

This takes 30-60 seconds on CPU.

## Step 4: Check the DP Budget Calculator

The platform includes a privacy budget calculator. Try it:

```bash
python tools/dp_budget.py --all --rounds 100
```

**Expected output:**
```
DP Budget Summary — 100 rounds, delta=1e-05
Preset             sigma      C    epsilon
------------------------------------------
DP_STRONG            1.5    1.0      55.96
DP_MODERATE          0.8    1.0     167.76
DP_RELAXED           0.5    1.0     411.51
```

You'll learn what these numbers mean in [Tutorial 4: Differential Privacy](../intermediate/04-differential-privacy.md).

## What Just Happened?

You ran a **federated learning simulation** where:

- **No data was centralised** — each simulated client only saw its own partition
- **Only model updates were shared** — the server never saw raw data
- **FedAvg** aggregated updates by averaging model weights

This is the core FL loop. Every other tutorial builds on this foundation.

## Next Steps

- [Tutorial 2: Your First Model](02-first-model.md) — try different tasks and models
