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

```bash
python3 -c "import flwr; print('Flower', flwr.__version__)"
python3 -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python3 -c "import fl_pets; print('fl_pets: ok')"
```

**Expected output:**
```
Flower 1.30.0
PyTorch 2.x.x | CUDA: True (or False on CPU)
fl_pets: ok
```

## Step 3: Generate Synthetic Data

```bash
python data/generators/generate_all.py --task fraud --num-samples 500
```

This creates `data/samples/fraud/` with `data.npz`, `manifest.json`, and `data_card.md`.

## Step 4: Run the Smoke Test

Run FedAvg with 2 simulated clients on synthetic fraud data:

```bash
python runners/run_ec2.py fraud --synthetic
```

**What's happening:**
1. Synthetic data is partitioned across 2 virtual clients
2. A Flower simulation runs 3 rounds of FedAvg
3. Each round: clients train locally, send model updates, server aggregates
4. Final accuracy is reported

This takes 30-60 seconds on CPU.

## Step 5: Check the DP Budget

```bash
python tools/dp_budget.py --all --rounds 100
```

You'll learn what these numbers mean in [Tutorial 4: Differential Privacy](../intermediate/04-differential-privacy.md).

## What Just Happened?

You ran a **federated learning simulation** where:

- **No data was centralised** — each simulated client only saw its own partition
- **Only model updates were shared** — the server never saw raw data
- **FedAvg** aggregated updates by averaging model weights

This is the core FL loop. Every other tutorial builds on this foundation.

## Next Steps

- [Tutorial 2: Your First Model](02-first-model.md) — train a centralised baseline, then compare with FL
