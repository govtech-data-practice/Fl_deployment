# Tutorial 2: Your First Model

**Time:** 25 minutes | **Level:** Beginner | **Prerequisites:** [Tutorial 1](01-setup.md)

## What You'll Learn

- Train a centralised baseline model (upper bound)
- Train the same model with federated learning
- Compare centralised vs FL results
- Understand the accuracy/privacy trade-off

> **Note:** This tutorial uses simplified training for learning purposes. In production,
> you should add sanity checks: data validation (`tools/validate_manifest.py`),
> privacy budget verification (`tools/dp_budget.py`), model evaluation on held-out
> test sets, and privacy attack testing (see [Tutorial 7](../intermediate/07-privacy-attacks.md)).

## Step 1: Generate Sample Data

```bash
python data/generators/generate_all.py --task fraud --num-samples 500
```

## Step 2: Train a Centralised Baseline

First, train on **all data pooled together** — no federation, no privacy. This is the upper bound that FL should approach:

```bash
python benchmarks/centralized/train_task.py fraud --epochs 10
```

**Expected output:**
```
 Epoch   Train Loss   Val Loss    Val Acc
------------------------------------------
     1       0.6941     0.6826     0.5400
     2       0.6682     0.6655     0.6400
     3       0.6405     0.6403     0.7000
     ...
    10       0.1930     0.2842     0.8800
------------------------------------------
Final: accuracy=0.8800  loss=0.2842  time=0.4s
```

Note the final accuracy (~0.88). This is the target.

## Step 3: Train with Federated Learning

Now train the **same model** — but data is split across 3 simulated clients:

```bash
python runners/run_ec2.py fraud --synthetic
```

Each client only sees its own data partition. Model updates (not raw data) are sent to the coordinator for aggregation.

## Step 4: Compare Side-by-Side

Run the benchmark comparison:

```bash
python benchmarks/run_benchmarks.py --tasks fraud --epochs 10 --fl-rounds 5
```

**Expected comparison:**
```
Task            Mode                   Accuracy       Time
----------------------------------------------------------
fraud           Centralised              0.8800      0.4s
fraud           FL (FedAvg)              0.6120      0.2s
```

FL accuracy is lower with only 5 rounds. With 30+ rounds, FL approaches the centralised baseline.

**Key insight:**

| Approach | Data | Accuracy | Privacy |
|----------|------|----------|---------|
| Centralised | All data pooled in one place | ~0.88 (upper bound) | None |
| Federated (5 rounds) | Data stays at each client | ~0.61 | Strong — data never moves |
| Federated (30 rounds) | Data stays at each client | ~0.85+ | Strong — data never moves |

FL trades some convergence speed for privacy — each participant's raw data never leaves their site.

## Step 5: Try Different Tasks

Each task uses a different model architecture:

```bash
# Time-series: sepsis early warning (BiLSTM)
python benchmarks/centralized/train_task.py sepsis --epochs 10

# Signal processing: ECG arrhythmia (BiLSTM)
python benchmarks/centralized/train_task.py ecg --epochs 10

# Unsupervised: anomaly detection (Autoencoder)
python benchmarks/centralized/train_task.py anomaly --epochs 10
```

## Step 6: Model-Task Reference

| Task | Model | Parameters | Data Type |
|------|-------|-----------|-----------|
| `fraud` | MLP | 6K | Tabular (30 features) |
| `sepsis` | BiLSTM | 41K | Time-series (48 steps, 14 features) |
| `ecg` | BiLSTM | 41K | Time-series (250 steps, 12 features) |
| `anomaly` | Autoencoder | 2K | Tabular (40 features) |
| `mortality` | TabNet | 1M | Tabular (25 features) |
| `readmission` | LogReg | 10K | Tabular (20 features) |
| `satellite` | ResNet-small | 5M | Images (64x64x3) |
| `chest_xray` | DenseNet-121 | 8M | Images (224x224x3) |

Models live in `models/hfl/`, tasks in `tasks/hfl/`.

## Step 7: Explore the Code

```
models/hfl/mlp/
  server_app.py    # Aggregation strategy + model definition
  client_app.py    # Local training loop + data loading
tasks/hfl/fraud/
  data.py          # Data loading + partitioning
```

> **Production note:** The simplified examples above skip several steps that are
> essential in production:
> - **Data validation** — run `python tools/validate_manifest.py` before training
> - **Privacy budget** — verify epsilon with `python tools/dp_budget.py`
> - **DP + SecAgg** — enable privacy controls (see [Tutorial 4](../intermediate/04-differential-privacy.md) and [Tutorial 5](../intermediate/05-secure-aggregation.md))
> - **Privacy testing** — run attack suite before model release (see [Tutorial 7](../intermediate/07-privacy-attacks.md))
> - **Model card** — document model provenance (see `templates/model_card.md`)

## Next Steps

- [Tutorial 3: Data Pipeline](03-data-pipeline.md) — ingest, validate, and manage data
