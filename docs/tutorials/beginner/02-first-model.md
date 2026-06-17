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

**Output (EC2, NVIDIA L4):**
```
 Epoch   Train Loss   Val Loss    Val Acc
------------------------------------------
     1       0.6931     0.6818     0.5300
     2       0.6684     0.6625     0.7000
     3       0.6389     0.6348     0.7600
     4       0.5967     0.5966     0.7800
     5       0.5333     0.5378     0.7900
     6       0.4619     0.4670     0.8300
     7       0.3771     0.3975     0.8400
     8       0.3031     0.3361     0.8700
     9       0.2424     0.2874     0.8900
    10       0.1964     0.2496     0.8800
------------------------------------------
Final: accuracy=0.8800  loss=0.2496  time=0.4s
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

**Output (EC2, NVIDIA L4):**
```
Task            Mode                   Accuracy       Time
----------------------------------------------------------
fraud           Centralised              0.8800      0.4s
fraud           FL (FedAvg, 5 rounds)    0.6160      0.0s
```

FL accuracy is lower with only 5 rounds. With 30+ rounds, FL approaches the centralised baseline.

**Key insight:**

| Approach | Data | Accuracy | Privacy |
|----------|------|----------|---------|
| Centralised | All data pooled in one place | 0.8800 | None |
| Federated (5 rounds) | Data stays at each client | 0.6160 | Strong — data never moves |
| Federated (30 rounds) | Data stays at each client | ~0.85+ | Strong — data never moves |

FL trades some convergence speed for privacy — each participant's raw data never leaves their site.

## Step 5: Vertical FL — Split Features Across Parties

HFL assumes all clients have the **same features** but different samples.
VFL handles the case where clients have **different features** for the same entities.

```
HFL (what we just did):              VFL (different features):
  Client A: [f1..f30] [label]          Bank A: [f1..f10]
  Client B: [f1..f30] [label]          Bank B: [f11..f20]
  Client C: [f1..f30] [label]          Bank C: [f21..f30] [label]
```

### Entity Alignment with PSI

Before VFL training, entities must be aligned across parties using Private Set Intersection:

```python
from fl_pets.psi import align_entities
import os

# 3 banks with overlapping customers
parties = {
    "bank_a": [f"cust_{i:04d}" for i in range(500)],
    "bank_b": [f"cust_{i:04d}" for i in range(200, 700)],
    "bank_c": [f"cust_{i:04d}" for i in range(300, 800)],
}

aligned = align_entities(parties, salt=os.urandom(32))

for name in parties:
    print(f"  {name}: {len(parties[name])} records -> {len(aligned[name])} aligned")
print(f"  Common entities: {len(aligned['bank_a'])}")
```

**Output (EC2):**
```
  bank_a: 500 records -> 200 aligned
  bank_b: 500 records -> 200 aligned
  bank_c: 500 records -> 200 aligned
  Common entities: 200
```

200 common entities found across all 3 banks — without revealing non-matching records.

### VFL Model

Each party runs a bottom model on its features, producing an embedding. Embeddings are combined for prediction:

```python
import torch
from models.vfl.vfl_mlp.server_app import VFLBottomModel

# Each party has 10 features
bottom = VFLBottomModel(input_dim=10, embed_dim=16)
print(f"VFL bottom model: {sum(p.numel() for p in bottom.parameters()):,} params per party")

# Each party computes its embedding
x_party_a = torch.randn(8, 10)  # 8 samples, 10 features
embedding = bottom(x_party_a)
print(f"Input: {list(x_party_a.shape)} -> Embedding: {list(embedding.shape)}")
print("3 parties concatenate embeddings (48-dim) for the top classifier")
```

### Run VFL Training

```bash
python runners/run_ec2.py vfl_fraud --synthetic
```

This runs 3 parties, each holding features 0-9, 10-19, and 20-29. Raw features never leave each party.

See [Tutorial 10: Vertical FL & PSI](../advanced/10-vertical-fl.md) for the full VFL deep dive.

## Step 6: Model-Task Reference

### HFL Models (Horizontal — same features, different samples)

| Task | Model | Parameters | Data Type | Real Dataset |
|------|-------|-----------|-----------|-------------|
| `fraud` | MLP | 6K | Tabular (30 features) | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `sepsis` | BiLSTM | 41K | Time-series (48 steps, 14 features) | [PhysioNet eICU](https://physionet.org/content/eicu-crd/2.0/) |
| `ecg` | BiLSTM | 41K | Time-series (250 steps, 12 features) | [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) |
| `anomaly` | Autoencoder | 2K | Tabular (40 features) | Synthetic embeddings |
| `mortality` | TabNet | 1M | Tabular (25 features) | [PhysioNet eICU](https://physionet.org/content/eicu-crd/2.0/) |
| `readmission` | LogReg | 10K | Tabular (20 features) | [PhysioNet MIMIC-III](https://physionet.org/content/mimiciii/1.4/) |
| `satellite` | ResNet-small | 5M | Images (64x64x3) | [EuroSAT](https://github.com/phelber/eurosat) |
| `chest_xray` | DenseNet-121 | 8M | Images (224x224x3) | [NIH Chest X-ray 14](https://nihcc.app.box.com/v/ChestXray-NIHCC) |

### VFL Models (Vertical — different features, same entities)

| Task | Model | Parameters | Data Type | Real Dataset |
|------|-------|-----------|-----------|-------------|
| `vfl_fraud` | VFL MLP | 1K per party | Tabular (10 features/party) | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (vertically partitioned) |
| `split_sepsis` | Split BiLSTM | 41K | Time-series (split) | [PhysioNet eICU](https://physionet.org/content/eicu-crd/2.0/) |

### LLM Models

| Task | Model | Parameters | Data Type | Real Dataset |
|------|-------|-----------|-----------|-------------|
| `gov_doc` | OLMo LoRA | 1-7B (adapter only) | Text | Custom document corpus |
| `gov_llm` | Mistral QLoRA | 7B (160MB adapter) | Text | Custom clinical notes |

> **Note:** Tutorials use synthetic data generated by `data/generators/generate_all.py`.
> Real datasets require separate download and credentialed access (PhysioNet requires training certification).
> See each dataset link above for access instructions.

HFL models: `models/hfl/` | VFL models: `models/vfl/` | LLM models: `models/llm/`

## Step 7: Explore the Code

```
models/hfl/mlp/              HFL model
  server_app.py                Aggregation strategy + model
  client_app.py                Local training + data loading

models/vfl/vfl_mlp/          VFL model
  server_app.py                VFL bottom model + aggregation
  client_app.py                Per-party training on feature subset

tasks/hfl/fraud/data.py      HFL data partitioning (by samples)
psi/                         Entity alignment for VFL
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
