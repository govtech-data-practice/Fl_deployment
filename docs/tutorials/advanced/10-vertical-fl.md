# Tutorial 10: Vertical FL & PSI

**Time:** 25 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 6](../intermediate/06-strategies.md)

## What You'll Learn

- Difference between HFL and VFL
- Align entities across parties using Private Set Intersection (PSI)
- Run VFL training with vertically partitioned data
- Understand split learning

## Concept: Vertical FL

In HFL, all sites have the same features but different samples. In VFL, sites have different features for the *same* entities:

```
HFL:                            VFL:
  Site A: [f1 f2 f3] [y]         Org A: [f1 f2 f3]
  Site B: [f1 f2 f3] [y]         Org B: [f4 f5 f6]
  Site C: [f1 f2 f3] [y]         Org C: [f7 f8 f9] [y]
  (same features)                (different features, same entities)
```

**Challenge:** Before training, you must align entities across parties — without revealing non-matching records. This is where PSI comes in.

## Step 1: Entity Alignment with PSI

```python
from fl_pets.psi import align_entities
import os

# Each party has a list of pseudonymised identifiers
# (never raw IDs — always pre-hashed)
org_a_ids = ["hash_001", "hash_002", "hash_003", "hash_004"]
org_b_ids = ["hash_002", "hash_003", "hash_005"]
org_c_ids = ["hash_003", "hash_004", "hash_002"]

# Align: find entities present in ALL parties
result = align_entities(
    parties={
        "org_a": org_a_ids,
        "org_b": org_b_ids,
        "org_c": org_c_ids,
    },
    salt=os.urandom(32),
)

print(f"Common entities: {len(result['org_a'])}")
# result["org_a"] = [1, 2]  → indices of matching records in org_a
# result["org_b"] = [0, 1]  → indices of matching records in org_b
# result["org_c"] = [0, 2]  → indices of matching records in org_c
```

**Security:** Only hashed values are exchanged. Non-matching records are never revealed.

## Step 2: Run VFL Training

```bash
python runners/run_ec2.py vfl_fraud --synthetic
```

This runs VFL with 3 parties, each holding a different subset of 30 features:
- Party 0: features 0-9
- Party 1: features 10-19
- Party 2: features 20-29 + labels

The VFL model is in `models/vfl/vfl_mlp/`.

## Step 3: Understand VFL vs HFL Communication

| Aspect | HFL | VFL |
|--------|-----|-----|
| Communication frequency | Per round | Per batch |
| What's shared | Model updates (Δw) | Activations + gradients |
| Data alignment | Not needed | Required (PSI) |
| Label holder | All clients | Usually one party |
| Model architecture | Same model everywhere | Split across parties |

VFL has higher communication cost (per-batch vs per-round) but enables collaboration when features are distributed.

## Step 4: Split Learning

Split learning is a variant where the model is physically split:

```bash
python runners/run_ec2.py split_sepsis --synthetic
```

```
Site (bottom half):    Cloud (top half):
  Raw data             Classifier
    ↓                     ↓
  LSTM encoder   ───>  Dense layers
    ↓                     ↓
  Embeddings           Predictions
  (sent to cloud)      (gradients sent back)
```

The bottom half (data-facing) stays local. Only intermediate activations (embeddings) cross the network boundary. Raw data never leaves the site.

The split model is in `models/vfl/split_bilstm/`.

## What You Learned

- VFL enables collaboration when features are distributed across organisations
- PSI aligns entities without revealing non-matching records
- Split learning keeps the data-facing model on-premise
- VFL has higher communication cost than HFL (per-batch vs per-round)

## Next Steps

- [Tutorial 11: LLM Federated Fine-tuning](11-llm-finetuning.md) — federate large language models
