# Model Card

## Model Details

- **Model name:**
- **Model version:**
- **Model architecture:**
- **Parameters:** (total / trainable)
- **Training framework:** Flower + PyTorch
- **Release date:**
- **Run ID:**

## Intended Use

- **Primary use case:**
- **Target population:**
- **Out-of-scope uses:**

## Training Data

- **Task:**
- **Number of participants:**
- **Total records:** (across all participants)
- **Data period:**
- **Data manifests:** (reference checksums)

## Training Configuration

- **FL strategy:**
- **Rounds:**
- **Local epochs:**
- **Batch size:**
- **Learning rate:**
- **DP preset:**
- **Epsilon achieved:** (at training completion)
- **SecAgg:** enabled / disabled

## Evaluation

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | | |
| AUC | | |
| F1 | | |
| Per-participant variance | | |

## Privacy Testing

| Attack | Metric | Result | Threshold | Pass |
|--------|--------|--------|-----------|------|
| Membership inference | Attack AUC | | < 0.6 | |
| Attribute inference | Inference accuracy | | | |
| Model inversion | Reconstruction SSIM | | | |
| Gradient leakage | Cosine similarity | | < 0.1 | |
| Canary insertion | Extraction rate | | < 0.05 | |

## Fairness

- **Per-participant performance:**
- **Subgroup analysis:**
- **Fairness criteria:**

## Governance

- **Approved by:**
- **Approval date:**
- **Review notes:**
- **Restrictions:**
