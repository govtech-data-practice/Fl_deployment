# Tutorial 7: Privacy Attack Testing

**Time:** 25 minutes | **Level:** Intermediate | **Prerequisites:** [Tutorial 6](06-strategies.md)

## What You'll Learn

- Run privacy attacks against trained FL models
- Interpret attack metrics (AUC, cosine similarity, extraction rate)
- Compare attack success with and without DP
- Understand what each attack reveals

## Concept: Why Attack Your Own Model?

Privacy testing verifies that your defences actually work. The deployment guide requires five attack types before model release:

| Attack | What It Tests | Metric | Pass Threshold |
|--------|--------------|--------|----------------|
| Membership inference | Can an adversary tell if a record was in training? | Attack AUC | < 0.6 |
| Attribute inference | Can an adversary infer sensitive attributes? | Accuracy | Task-specific |
| Model inversion | Can an adversary reconstruct training inputs? | Cosine similarity | < 0.1 |
| Gradient leakage | Can an adversary reconstruct samples from gradients? | Cosine similarity | < 0.1 |
| Canary insertion | Can planted secrets be extracted from the model? | Extraction rate | < 0.05 |

## Step 1: Run the Privacy Attack Suite

```bash
python runners/run_ec2.py privacy --synthetic
```

This runs membership inference and gradient leakage attacks on a fraud model, comparing results with and without DP.

## Step 2: Membership Inference Attack (MIA)

MIA checks if an adversary can determine whether a specific record was used for training.

```python
# The attack trains a binary classifier:
# Input: model's loss on a sample
# Output: "member" (in training set) or "non-member"

# Attack AUC = 0.5 means the attack is no better than random (good!)
# Attack AUC = 1.0 means perfect membership detection (bad!)
```

**Interpreting results:**
- AUC < 0.55: strong privacy (DP is working)
- AUC 0.55-0.65: acceptable for most use cases
- AUC > 0.65: consider increasing DP noise

## Step 3: Gradient Leakage Attack (DLG)

DLG attempts to reconstruct training samples from observed gradients.

```python
# The attack:
# 1. Observes the gradient of a training batch
# 2. Optimises a dummy input to produce the same gradient
# 3. Measures how close the reconstruction is to the real data
```

**Interpreting results:**
- Cosine similarity < 0.1: reconstruction failed (good!)
- Cosine similarity > 0.5: significant leakage (needs stronger DP or SecAgg)

## Step 4: Model Inversion Attack

```python
from privacy.model_inversion import run_model_inversion_evaluation

# Run against a trained model
results = run_model_inversion_evaluation(
    model=trained_model,
    X_train=X_train,
    y_train=y_train,
    num_classes=2,
    device="cpu"
)

print(f"Max cosine similarity: {results['max_cosine_similarity']:.3f}")
print(f"Mean confidence: {results['mean_confidence']:.3f}")
```

## Step 5: Compare With and Without DP

Run the attack battery twice — once without DP, once with:

```bash
# Without DP
python runners/run_ec2.py fraud --synthetic --strategies IID
# Then run attacks on the saved model

# With DP
python runners/run_ec2.py fraud --synthetic --strategies DP-Central
# Then run attacks on the DP-protected model
```

**Expected observation:**
- MIA AUC drops from ~0.6-0.7 to ~0.5-0.55 with DP
- Gradient leakage cosine similarity drops below 0.1 with DP
- LLM canary extraction drops from ~41.7% to ~16.7% with DP

## Step 6: Use the Model Card Template

After privacy testing, document results in a model card:

```bash
cat templates/model_card.md
```

The model card includes a privacy testing table where you record each attack's result against the pass threshold.

## What You Learned

- Privacy attacks are a required step before model release
- Five attack types cover different threat scenarios
- DP significantly reduces attack success rates
- Results should be documented in the model card template

## Next Steps

You've completed intermediate tutorials. Move on to:

- [Tutorial 8: Distributed Deployment](../advanced/08-distributed-deployment.md) — deploy to real infrastructure
