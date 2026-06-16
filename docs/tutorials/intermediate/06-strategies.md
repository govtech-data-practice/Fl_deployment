# Tutorial 6: FL Strategies Deep Dive

**Time:** 30 minutes | **Level:** Intermediate | **Prerequisites:** [Tutorial 5](05-secure-aggregation.md)

## What You'll Learn

- Why data heterogeneity (non-IID) matters in FL
- When to use FedProx, SCAFFOLD, or adaptive optimisers
- Run strategy comparison experiments
- Interpret convergence behaviour

## Concept: Non-IID Data

Real-world federated data is rarely uniform. Common heterogeneity patterns:

| Pattern | Example | Impact |
|---------|---------|--------|
| **Label skew** | Hospital A sees mostly cardiac; B sees respiratory | Client models diverge |
| **Quantity skew** | Large hospital: 50K records; small clinic: 2K | Under-represented clients |
| **Feature skew** | Different labs available at different sites | Missing features |

FedAvg works well with IID data but can diverge with severe non-IID distributions.

## Step 1: Strategy Comparison

Run the strategy showdown scenario:

```bash
# This runs multiple strategies on the same data
python run_ec2.py fraud --synthetic --strategies all
```

## Step 2: Understand Each Strategy

### FedAvg (Baseline)
- Simple weighted averaging of client models
- Works well with IID data
- Can diverge with non-IID data

### FedProx
- Adds a proximal term (mu=0.01) to the client loss
- Penalises client models that drift too far from the global model
- Better than FedAvg for moderate non-IID

```bash
python run_ec2.py fraud --synthetic --strategies FedProx
```

### SCAFFOLD
- Uses control variates to correct for client drift
- Each client maintains a correction term
- Best theoretical convergence for non-IID data

```bash
python run_ec2.py fraud --synthetic --strategies SCAFFOLD
```

### FedAdam / FedYogi
- Server-side adaptive optimisers
- Good for tasks with sparse gradients
- Can diverge on pretrained models (use with caution)

```bash
python run_ec2.py fraud --synthetic --strategies FedAdam
```

## Step 3: Strategy Selection Guide

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| IID data, quick test | FedAvg | Simplest, fastest |
| Mild non-IID | FedProx | Proximal term prevents drift |
| Severe non-IID | SCAFFOLD | Control variates correct drift |
| Privacy required | SecAgg + DP-Central | Layered defence |
| Pretrained model | FedAvg | Adaptive optimisers can diverge |
| Byzantine/poisoning | Trimmed Mean, Krum | Robust aggregation (v0.2) |

## Step 4: Non-IID Experiment

Run the non-IID impact scenario:

```bash
cat scenarios/noniid_impact.yaml
```

This sweeps Dirichlet alpha from 0.1 (severe non-IID) to 1.0 (nearly IID) to measure strategy robustness.

## Step 5: Key Findings from Validated Experiments

| Finding | Detail |
|---------|--------|
| FedAvg beats centralised | Chest X-ray AUC 0.819 > centralised 0.803 (implicit regularisation) |
| SCAFFOLD best for non-IID | Sepsis with 5 clients: SCAFFOLD converges fastest |
| FedAdam/Yogi diverge on pretrained | Use FedAvg+SCAFFOLD for transfer learning |
| DP impact varies by model size | Small models (BiLSTM): tolerable. Large (DenseNet): severe |

## What You Learned

- Non-IID data is the main challenge in real-world FL
- FedProx and SCAFFOLD address different types of data heterogeneity
- Strategy choice depends on data distribution, model size, and privacy requirements
- Always benchmark multiple strategies on your specific data

## Next Steps

- [Tutorial 7: Privacy Attack Testing](07-privacy-attacks.md) — verify your defences
