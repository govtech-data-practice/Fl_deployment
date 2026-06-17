# Tutorial 4: Differential Privacy

**Time:** 25 minutes | **Level:** Intermediate | **Prerequisites:** [Tutorial 3](../beginner/03-data-pipeline.md)

## What You'll Learn

- What differential privacy (DP) protects against
- How DP presets work (DP_STRONG, DP_MODERATE, DP_RELAXED)
- Measure the accuracy vs privacy trade-off
- Use the privacy budget calculator

## Concept: Why DP?

Without DP, model updates can leak information about training data. An adversary who sees the model updates could:

- **Membership inference** — determine if a specific record was in the training set
- **Gradient inversion** — reconstruct training samples from gradients
- **Canary extraction** — extract planted secrets from the model

DP adds calibrated noise to model updates, bounding how much any single record can influence the output.

## Step 1: Understand DP Presets

The platform provides three named presets:

| Preset | Noise (sigma) | Clipping (C) | Privacy | Accuracy Impact |
|--------|--------------|-------------|---------|----------------|
| `DP_STRONG` | 1.5 | 1.0 | Best | Highest noise |
| `DP_MODERATE` | 0.8 | 1.0 | Good | Moderate noise |
| `DP_RELAXED` | 0.5 | 1.0 | Basic | Lowest noise |

**Fail-closed:** If no preset is specified, `DP_STRONG` is applied automatically.

## Step 2: Compare DP Presets

Run the same task with different privacy levels:

```bash
# No DP (baseline)
python runners/run_ec2.py fraud --synthetic --strategies IID

# DP-Central (server-side, DP_STRONG default)
python runners/run_ec2.py fraud --synthetic --strategies DP-Central

# DP-Local (client-side, DP_MODERATE)
python runners/run_ec2.py fraud --synthetic --strategies DP-Local

# DP-Local with relaxed privacy
python runners/run_ec2.py fraud --synthetic --strategies DP-Local-Low
```

**Checkpoint:** Note the accuracy for each run. You should see:
- `IID` (no DP): highest accuracy (~0.95-0.98)
- `DP-Central`: slightly lower
- `DP-Local`: lower still
- `DP-Local-Low`: between DP-Local and no DP

## Step 3: Calculate Privacy Budget

The privacy budget (epsilon) quantifies the privacy guarantee. Lower epsilon = stronger privacy.

```bash
# Show all presets at 100 rounds
python tools/dp_budget.py --all --rounds 100

# Custom configuration
python tools/dp_budget.py --sigma 1.2 --rounds 50 --delta 1e-5

# See how budget grows with more rounds
python tools/dp_budget.py --preset DP_STRONG --rounds 10
python tools/dp_budget.py --preset DP_STRONG --rounds 50
python tools/dp_budget.py --preset DP_STRONG --rounds 100
python tools/dp_budget.py --preset DP_STRONG --rounds 500
```

**Checkpoint:** Observe that epsilon grows with the number of rounds. This is the privacy "cost" of continued training.

## Step 4: DP Sweep Experiment

Run a systematic sweep to visualise the trade-off:

```bash
# This sweeps sigma from 0.1 to 2.0 and measures accuracy at each level
python -m privacy.sweep_dp
```

**Key insight:** Small models (MLP, BiLSTM) tolerate DP noise well. Large models (DenseNet-121) are more sensitive — their accuracy drops sharply with DP.

## Step 5: How DP Works Under the Hood

The DP mechanism follows this order:

1. **Clip** — bound per-sample gradients to norm ≤ C (limits any one sample's influence)
2. **Noise** — add Gaussian noise N(0, σ²C²) to the clipped update
3. **Account** — track cumulative privacy loss using Renyi DP
4. **SecAgg** — mask the noisy update (if SecAgg enabled — see Tutorial 5)

```python
from fl_pets.dp import compute_epsilon, get_preset, RDPAccountant

# Get DP config (fail-closed to DP_STRONG)
cfg = get_preset("DP_STRONG")
print(f"sigma={cfg['noise_multiplier']}, C={cfg['max_grad_norm']}")

# Compute privacy budget using Opacus RDP accountant
eps = compute_epsilon(noise_multiplier=1.5, sample_rate=0.01, steps=100)
print(f"epsilon after 100 steps (q=0.01): {eps:.4f}")
```

## What You Learned

- DP adds noise to prevent model updates from leaking training data
- Three presets (STRONG/MODERATE/RELAXED) offer different privacy-accuracy trade-offs
- Epsilon grows with training rounds — more training = more privacy cost
- Small models handle DP well; large models are more sensitive

## Next Steps

- [Tutorial 5: Secure Aggregation](05-secure-aggregation.md) — hide individual updates from the server
