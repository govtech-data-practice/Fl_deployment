# Tutorial 5: Secure Aggregation

**Time:** 15 minutes | **Level:** Intermediate | **Prerequisites:** [Tutorial 4](04-differential-privacy.md)

## What You'll Learn

- What SecAgg protects against (curious server)
- How pairwise masking works
- Configure SecAgg parameters
- Combine SecAgg with DP for layered defence

## Concept: Why SecAgg?

DP protects against inference attacks on the *released model*. SecAgg protects against a *curious server* — even the coordinator cannot see individual client updates. It only sees the aggregate.

```
Without SecAgg:          With SecAgg:
  Client A: Δw_A  ─┐      Client A: Δw_A + mask_A  ─┐
  Client B: Δw_B  ─┤      Client B: Δw_B + mask_B  ─┤
  Client C: Δw_C  ─┘      Client C: Δw_C + mask_C  ─┘
                   │                                   │
  Server sees:             Server sees:
  Δw_A, Δw_B, Δw_C        Only Δw_A + Δw_B + Δw_C
  (individual updates)     (masks cancel out in sum)
```

## Step 1: Run with SecAgg

```bash
python run_ec2.py fraud --synthetic --strategies SecAgg
```

SecAgg uses pairwise deterministic masks that cancel when summed across all clients. The server receives masked updates and can only compute the aggregate — it never sees any individual client's update.

## Step 2: Understand the Configuration

```bash
cat secagg/config.yaml
```

```yaml
scale: 0.01           # Mask magnitude
min_quorum: 2         # Minimum clients needed
max_abort_rate: 0.20  # Alert if >20% of rounds abort
dropout_tolerant: true # Handle client dropout
seed_strategy: sequential
```

Key parameters:
- **`scale`** — controls mask magnitude (too large can affect numerical stability)
- **`min_quorum`** — minimum clients for SecAgg to proceed (fail-closed if fewer)

## Step 3: Test the Masking

```python
from fl_pets.secagg import mask_parameters, verify_cancellation
import numpy as np

# Simulate 3 clients with simple parameters
params = [np.array([1.0, 2.0, 3.0])]

# Each client masks their parameters
masked_0 = mask_parameters(params, client_id=0, num_clients=3, round_seed=42)
masked_1 = mask_parameters(params, client_id=1, num_clients=3, round_seed=42)
masked_2 = mask_parameters(params, client_id=2, num_clients=3, round_seed=42)

# Individual masks look random
print("Client 0:", masked_0[0])  # [1.xx, 2.xx, 3.xx] — noisy
print("Client 1:", masked_1[0])  # [1.xx, 2.xx, 3.xx] — different noise

# But the sum equals the true sum (masks cancel!)
total = masked_0[0] + masked_1[0] + masked_2[0]
print("Sum:     ", total)        # [3.0, 6.0, 9.0] — exact
print("Expected:", params[0] * 3)

# Or verify automatically:
result = verify_cancellation(params, num_clients=3, round_seed=42)
print("Max error:", result["max_error"])  # ~1e-07
```

**Checkpoint:** The sum of masked parameters should equal the sum of unmasked parameters.

## Step 4: Combine DP + SecAgg

For maximum protection, use both:

```bash
# DP adds noise to each client's update
# SecAgg hides individual (noisy) updates from the server
# Server only sees the aggregate of noisy updates
python run_ec2.py fraud --synthetic --strategies SecAgg
```

The order matters:
1. Client clips gradients (DP)
2. Client adds Gaussian noise (DP)
3. Client adds SecAgg mask
4. Server receives masked+noisy updates
5. Server sums (masks cancel, noise remains in aggregate)

## What You Learned

- SecAgg prevents the server from seeing individual model updates
- Pairwise masks cancel when summed — the server only sees the aggregate
- SecAgg and DP are complementary: DP protects the model, SecAgg protects the updates

## Next Steps

- [Tutorial 6: FL Strategies Deep Dive](06-strategies.md) — handle non-IID data
