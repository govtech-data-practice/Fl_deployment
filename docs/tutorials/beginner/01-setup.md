# Tutorial 1: Setup & First Run

**Time:** 20 minutes | **Level:** Beginner | **Requirements:** Python 3.10+, 4 GB RAM

## What You'll Learn

- Install the FL platform and its dependencies
- Verify your environment is ready
- Train a simple model and run inference
- Run the test suite

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

## Step 4: Train a Model and Run Inference

Train a simple fraud detection MLP and test it:

```python
import torch
import torch.nn as nn
import numpy as np

# Load synthetic data
data = np.load("data/samples/fraud/data.npz")
X = torch.from_numpy(data["X"])
y = torch.from_numpy(data["y"])

print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Label distribution: {y.mean():.2f} positive rate")

# Build model (same architecture as models/hfl/mlp/)
model = nn.Sequential(
    nn.Linear(30, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 1), nn.Sigmoid(),
)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for epoch in range(10):
    model.train()
    pred = model(X).squeeze()
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    acc = ((pred > 0.5).float() == y).float().mean()
    print(f"  Epoch {epoch+1:2d}: loss={loss.item():.4f}  accuracy={acc.item():.4f}")

# Inference — predict on new samples
model.eval()
with torch.no_grad():
    test_samples = torch.randn(5, 30)  # 5 new transactions
    predictions = model(test_samples).squeeze()
    
    print("\nInference on 5 new transactions:")
    for i, (pred, score) in enumerate(zip(predictions > 0.5, predictions)):
        label = "FRAUD" if pred else "legitimate"
        print(f"  Transaction {i}: {label} (score={score.item():.4f})")
```

## Step 5: Run the Test Suite

Verify everything works end-to-end:

```bash
# Run strategy tests (fraud, sepsis)
python tests/run_tests.py fraud

# Run FL smoke test
python runners/run_ec2.py fraud --synthetic
```

**Expected:** Both commands complete without errors.

## Step 6: Validate the PET Toolkit

Quick check that all privacy-enhancing technology modules load:

```python
# Differential Privacy (Opacus)
from fl_pets.dp import compute_epsilon, DP_PRESETS
eps = compute_epsilon(noise_multiplier=1.5, sample_rate=0.01, steps=100)
print(f"DP: epsilon={eps:.4f} at 100 steps (sigma=1.5)")

# Secure Aggregation (Flower)
from fl_pets.secagg import verify_cancellation
import numpy as np
result = verify_cancellation([np.array([1.0, 2.0])], num_clients=3, round_seed=42)
print(f"SecAgg: masks cancel with error {result['max_error']:.2e}")

# Homomorphic Encryption (TenSEAL)
from fl_pets.he import create_context, encrypt, decrypt
ctx = create_context()
enc = encrypt(ctx, [3.14, 2.71])
dec = decrypt(enc + enc)
print(f"HE: encrypted [3.14, 2.71] * 2 = [{dec[0]:.4f}, {dec[1]:.4f}]")

print("\nAll PET modules verified.")
```

## What Just Happened?

You:
1. **Installed** the FL platform with all dependencies
2. **Generated** synthetic fraud detection data
3. **Trained** a model and **ran inference** on new samples
4. **Tested** the FL pipeline and PET toolkit

This confirms your environment is ready for the remaining tutorials.

## Next Steps

- [Tutorial 2: Your First Model](02-first-model.md) — train a centralised baseline, then compare with FL
