# Tutorial 2: Your First Model

**Time:** 20 minutes | **Level:** Beginner | **Prerequisites:** [Tutorial 1](01-setup.md)

## What You'll Learn

- Run different FL tasks (healthcare, finance, geospatial)
- Understand what models and strategies are available
- Compare strategies on the same task

## Step 1: Try Different Tasks

Each task uses a different model architecture. Run a few:

```bash
# Time-series: sepsis early warning (BiLSTM)
python runners/run_ec2.py sepsis --synthetic

# Signal processing: ECG arrhythmia (BiLSTM)
python runners/run_ec2.py ecg --synthetic

# Unsupervised: anomaly detection (Autoencoder)
python runners/run_ec2.py anomaly --synthetic

# Imaging: satellite land-use classification (ResNet-small)
python runners/run_ec2.py satellite --synthetic
```

**Checkpoint:** Each task should complete with accuracy/AUC metrics reported.

## Step 2: Understand the Model-Task Mapping

| Task | Model | Architecture | Parameters | Data Type |
|------|-------|-------------|-----------|-----------|
| `fraud` | MLP | 3-layer feedforward | 50K | Tabular |
| `sepsis` | BiLSTM | Bidirectional LSTM | 500K | Time-series |
| `ecg` | BiLSTM | Bidirectional LSTM | 200K | Time-series |
| `anomaly` | Autoencoder | Encoder-decoder | 500K | Tabular |
| `mortality` | TabNet | Attention-based | 1M | Tabular |
| `readmission` | LogReg | Logistic Regression | 10K | Tabular |
| `satellite` | ResNet-small | Residual CNN | 5M | Images |
| `chest_xray` | DenseNet-121 | Dense CNN | 8M | Images |

Models live in `models/hfl/`, tasks in `tasks/hfl/`.

## Step 3: Compare FL Strategies

Run fraud detection with different strategies:

```bash
# Standard FedAvg
python runners/run_ec2.py fraud --synthetic --strategies IID

# FedProx (handles non-IID data better)
python runners/run_ec2.py fraud --synthetic --strategies FedProx

# SCAFFOLD (variance reduction)
python runners/run_ec2.py fraud --synthetic --strategies SCAFFOLD
```

Compare the accuracy and convergence speed across strategies.

## Step 4: Use a Scenario File

Instead of command-line flags, you can define experiments in YAML:

```bash
cat scenarios/quick_fraud.yaml
```

```yaml
name: "Quick Fraud Demo"
task: fraud
num_clients: 3
num_rounds: 3
strategies:
  - "IID"
synthetic: true
max_samples: 2000
```

Run all strategies at once:

```bash
python runners/run_ec2.py fraud --synthetic --strategies all
```

## Step 5: Explore the Code

Look at how a model is structured:

```
models/hfl/mlp/
  __init__.py
  server_app.py    # Strategy factory, model initialisation
  client_app.py    # Local training loop, data loading
```

Key concepts:
- **`server_app.py`** defines the aggregation strategy and global model
- **`client_app.py`** defines local training (what each participant runs)
- **`tasks/hfl/<task>/data.py`** handles data loading and partitioning

## What You Learned

- The platform supports 10+ tasks across healthcare, finance, and geospatial domains
- Each task is paired with an appropriate model architecture
- FL strategies (FedAvg, FedProx, SCAFFOLD) handle different data distribution scenarios
- Experiments can be configured via CLI flags or scenario YAML files

## Next Steps

- [Tutorial 3: Data Pipeline](03-data-pipeline.md) — ingest your own data
