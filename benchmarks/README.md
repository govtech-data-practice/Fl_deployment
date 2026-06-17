# Benchmarks

Centralised training baselines and FL vs centralised comparison framework.

## Purpose

FL is only justified if it achieves comparable accuracy to centralised training
while keeping data local. This folder provides:

1. **Centralised baselines** — train each model on pooled data (upper bound)
2. **Comparison runner** — runs centralised + FL side-by-side, produces comparison tables
3. **Results** — recorded benchmark outputs for reproducibility

## Structure

```
benchmarks/
  centralized/             Centralised training scripts (one per task)
    train_fraud.py           MLP on pooled fraud data
    train_sepsis.py          BiLSTM on pooled sepsis data
    train_ecg.py             BiLSTM on pooled ECG data
    train_anomaly.py         Autoencoder on pooled anomaly data
    train_mortality.py       TabNet on pooled mortality data
    train_satellite.py       ResNet on pooled satellite data
    train_readmission.py     LogReg on pooled readmission data
    train_drug.py            Generic MLP on pooled drug data
  run_benchmarks.py        Unified runner: centralised vs FL comparison
  results/                 Benchmark output (CSV + summary)
```

## Quick Start

```bash
# Run centralised baseline for a single task
python benchmarks/centralized/train_fraud.py

# Run full comparison: centralised vs FL (all tasks)
python benchmarks/run_benchmarks.py

# Run comparison for specific tasks
python benchmarks/run_benchmarks.py --tasks fraud sepsis ecg
```

## Output

`run_benchmarks.py` produces a comparison table:

```
Task          Centralised     FL (FedAvg)     FL (SCAFFOLD)   FL (DP-Strong)
            Acc    Time      Acc    Time      Acc    Time      Acc    Time
─────────────────────────────────────────────────────────────────────────────
fraud       0.98   12s       0.97   45s       0.97   52s       0.94   48s
sepsis      0.83   25s       0.81   90s       0.82   105s      0.72   95s
ecg         0.91   18s       0.89   65s       0.90   78s       0.81   70s
```
