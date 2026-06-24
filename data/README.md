# Data Directory

Raw datasets, processed samples, and synthetic data generators for FL experiments.

## Structure

```
data/
  raw/                   Raw source datasets
    creditcard_2023_sample_25k.csv  25K sample of CC fraud transactions (committed, 13 MB)
    creditcard_2023.csv             Full 568K dataset (gitignored, download from Kaggle)
    METABRIC_RNA_Mutation.csv       1,904 breast cancer patients (committed, 8 MB)

  samples/               Processed, ready-to-use datasets (not committed to git)
    fraud/                 Real: 568K transactions, 29 features (V1-V28 + Amount)
    cancer/                Real: 1,310 patients, 14 clinical features
    cancer_vfl/            Real: METABRIC split for VFL (clinical + genomic)
    sepsis/                Synthetic: 500 vitals time series (48 steps, 14 features)
    ecg/                   Synthetic: 500 ECG signals (250 steps, 12 leads)
    anomaly/               Synthetic: 500 embeddings (40 features)
    mortality/             Synthetic: 500 ICU records (25 features)
    drug/                  Synthetic: 500 molecular fingerprints (200 features)
    readmission/           Synthetic: 500 discharge records (20 features)
    satellite/             Synthetic: 500 multispectral patches (64x64x3)
    psa/                   Synthetic: Singapore hospital records for PSA testing

  generators/            Data generation scripts
    generate_all.py        Generate synthetic samples for all tasks
    sg_synthetic.py        Singapore patient data generator (PSA testing)

  audit/                 Generation audit trail (JSONL)
```

## Real Datasets

| Dataset | Records | Features | Source | Licence |
|---------|---------|----------|--------|---------|
| **Credit Card Fraud 2023** | 568,630 (25K sample included) | 30 (V1-V28 + Amount + pad) | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) | CC BY 4.0 |
| **METABRIC Breast Cancer** | 1,904 | 693 (31 clinical + 489 RNA + 173 mutations) | [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric) | Open access |

## Quick Start

```bash
# Process raw datasets into samples/ (if raw/ files exist)
python data/generators/generate_all.py --task fraud --num-samples 500

# Generate synthetic data for tasks without real data
for task in sepsis ecg anomaly mortality drug readmission satellite; do
    python data/generators/generate_all.py --task $task --num-samples 500
done

# Use in training
python runners/run_ec2.py fraud --data-dir data/samples/fraud
```

## Each sample directory contains

- `data.npz` — features (X) and labels (y) as numpy arrays
- `manifest.json` — metadata, checksums, label distribution
- `data_card.md` — dataset documentation and provenance
