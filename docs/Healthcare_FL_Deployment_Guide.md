# Federated Learning & Privacy-Enhancing Technologies: Deployment Guide

## 1. Executive Summary

This guide covers the deployment of privacy-preserving federated learning (FL) systems across industries. FL enables organisations to collaboratively train models without sharing raw data — critical when data is distributed across jurisdictions, devices, or organisations that cannot or will not centralise it.

**Why FL matters — government and public sector use cases:**

| Domain | Why data can't be centralised | Scale | FL impact |
|--------|------------------------------|-------|-----------|
| **Healthcare (public hospitals)** | Patient privacy (HIPAA/PDPA), data stays in institutions | 1000s of hospitals | Multi-site clinical models without patient data sharing |
| **Defence/Intelligence** | Classified data across agencies, compartmentalisation | Multiple agencies/nations | Intelligence sharing without cross-clearance access |
| **Border Security/Immigration** | Biometric data across countries, sovereignty | Multiple nations | Shared threat detection without sharing citizen data |
| **Public Health/Epidemiology** | Disease surveillance across jurisdictions | National/regional agencies | Outbreak prediction without centralising health records |
| **Smart City/Transport** | Traffic, sensor, CCTV data across municipalities | 1000s of sensors/cameras | Traffic prediction, safety models across districts |
| **Tax/Revenue** | Taxpayer data across agencies, legal restrictions | Millions of records | Cross-agency fraud detection |
| **Education** | Student records across institutions, FERPA | 1000s of schools | Learning outcome prediction without sharing student data |
| **Telecommunications (regulated)** | Network data, national security, lawful intercept | Billions of records | Network threat detection across carriers |
| **Energy/Utilities (critical infra)** | Smart meter data, SCADA, critical infrastructure | 100M+ meters | Demand forecasting, grid anomaly detection |
| **Finance (regulated)** | Bank secrecy, AML across institutions | 100s of banks | Cross-bank fraud detection, regulatory compliance |

**Key capabilities of this system:**
- Federated learning across distributed GPU nodes with TLS encryption
- 9 aggregation strategies (FedAvg, FedProx, FedAdam, FedYogi, SCAFFOLD, SecAgg+, DP-Central, DP-Local, OneOwner)
- Privacy-Enhancing Technologies: Differential Privacy, Secure Aggregation, Federated LoRA
- Privacy attack suite: Membership Inference, Canary Extraction, Gradient Inversion
- Parameter-Efficient Fine-Tuning (PEFT) of Mistral 7B with QLoRA

**Validated results (healthcare benchmark):**
- Chest X-ray (NIH, 112K images): FedAvg AUC = 0.819 (centralized baseline: 0.803)
- Sepsis prediction (eICU, 100K+ samples): FedAvg accuracy = 0.806
- Federated LoRA on Mistral 7B: 160 MB adapter payload vs 14 GB full model

---

## 2. Architecture Overview

### 2.1 System Components

```
                        +------------------+
                        |   SuperLink      |
                        |   (FL Server)    |
                        |   TLS :9092      |
                        +--------+---------+
                                 |
              +------------------+------------------+
              |                  |                  |
     +--------+------+  +-------+-------+  +-------+-------+
     |  SuperNode 0  |  |  SuperNode 1  |  |  SuperNode N  |
     |  Hospital A   |  |  Hospital B   |  |  Hospital N   |
     |  (GPU client) |  |  (GPU client) |  |  (GPU client) |
     +---------------+  +---------------+  +---------------+
```

### 2.2 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| FL Framework | Flower (flwr) | 1.25+ |
| Deep Learning | PyTorch | 2.5+ |
| LLM Fine-tuning | PEFT / QLoRA | 0.17+ |
| Container Runtime | Docker | 25.0+ |
| Infrastructure | AWS EC2 (g6.xlarge+) | - |
| IaC | Terraform | 1.5+ |
| TLS | Self-signed CA + server certs | EC P-256 |
| GPU | NVIDIA L4 (24GB) | Driver 570+ |

### 2.3 Repository Structure

```
healthcare-fl/
+-- fl_common/                  Shared FL code
|   +-- strategies.py           All 9 strategies
|   +-- scaffold.py             SCAFFOLD index mapping
|   +-- secagg.py               Pairwise mask protocol
|   +-- dp.py                   DP primitives + RDP accountant
|
+-- sepsis/                     Sepsis prediction (BiLSTM)
|   +-- server_app.py           ServerApp
|   +-- client_app.py           ClientApp
|   +-- partition_utils.py      NPZ data partitioning
|
+-- chest_xray/                 Chest X-ray classification (DenseNet-121)
|   +-- server_app.py           ServerApp
|   +-- client_app.py           ClientApp (real + synthetic modes)
|   +-- partition_utils.py      Patient-level partitioning
|
+-- peft/                       PEFT / LoRA
|   +-- train_mistral.py        Mistral 7B QLoRA fine-tuning
|   +-- fed_mistral_privacy.py  Federated LoRA + MIA
|   +-- attack_suite.py         Full privacy attack battery
|   +-- demo_fed_lora.py        Interactive demo
|
+-- experiments/                Benchmarks
|   +-- full_run.py             Full benchmark (300 rounds, early stop)
|   +-- centralized_baseline.py Centralized comparison
|
+-- deploy/                     Infrastructure
|   +-- gen_certs.sh            TLS certificate generation
|   +-- distributed/            Multi-node Docker deployment
|   +-- terraform/              AWS IaC
|
+-- Dockerfile                  Unified container image
+-- docker-compose.test.yml     Local simulation testing
+-- docker-compose.deploy.yml   TLS distributed deployment
+-- run_tests.py                Unified test runner
```

---

## 3. Privacy-Enhancing Technologies (PETs)

### 3.1 Implemented PETs

#### 3.1.1 Secure Aggregation (SecAgg+)

**What it does:** Prevents the server from seeing individual client model updates. Each client masks their update with pairwise random masks that cancel when summed.

**How it works:**
1. For each pair of clients (i, j), a shared seed generates mask M
2. Client with smaller ID adds +M, larger adds -M
3. Server sums all masked updates: masks cancel, leaving clean aggregate

**Deployment:**
```python
# Experiment name format
"SecAgg_Alpha_0.5"

# The strategy enforces all clients participate (masks only cancel with full participation)
# Uses equal-weight averaging (1/N) — required for exact cancellation
# Mask scale: 0.01 (balances security vs float32 precision)
```

**Limitations:**
- All N clients must participate every round (no dropout tolerance)
- Equal-weight averaging only (not weighted by dataset size)
- Validated AUC: 0.763 (vs FedAvg 0.811) — the equal-weighting gap

**When to use:** When the FL server is untrusted and you need to prevent it from inspecting individual hospital updates.

#### 3.1.2 Differential Privacy (DP)

**What it does:** Adds calibrated noise to model updates, providing a mathematical privacy guarantee (epsilon-delta).

**Two modes:**

| Mode | Where noise is added | Trust model |
|------|---------------------|-------------|
| Central DP | Server clips + noises the aggregate | Server is trusted |
| Local DP | Each client clips + noises before sending | Server is untrusted |

**Deployment:**
```python
# Central DP with epsilon=50, clip norm=5
"DP_Central_Eps50.0_Clip5.0_Alpha_0.5"

# Local DP
"DP_Local_Eps50.0_Clip5.0_Alpha_0.5"

# sigma = 1/epsilon (noise multiplier)
# Clip must match model update magnitude (~5 for DenseNet, ~1 for BiLSTM)
```

**Privacy accounting (RDP):**
```python
from fl_common.dp import PrivacyAccountant
accountant = PrivacyAccountant(noise_multiplier=0.02, sample_rate=1.0, delta=1e-5)
accountant.step(num_steps=10)  # 10 rounds
print(f"epsilon = {accountant.get_epsilon():.2f}")
```

**Limitations:**
- Large models (8M+ params) need epsilon >> 100 for meaningful learning
- Works well on small models (BiLSTM: 0.65 accuracy with DP)
- Does NOT work on DenseNet-121 at useful privacy levels (AUC = 0.50)

**When to use:** When you need formal privacy guarantees and can tolerate utility loss. Best for small models or large datasets.

#### 3.1.3 SCAFFOLD (Control Variates)

**What it does:** Corrects for client drift in non-IID settings using control variates that track the difference between local and global gradient estimates.

**How it works:**
1. Server maintains control variate c (updated each round)
2. Each client maintains local control variate c_i
3. During training, gradients are corrected: grad += (c - c_i)
4. Uses trainable-to-state-dict index mapping for models with BatchNorm

**Deployment:**
```python
"SCAFFOLD_Alpha_0.5"
```

**Validated results:** AUC = 0.762 on chest X-ray (stable across IID/non-IID).

#### 3.1.4 Federated LoRA (Parameter-Efficient FL)

**What it does:** Instead of sharing full model weights (14 GB for Mistral 7B), only LoRA adapter weights are shared (160 MB). Reduces communication 90x while preserving privacy.

**How it works:**
1. Each hospital fine-tunes only LoRA matrices (1.1% of params)
2. Adapter weights extracted: `{k: v for k, v in model.named_parameters() if "lora" in k}`
3. Server aggregates adapters via FedAvg
4. Adapters sent back to clients and loaded

**Deployment:**
```bash
# Generate clinical data
python peft/gen_clinical_data.py

# Run federated LoRA with privacy test
python peft/fed_mistral_privacy.py

# Interactive demo (3 hospitals, 5 rounds)
python peft/demo_fed_lora.py --clients 3 --rounds 5
```

**Validated results:**
- FL LoRA payload: 160 MB/round (vs 14 GB full model)
- QA score: 0.167 (FL) vs 0.367 (FL+DP)
- MIA advantage: 1.000 (FL) vs 0.833 (FL+DP)

### 3.2 Additional PETs (Reference)

#### 3.2.1 Trusted Execution Environments (TEE)

**What it does:** Hardware-isolated enclaves where code and data are protected from the host OS, hypervisor, and even physical access. The server aggregates model updates inside the enclave — even a compromised server admin cannot inspect individual updates.

**Platforms:**
| Platform | Provider | Instance types | Attestation |
|----------|----------|---------------|-------------|
| AWS Nitro Enclaves | AWS | Most .metal, c5.xlarge+ | Nitro attestation document |
| AMD SEV-SNP | AWS/GCP | c6a, m6a (AWS); N2D (GCP) | vTPM + SEV firmware |
| Intel SGX | Azure | DCsv3 series | DCAP remote attestation |
| GCP Confidential Space | GCP | N2D + Confidential VM | Attestation verifier token |

**How to deploy with FL:**
1. Build FL server as an enclave image (.eif for Nitro, Docker for Confidential Space)
2. Aggregation runs inside the enclave
3. Clients verify enclave attestation before sending updates
4. Model updates are encrypted in transit (TLS) and at rest (enclave memory encryption)

**When to use:** When the FL server operator must be provably unable to access individual updates. Required for regulated environments where "trust but verify" is insufficient.

**Limitation:** Enclave memory is limited (e.g., Nitro: ~16 GB). Large models may not fit.

#### 3.2.2 Homomorphic Encryption (HE)

**What it does:** Allows computation on encrypted data without decryption. Clients encrypt their model updates; the server aggregates encrypted updates; the result is decrypted only by the clients.

**Schemes:**
| Scheme | Operations | Libraries | Use case |
|--------|-----------|-----------|----------|
| Partially HE (Paillier) | Addition only | python-paillier | Summing encrypted model updates |
| Somewhat HE (BFV/BGV) | Add + limited multiply | TenSEAL, SEAL | Simple aggregation |
| Fully HE (CKKS) | Add + multiply (approx.) | OpenFHE, Lattigo | Complex operations on encrypted data |

**Practical considerations:**
- Ciphertext expansion: 10-1000x larger than plaintext
- Computation: 1000-10000x slower than plaintext
- Best used for simple aggregation (FedAvg = sum), not training

**When to use:** When even the encrypted form of model updates is considered sensitive and SecAgg's trust assumptions are insufficient.

#### 3.2.3 Synthetic Data Generation

**What it does:** Generates artificial data that preserves statistical properties of the original without containing real patient records.

**Methods:**
| Method | Fidelity | Privacy | Tools |
|--------|---------|---------|-------|
| **DP-Synthetic (DPGAN, DP-CTGAN)** | Medium | Formal DP guarantee | Gretel, SmartNoise |
| **Variational Autoencoders** | High | No formal guarantee | SDV (Synthetic Data Vault) |
| **Copula-based** | Medium | Statistical disclosure control | Synthpop (R) |
| **LLM-generated** | High for text | No guarantee | GPT/Mistral with prompts |
| **Diffusion models** | Very high for images | No guarantee | MedSyn, RoentGen |

**When to use:** When data cannot leave the hospital at all (even as model updates). Generate synthetic data locally, validate utility, share synthetic data for model development.

**Limitation:** Utility gap between synthetic and real data. Must validate downstream task performance.

#### 3.2.4 K-Anonymity, L-Diversity, T-Closeness

**What it does:** Transforms structured data so that individuals cannot be re-identified.

| Technique | Guarantee | Method |
|-----------|----------|--------|
| K-Anonymity | Each record is indistinguishable from k-1 others | Generalise quasi-identifiers (age ranges, zip code prefixes) |
| L-Diversity | Each equivalence class has l distinct sensitive values | Prevents attribute disclosure |
| T-Closeness | Distribution of sensitive attribute in each class is within t of overall distribution | Prevents skewness attack |

**Tools:** ARX (Java), sdcMicro (R), Amnesia

**When to use:** For structured tabular data sharing (demographics, lab values, diagnoses). Not applicable to images or model weights.

#### 3.2.5 Federated Distillation / Knowledge Transfer

**What it does:** Instead of sharing model weights, clients share predictions (soft labels) on a public dataset. The server trains a global model from these predictions.

**How it works:**
1. Server distributes a public (non-sensitive) dataset to all clients
2. Each client runs inference on the public data using their local model
3. Clients send predicted logits (not weights) to server
4. Server trains a student model from the ensemble of predictions

**Advantages over standard FL:**
- Clients never share model weights (architecture can be private)
- Communication is proportional to public dataset size, not model size
- Heterogeneous client models are supported

**When to use:** When model architecture is proprietary, or when clients have different model architectures.

#### 3.2.6 Split Learning

**What it does:** The model is split at a "cut layer" — clients hold the bottom layers, server holds the top layers. Neither party has the full model.

**How it works:**
1. Client processes input through bottom layers, sends activations to server
2. Server processes activations through top layers, computes loss
3. Server sends gradient of cut layer back to client
4. Client updates bottom layers; server updates top layers

**Variants:**
| Variant | Who has what | Communication |
|---------|-------------|---------------|
| Vanilla split | Client: bottom, Server: top | Activations + gradients per batch |
| U-shaped split | Client: bottom+top, Server: middle | Activations both directions |
| Split + FL | Multiple clients split-train, server aggregates | Combines split + federated |

**When to use:** When clients have limited compute (mobile, edge devices) and cannot train the full model locally.

**Limitation:** Activation sharing can leak information. Gradient attacks on the cut layer are possible.

#### 3.2.7 Secure Multi-Party Computation (SMPC)

**What it does:** Multiple parties jointly compute a function over their inputs without revealing individual inputs to each other. No trusted third party needed.

**Protocols:**
| Protocol | Parties | Communication | Libraries |
|----------|---------|--------------|-----------|
| Secret Sharing (Shamir) | 3+ | O(n^2) per op | MP-SPDZ, CrypTen |
| Garbled Circuits | 2 | High constant | EMP-toolkit |
| Oblivious Transfer | 2 | Moderate | libOTe |

**FL application:** Clients secret-share their updates. The server(s) compute the aggregate without seeing individual shares. More robust than SecAgg (handles dropout via Shamir threshold).

**When to use:** When SecAgg's all-or-nothing participation is too restrictive, and you need dropout tolerance with cryptographic privacy.

#### 3.2.8 Federated Analytics (without ML)

**What it does:** Compute aggregate statistics (counts, histograms, means) across hospitals without sharing individual records. No model training involved.

**Examples:**
- How many patients across all hospitals have condition X?
- What is the average length of stay for diagnosis Y?
- What is the distribution of lab values across sites?

**Implementation:** SecAgg + DP applied to simple aggregation queries.

**When to use:** For population health studies, clinical trial feasibility, epidemiological surveillance — where you need statistics, not models.

### 3.3 Healthcare Data Types and Recommended PETs

Each healthcare data type has different privacy risks, regulatory requirements, and compatible PETs.

#### 3.3.1 Structured Clinical Data (EHR/EMR)

**Examples:** Patient demographics, vital signs, lab results, diagnoses (ICD codes), medications, procedures

**Privacy risks:**
- Direct identifiers (name, DOB, MRN) — must be removed before any sharing
- Quasi-identifiers (age, zip, admission date) — can be linked to external data
- Sensitive attributes (HIV status, mental health, substance abuse) — special protection required

**Recommended PETs:**

| Scenario | PET Stack | Why |
|----------|----------|-----|
| Cross-hospital ML on vitals/labs | **FL (FedAvg) + DP** | Tabular data works well with DP; small models tolerate noise |
| Aggregate statistics only | **Federated Analytics + DP** | No model needed; formal privacy guarantee |
| Data sharing for research | **K-Anonymity + Synthetic Data** | Researchers need data access, not model |
| Regulatory audit | **FL + DP + TEE** | Verifiable computation + formal epsilon |

#### 3.3.2 Medical Images (X-ray, CT, MRI, Pathology)

**Examples:** Chest X-rays, CT scans, histopathology slides, retinal images, mammograms

**Privacy risks:**
- Embedded metadata (DICOM headers contain patient info)
- Facial reconstruction from head CTs/MRIs
- Rare conditions identifiable by image features alone
- Large file sizes (100MB+ per study) make communication expensive

**Recommended PETs:**

| Scenario | PET Stack | Why |
|----------|----------|-----|
| Multi-site classification model | **FL (FedAvg) + SecAgg** | Large models; DP destroys image features; SecAgg hides updates |
| Single-owner model | **FL (OneOwner) + SecAgg** | Hospitals contribute but don't get the model |
| Communication-constrained | **FL + LoRA** | Share adapter weights (160MB) not full model (GB+) |
| Image sharing for annotation | **De-identification + Synthetic** | Remove DICOM headers, generate synthetic variants |

#### 3.3.3 Clinical Notes / Free Text (NLP)

**Examples:** Discharge summaries, radiology reports, progress notes, pathology reports, consult notes

**Privacy risks:**
- **Highest risk data type** — notes contain names, dates, diagnoses in free text
- LLMs memorize training text verbatim (canary extraction: 41.7% leak rate)
- De-identification is imperfect (regex misses unusual name formats)
- Context can re-identify even with names removed ("the 47yo senator admitted after...")

**Recommended PETs:**

| Scenario | PET Stack | Why |
|----------|----------|-----|
| Federated LLM fine-tuning | **FL LoRA + DP** | Only share adapter weights + noise; MIA drops from 1.0 to 0.83 |
| Clinical NLP model | **FL (FedAvg) + DP + De-identification** | Remove PHI first, then FL with DP as defense in depth |
| Note generation/summarisation | **Local fine-tuning only** | Do not federate — memorisation risk too high |
| Research on clinical text | **Synthetic note generation** | Use LLM to generate synthetic notes preserving clinical patterns |

#### 3.3.4 Genomic / Molecular Data

**Examples:** Whole genome sequences, SNP arrays, gene expression, proteomics, metabolomics

**Privacy risks:**
- **Uniquely identifying** — genome is a permanent identifier
- Family members share genetic variants (privacy extends beyond the individual)
- Re-identification from as few as 75-100 SNPs
- GINA (US) and GDPR special category protections

**Recommended PETs:**

| Scenario | PET Stack | Why |
|----------|----------|-----|
| GWAS across hospitals | **Federated Analytics + DP** | Aggregate allele frequencies with noise |
| Genomic ML model | **FL + DP + TEE** | Maximum protection required; TEE verifies computation |
| Pharmacogenomics | **FL + Secure Aggregation** | Drug response prediction without sharing genomes |
| Variant sharing | **Beacon protocol + DP** | Yes/no query ("does your dataset contain variant X?") with noise |

#### 3.3.5 Wearable / IoT / Continuous Monitoring Data

**Examples:** ECG waveforms, accelerometer, continuous glucose monitors, SpO2 time series, sleep data

**Privacy risks:**
- High temporal resolution can reveal location, activity, identity
- Biometric identification from ECG waveforms
- Longitudinal data reveals patterns even after anonymisation

**Recommended PETs:**

| Scenario | PET Stack | Why |
|----------|----------|-----|
| Arrhythmia detection across sites | **FL (FedAvg) + DP** | Small models (LSTM/CNN) work well with DP |
| Real-time monitoring | **Split Learning** | Edge device has limited compute; split at early layers |
| Wearable data aggregation | **Federated Analytics + Local DP** | Device adds noise before upload |

#### 3.3.6 Administrative / Claims Data

**Examples:** Insurance claims, billing codes, hospital admissions, readmission records, cost data

**Privacy risks:**
- Quasi-identifier linkage (hospital + date + procedure = unique)
- Longitudinal claims reveal full medical history
- Less regulated than clinical data in some jurisdictions

**Recommended PETs:**

| Scenario | PET Stack | Why |
|----------|----------|-----|
| Readmission prediction | **FL (FedAvg) + DP** | Tabular data, small models |
| Cost benchmarking | **Federated Analytics** | Aggregate statistics only |
| Fraud detection | **FL + SecAgg** | Hide individual hospital patterns |
| Data sharing | **K-Anonymity + T-Closeness** | Well-suited to structured claims data |

### 3.4 PET Comparison Matrix

| PET | Protects data at rest | Protects data in transit | Protects during computation | Formal guarantee | Works with large models | Handles dropout |
|-----|------|------|------|------|------|------|
| **Federated Learning** | Yes (data stays local) | N/A (only updates shared) | No | No | Yes | Yes |
| **Differential Privacy** | No | No | Yes (bounds info leakage) | Yes (epsilon-delta) | No (noise too high) | Yes |
| **Secure Aggregation** | No | Yes (masks in transit) | Partially (server sees aggregate) | Yes (info-theoretic) | Yes | No |
| **TEE** | Yes (enclave memory) | Yes (attestation) | Yes (isolated execution) | Hardware-dependent | Limited by enclave memory | Yes |
| **Homomorphic Encryption** | Yes (encrypted at rest) | Yes (encrypted in transit) | Yes (compute on ciphertext) | Yes (crypto hardness) | No (10000x overhead) | Yes |
| **Synthetic Data** | Yes (no real data leaves) | N/A | N/A | Optional (DP-synthetic) | N/A | N/A |
| **K-Anonymity** | No | No | No | Partial (k-guarantee) | N/A (tabular only) | N/A |
| **Split Learning** | Partially (model split) | No (activations shared) | No | No | Yes | Yes |
| **SMPC** | Yes (secret shared) | Yes (shares meaningless) | Yes | Yes (crypto) | No (communication heavy) | Yes (Shamir threshold) |
| **Federated LoRA** | Yes (data local) | Partially (only adapter) | No | No | Yes (160MB not 14GB) | Yes |

### 3.5 Regulatory Mapping

| Regulation | Region | Key requirements | Recommended PET stack |
|-----------|--------|-----------------|----------------------|
| **HIPAA** | USA | De-identification (Safe Harbor / Expert Determination), minimum necessary, BAA | FL + DP + de-identification |
| **GDPR** | EU | Lawful basis, data minimisation, right to erasure, DPIA | FL + DP (formal guarantee) + consent management |
| **PDPA** | Singapore | Consent, purpose limitation, data protection officer | FL + SecAgg + audit logging |
| **PIPL** | China | Separate consent for cross-border, data localisation | FL (data stays in-country) + TEE |
| **PIPEDA** | Canada | Meaningful consent, limited collection | FL + DP + synthetic data for research |
| **LGPD** | Brazil | Legal basis, data minimisation, DPO | FL + DP |
| **HITECH** | USA | Breach notification, encryption requirement | FL + TLS + SecAgg + TEE |
| **21 CFR Part 11** | USA (FDA) | Audit trail, electronic signatures, validation | FL + audit logging + TEE (verified computation) |
| **PCI DSS** | Global (payments) | Cardholder data protection, encryption at rest/transit | FL + SecAgg + TLS (data never leaves PCI scope) |
| **SOX** | USA (finance) | Financial reporting integrity, audit trails | FL + audit logging |
| **CCPA/CPRA** | California | Consumer data rights, opt-out of sale | FL + DP (minimise data collection) |
| **AI Act** | EU | Risk-based AI regulation, transparency | FL + DP + explainability + audit |

### 3.6 Testable Data Types, Models and FL Compatibility

This section covers data types where FL can be demonstrated end-to-end using public datasets or synthetic data. Each entry includes validated results from our benchmarks or models that are directly reusable.

#### 3.6.1 Structured Clinical Data (Tabular: Vitals, Labs, Diagnoses) -- TESTED

| Attribute | Details |
|-----------|---------|
| **Format** | CSV/Parquet, rows = patient encounters, columns = features |
| **Size** | 1K-10M rows, 10-500 features |
| **Our model** | BiLSTM (500K params) -- validated |
| **Our data** | eICU (200K stays, 188 hospitals, public via PhysioNet) |

**Model architectures:**

| Model | Parameters | FL tested | DP tested | Notes |
|-------|-----------|-----------|-----------|-------|
| **BiLSTM** | ~500K | **Yes: 0.806 acc** | **Yes: 0.651 acc** | Our sepsis model |
| Logistic Regression | <10K | Easy to test | Excellent DP tolerance | Baseline |
| MLP | 10K-1M | Easy to test | Good DP tolerance | Simple, fast |
| XGBoost | ~100K | Partial (FedXGB) | Difficult | Tree-based FL is research-stage |

**Public datasets for testing:**

| Dataset | Task | Size | Access |
|---------|------|------|--------|
| **eICU** (our data) | Sepsis prediction | 200K stays, 208 hospitals | PhysioNet (free, requires credentialing) |
| **MIMIC-IV** | Mortality, readmission, LOS | 524K admissions | PhysioNet (free, requires credentialing) |
| **Heart Disease UCI** | Cardiac diagnosis | 303 patients | UCI ML Repo (instant download) |
| **Diabetes 130-US** | Readmission prediction | 100K encounters | UCI ML Repo (instant download) |

**Validated FL results:**

| Strategy | Accuracy (IID) | Accuracy (Extreme non-IID) |
|----------|---------------|---------------------------|
| FedAvg | **0.806** | 0.788 |
| SCAFFOLD | **0.809** | 0.787 |
| SecAgg | 0.804 | 0.648 |
| DP Central | 0.651 | 0.687 |
| DP Local | 0.493 | 0.420 |

---

#### 3.6.2 Medical Images (2D: X-ray, Skin, Retinal) -- TESTED

| Attribute | Details |
|-----------|---------|
| **Format** | PNG/JPEG, 224x224 to 1024x1024 pixels |
| **Size** | 10K-200K images |
| **Our model** | DenseNet-121 (8M params) -- validated |
| **Our data** | NIH Chest X-ray (112K images, 14 pathologies, public) |

**Model architectures:**

| Model | Parameters | FL tested | DP tested | Notes |
|-------|-----------|-----------|-----------|-------|
| **DenseNet-121** | 8M | **Yes: 0.819 AUC** | Fails (0.50 AUC) | Our chest model |
| ResNet-50 | 25M | Same architecture, easy to swap | Fails | Standard alternative |
| EfficientNet-B4 | 19M | Same architecture | Fails | Better accuracy/param |
| MobileNetV3 | 5M | Same architecture | Marginal | Smaller, better DP tolerance |

**Public datasets for testing:**

| Dataset | Task | Size | Access |
|---------|------|------|--------|
| **NIH Chest X-ray** (our data) | 14 pathology classification | 112K images, 42GB | NIH (free download) |
| **ISIC Skin Lesion** | Melanoma detection | 33K images | ISIC Archive (free) |
| **Diabetic Retinopathy** | Severity grading | 88K images | Kaggle (free) |
| **RSNA Pneumonia** | Pneumonia detection | 30K images | Kaggle (free) |

**Validated FL results (300 rounds, early stop, global eval):**

| Strategy | IID AUC | Moderate AUC | Extreme AUC | Rounds to converge |
|----------|---------|-------------|-------------|-------------------|
| Centralized | **0.803** | - | - | 6 epochs |
| **FedAvg** | **0.811** | **0.814** | **0.817** | 15-20 |
| FedProx | 0.800 | 0.807 | 0.801 | 50-60 |
| SCAFFOLD | 0.762 | 0.767 | 0.760 | 25-34 |
| SecAgg | 0.763 | 0.765 | 0.770 | 54-59 |
| OneOwner | 0.763 | 0.762 | 0.767 | 29-37 |

---

#### 3.6.3 Clinical Text / LLM Fine-tuning -- TESTED

| Attribute | Details |
|-----------|---------|
| **Format** | Free text clinical notes (synthetic or real) |
| **Size** | 600-6000 notes per hospital |
| **Our model** | Mistral 7B Instruct (QLoRA, 160 MB adapter) -- validated |
| **Our data** | Synthetic clinical notes (cardiology, pulmonology, sepsis) |

**Model architectures:**

| Model | Parameters | FL tested | DP tested | Adapter size |
|-------|-----------|-----------|-----------|-------------|
| **Mistral 7B** (our model) | 7.2B | **Yes (QLoRA)** | **Yes (MIA 1.0->0.83)** | **160 MB** |
| Llama 3 8B | 8B | Same approach | Same | 160 MB |
| BERT/BioBERT | 110M | LoRA, easier | Marginal | 1-5 MB |
| OPT-125M | 125M | Tested (fast iteration) | Yes | 1.2 MB |

**Data (synthetic -- can generate instantly):**

```bash
python peft/gen_clinical_data.py    # generates 3 x 2000 clinical notes
```

**Validated FL + privacy results:**

| Method | QA Score | MIA Advantage | Canary Leaked |
|--------|---------|--------------|--------------|
| Base (no fine-tune) | 0.150 | N/A | N/A |
| Centralized | 0.000 (overfit) | **1.000** (fully vulnerable) | 5/12 (41.7%) |
| **Federated LoRA** | 0.167 | **1.000** (still vulnerable) | - |
| **Federated LoRA + DP** | **0.367** | **0.833** (reduced) | 2/12 (16.7%) |

---

#### 3.6.4 Time Series / Waveforms (ECG, EEG, ICU Monitoring) -- REUSABLE

| Attribute | Details |
|-----------|---------|
| **Format** | 1D time series, 100Hz-1kHz sampling |
| **Size** | 1K-200K recordings |
| **Our model** | BiLSTM (same architecture as sepsis -- directly reusable) |
| **Data** | PTB-XL (free download, no registration) |

**Model architectures:**

| Model | Parameters | FL compatible | DP compatible | Notes |
|-------|-----------|---------------|---------------|-------|
| **BiLSTM** (reuse our model) | 200K-2M | **Yes (same code)** | **Yes** | Change input_dim, reuse everything |
| 1D CNN | 100K-1M | Same FL pipeline | **Yes** | Faster than LSTM |
| Temporal CNN | 500K-5M | Same FL pipeline | Yes | Dilated convolutions |

**Public datasets for testing:**

| Dataset | Task | Size | Access |
|---------|------|------|--------|
| **PTB-XL** | 12-lead ECG classification | 21K recordings | PhysioNet (free, instant download) |
| **PhysioNet 2017** | AFib detection | 8.5K recordings | PhysioNet (free) |
| **Sleep-EDF** | Sleep staging | 197 recordings | PhysioNet (free) |
| **CHBMIT** | Seizure detection EEG | 23 patients | PhysioNet (free) |

**How to test:** Replace `sepsis/partition_utils.py` data loader with ECG data. The BiLSTM model, FL strategies, and entire pipeline work unchanged -- just change `input_dim` and data path.

---

#### 3.6.5 Financial Transaction Data (Fraud, AML) -- TESTABLE

| Attribute | Details |
|-----------|---------|
| **Format** | Tabular: transaction amount, merchant, time, location |
| **Size** | 100K-6M transactions |
| **Model** | Same BiLSTM or MLP from our pipeline |
| **Data** | Kaggle (free, instant download) |

**Model architectures:**

| Model | Parameters | FL compatible | DP compatible | Notes |
|-------|-----------|---------------|---------------|-------|
| **LSTM** (reuse our code) | 200K-2M | **Yes** | **Yes** | Transaction sequences |
| MLP | 10K-1M | Same pipeline | **Yes** | Tabular features |
| Autoencoder | 500K-5M | Same pipeline | Yes | Anomaly detection |

**Public datasets for testing:**

| Dataset | Task | Size | Access |
|---------|------|------|--------|
| **IEEE-CIS Fraud Detection** | Card fraud | 590K transactions | Kaggle (free, instant) |
| **PaySim (Synthetic)** | Mobile money fraud | 6M transactions | Kaggle (free, instant) |
| **Credit Card Fraud** | Binary fraud detection | 284K transactions | Kaggle (free, instant) |
| **Elliptic Bitcoin** | Illicit transaction detection | 200K transactions | Kaggle (free, instant) |

**How to test:** Convert transaction CSV to NPZ format (same as sepsis pipeline). The FL infrastructure, strategies, and privacy tools work unchanged.

**Government relevance:** Cross-bank AML, tax fraud detection across jurisdictions, sanctions compliance.

---

#### 3.6.6 Large Language Models (Government / Multi-Agency NLP) -- TESTED

| Attribute | Details |
|-----------|---------|
| **Format** | Text documents (policy, reports, correspondence) |
| **Size** | GB-TB per agency |
| **Our model** | Mistral 7B QLoRA -- validated |
| **Our data** | Synthetic (generate with gen_clinical_data.py, adapt templates) |

**Model architectures:**

| Model | Parameters | FL tested | DP tested | Adapter size |
|-------|-----------|-----------|-----------|-------------|
| **Mistral 7B** (validated) | 7.2B | **Yes** | **Yes** | 160 MB |
| Llama 3 8B | 8B | Same approach | Same | 160 MB |
| BERT/RoBERTa | 110-355M | LoRA | Marginal | 1-5 MB |

**How to test any domain:** Modify `gen_clinical_data.py` templates to generate domain-specific text:
- Tax: replace medical terms with tax forms, regulations, audit findings
- Immigration: replace with visa types, country codes, risk indicators
- Legal: replace with case types, statutes, rulings
- Defence: replace with threat types, locations, entity names

The federated LoRA pipeline, privacy attacks (MIA, canary extraction), and DP protection all work on any text domain without code changes.

**Government applications:**
- Immigration: federate across border agencies without sharing traveller records
- Tax/Revenue: cross-agency fraud NLP without sharing taxpayer data
- Defence: intelligence summarisation without cross-clearance access
- Judicial: case law analysis without sharing sealed records
- Citizen services: chatbot training across departments

### 3.7 Model Size vs PET Compatibility Summary

| Model size | Examples | FedAvg | DP | SecAgg | LoRA | Split Learning |
|-----------|---------|--------|----|----|------|------|
| **< 1M params** | Logistic, MLP, small LSTM | Excellent | **Excellent** | Excellent | Not needed | Not needed |
| **1-10M params** | BiLSTM, 1D CNN, MobileNet | Excellent | **Marginal** | Excellent | Optional | Optional |
| **10-100M params** | DenseNet, ResNet, BERT | Good | **Fails** | Good | Recommended | Recommended |
| **100M-1B params** | Large BERT, ViT-L, 3D UNet | Good (slow) | Fails | Good | **Required** | Required |
| **1B+ params** | Mistral 7B, Llama 8B | LoRA only | DP on adapter only | LoRA + SecAgg | **Essential** | Essential |

**Key insight:** DP works inversely to model size. For models > 10M params, use SecAgg (hides updates cryptographically) instead of DP (adds noise that destroys the model).

---

## 4. Deployment Guide

### 4.1 Prerequisites

```bash
# Local machine
brew install terraform docker
pip install flwr torch

# AWS
aws configure  # set region to ap-southeast-1
# Need: EC2 key pair, S3 bucket for data, security groups configured
```

### 4.2 Local Testing (Docker Simulation)

```bash
# Run all 8 strategies on both tasks (sepsis + chest X-ray synthetic)
docker compose -f docker-compose.test.yml up --build

# Run specific task
docker compose -f docker-compose.test.yml run test python run_tests.py sepsis
docker compose -f docker-compose.test.yml run test python run_tests.py chest
```

### 4.3 TLS Certificate Generation

```bash
# Generate self-signed CA + server cert
./deploy/gen_certs.sh ./certs

# Output:
#   certs/ca.pem      - distribute to all nodes
#   certs/server.pem  - SuperLink only
#   certs/server.key  - SuperLink only (keep secret)
```

### 4.4 Distributed Deployment (Docker on EC2)

#### Step 1: Provision Infrastructure

```bash
# Using Terraform
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
terraform init && terraform apply

# Or manually: 1 server (t3.large+), N clients (g6.xlarge+ for GPU)
```

#### Step 2: Build and Distribute Docker Image

```bash
# Build on SuperLink machine
docker build -t healthcare-fl:latest .
docker save healthcare-fl:latest | gzip > fl-image.tar.gz

# Ship to each client
scp fl-image.tar.gz ec2-user@CLIENT_IP:/tmp/
ssh ec2-user@CLIENT_IP 'docker load < /tmp/fl-image.tar.gz'
```

#### Step 3: Start SuperLink (Server)

```bash
# With TLS
docker run -d --name fl-superlink \
  --restart unless-stopped --network host \
  -v /path/to/certs:/certs:ro \
  --log-opt max-size=50m --log-opt max-file=5 \
  healthcare-fl:latest \
  flower-superlink \
    --ssl-certfile /certs/server.pem \
    --ssl-keyfile /certs/server.key \
    --ssl-ca-certfile /certs/ca.pem
```

#### Step 4: Start SuperNodes (Clients)

```bash
# On each client machine (with GPU)
docker run -d --name fl-supernode \
  --restart unless-stopped --network host \
  --gpus all \
  -v /path/to/certs/ca.pem:/certs/ca.pem:ro \
  -v /path/to/data:/data:ro \
  -e DATA_PATH=/data/flower_data \
  --log-opt max-size=50m --log-opt max-file=5 \
  healthcare-fl:latest \
  flower-supernode \
    --root-certificates /certs/ca.pem \
    --superlink SUPERLINK_PRIVATE_IP:9092 \
    --node-config "partition-id=0 num-clients=5 max-samples=0" \
    --clientappio-api-address 0.0.0.0:7070
```

#### Step 5: Submit Experiment

```bash
# From SuperLink machine
flwr run ./sepsis distributed \
  --run-config 'strategy="IID" num-rounds="20" num-clients="5"'
```

### 4.5 PEFT Deployment (Mistral 7B)

```bash
# Install PEFT stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft accelerate bitsandbytes trl datasets scikit-learn

# Fine-tune Mistral 7B on clinical notes
python peft/train_mistral.py

# Federated LoRA with privacy test
python peft/gen_clinical_data.py          # generate 3 x 2000 notes
python peft/fed_mistral_privacy.py        # FL + MIA (~90 min)

# Full attack suite
python peft/attack_suite.py               # 6 attacks, DP comparison

# Interactive demo
python peft/demo_fed_lora.py --clients 3 --rounds 5 --dp-noise 0.5
```

### 4.6 Management Commands

```bash
# Check cluster status
docker logs fl-superlink --tail 20
docker logs fl-supernode --tail 20

# Monitor GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

# Stop everything
docker rm -f fl-superlink fl-supernode

# View experiment results
cat experiments/results/*.json
```

---

## 5. FL Strategy Guide

### 5.1 Strategy Selection

| Scenario | Recommended Strategy | Why |
|---------|---------------------|-----|
| **Default / IID data** | FedAvg | Simplest, fastest convergence (20 rounds) |
| **Mild non-IID** | FedProx (mu=0.1) | Proximal term prevents client drift |
| **Strong non-IID** | SCAFFOLD | Control variates correct gradient drift |
| **Privacy required** | SecAgg+ | Masks hide individual updates from server |
| **Formal privacy guarantee** | DP-Central | Mathematical epsilon-delta bound |
| **Single model owner** | OneOwner | Standard FedAvg, access control at deployment |
| **Communication constrained** | Federated LoRA | 160 MB adapter vs 14 GB full model |

### 5.2 Benchmark Results

#### Chest X-ray (NIH, 112K real images, DenseNet-121)

| Strategy | IID AUC | Moderate AUC | Extreme AUC | Rounds | Notes |
|---------|---------|-------------|-------------|--------|-------|
| **Centralized** | **0.803** | - | - | 6 epochs | Upper bound |
| **FedAvg** | **0.811** | **0.814** | **0.817** | 20 | Best FL strategy |
| FedProx | 0.800 | 0.807 | 0.801 | 50-60 | Stable but slow |
| SCAFFOLD | 0.762 | 0.767 | 0.760 | 25-34 | Good for non-IID |
| SecAgg | 0.763 | 0.765 | 0.770 | 54-59 | Equal to SCAFFOLD |
| OneOwner | 0.763 | 0.762 | 0.767 | 29-37 | Same as FedAvg |
| FedAdam | 0.605 | 0.585 | 0.542 | 17-18 | Diverges on pretrained |
| FedYogi | 0.578 | 0.587 | 0.549 | 9-10 | Same issue |
| DP Central | 0.505 | 0.520 | 0.511 | 12-20 | Too much noise for 8M params |
| DP Local | 0.500 | 0.504 | 0.500 | 6-11 | Noise destroys model |

#### Sepsis (eICU, 100K+ samples, BiLSTM)

| Strategy | IID | Moderate | Extreme |
|---------|-----|---------|---------|
| **FedAvg** | 0.804 | **0.806** | 0.788 |
| SCAFFOLD | **0.809** | 0.804 | 0.787 |
| OneOwner | 0.808 | 0.806 | 0.787 |
| FedProx | 0.805 | 0.802 | 0.769 |
| SecAgg | 0.804 | 0.805 | 0.648 |
| DP Central | 0.651 | 0.580 | 0.687 |
| DP Local | 0.493 | 0.479 | 0.420 |

### 5.3 Experiment Name Format

```
Strategy_Parameter_Alpha_value

Examples:
  IID                           FedAvg with uniform data
  FedProx_Mu0.1_Alpha_0.5      FedProx, mu=0.1, Dirichlet alpha=0.5
  FedAdam_Alpha_1.0             FedAdam, moderate non-IID
  DP_Central_Eps50.0_Clip5.0    Central DP, epsilon=50, clip=5
  SecAgg_Alpha_0.1              SecAgg, extreme non-IID
  OneOwner_Alpha_0.5            Single-owner model, strong non-IID
  AdaptiveWarmup_W10_Alpha_0.5  Adam warmup 10 rounds then FedAvg
```

---

## 6. Privacy Analysis

### 6.1 Attack Results

#### Membership Inference Attack (MIA)

Can an attacker determine if a specific patient's data was used in training?

| Model | Method | MIA Advantage | Verdict |
|-------|--------|--------------|---------|
| BiLSTM (sepsis) | FL, no DP | 0.047 | PROTECTED |
| BiLSTM (sepsis) | FL + DP (sigma=1.0) | 0.007 | PROTECTED |
| Mistral 7B | Centralized | 1.000 | VULNERABLE |
| Mistral 7B | FL LoRA | 1.000 | VULNERABLE |
| Mistral 7B | FL LoRA + DP | 0.833 | Partially protected |

#### Canary Extraction

Can planted fake patient IDs (MRN, SSN) be extracted from the model?

| Model | Method | Canaries leaked | Total |
|-------|--------|----------------|-------|
| Mistral 7B | No DP | 5/12 (41.7%) | Leaked MRN, SSN, doctor name |
| Mistral 7B | With DP | 2/12 (16.7%) | Reduced but not eliminated |

#### Training Data Extraction

Can generated text match training data verbatim?

| Metric | No DP | With DP |
|--------|-------|---------|
| 5-gram overlap | 19.2% | 0.2% |

### 6.2 DP Noise Sweep Results

| sigma | epsilon | Loss | QA Score | Canary leaked | MIA Advantage |
|-------|---------|------|---------|--------------|--------------|
| 0.00 | inf | 0.145 | 15.3% | 1/9 | 0.050 |
| 0.01 | inf | 0.164 | 32.7% | 1/9 | 0.075 |
| 0.05 | inf | 0.287 | 37.3% | 1/9 | 0.100 |
| 0.10 | 357.6 | 0.415 | 18.0% | 1/9 | 0.125 |
| 0.50 | 31.5 | 1.000 | 11.3% | 1/9 | 0.000 |
| 1.00 | 13.2 | 4.155 | 11.3% | 1/9 | 0.050 |

**Recommendation:** sigma=0.05 for utility-focused, sigma=0.5 for privacy-focused deployments.

### 6.3 Privacy Recommendations

| Data type | Recommended PETs | Why |
|-----------|-----------------|-----|
| Hospital vitals/labs (tabular) | FL + DP (Central, sigma=0.5) | Small model, DP works well |
| Medical images (X-ray, CT) | FL + SecAgg | Large model, DP destroys utility |
| Clinical notes (LLM) | FL LoRA + DP | Adapter-only sharing + noise |
| Cross-institution research | FL + SecAgg + DP | Defense in depth |
| Regulatory compliance (HIPAA) | FL + DP + audit logging | Formal privacy guarantee needed |

---

## 7. Operational Guide

### 7.1 Adding a New Hospital (Client)

1. Provision EC2 instance (g6.xlarge+ for GPU workloads)
2. Install Docker + NVIDIA Container Toolkit
3. Load the FL Docker image
4. Copy CA certificate (`ca.pem`) to the client
5. Prepare local data in the expected format
6. Start SuperNode container with the correct `partition-id`
7. Verify connection: check SuperLink logs for `ActivateNode`

### 7.2 Running an Experiment

```bash
# 1. Verify all clients connected
docker logs fl-superlink | grep "ActivateNode" | sort -u | wc -l

# 2. Submit experiment
flwr run ./sepsis distributed \
  --run-config 'strategy="FedAvg" num-rounds="20" num-clients="5"'

# 3. Monitor
docker logs fl-superlink -f | grep "ROUND\|aggregate"

# 4. Check results
docker logs fl-superlink | grep "SUMMARY" -A 20
```

### 7.3 GPU Driver Recovery

The NVIDIA driver occasionally crashes after long Docker runs. Recovery:

```bash
# Check if GPU is responsive
nvidia-smi

# If "No devices found":
sudo reboot
# Wait 60s, verify:
nvidia-smi --query-gpu=name --format=csv,noheader
```

### 7.4 Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| SecAgg crash | `'float' object has no attribute 'astype'` | Scalar params in model — fixed in secagg.py |
| DP no learning | AUC = 0.50 | Clip norm too small for model — increase to match update magnitude |
| FedAdam diverges | AUC = 0.46-0.50 | Server adaptive LR unstable on pretrained models — use FedAvg |
| TLS handshake fail | `WRONG_VERSION_NUMBER` | Flower applies TLS to all ports — use insecure for Control API |
| CUDA OOM | `out of memory` | Reduce `num_gpus` per client or batch size |
| Config TOML error | `Invalid value` | String values in node-config need TOML quoting |

---

## 8. Security Checklist

- [ ] TLS certificates generated and distributed
- [ ] `--insecure` flag removed from all commands
- [ ] SSH keys not stored in repo (use secrets manager)
- [ ] Security groups restrict port 9092 to client IPs only
- [ ] SuperNodes in private subnets (no public IP)
- [ ] DP enabled for experiments with sensitive data
- [ ] Privacy budget (epsilon) tracked across experiments
- [ ] MIA attack run on final model before deployment
- [ ] Canary test run to verify no PII leakage
- [ ] Audit log of all FL rounds and participants

---

## 9. Known Limitations

1. **FedAdam/FedYogi** do not converge on pretrained DenseNet-121. Server adaptive optimizers amplify noise from pseudo-gradients. Use FedAvg or SCAFFOLD instead.

2. **DP on large models** (8M+ params) produces random output at any useful epsilon. DP works on BiLSTM (0.65 accuracy) but not DenseNet-121 (0.50 AUC = random).

3. **SecAgg requires all clients** to participate every round. If any client drops, masks don't cancel and the model is corrupted.

4. **FedAvg AUC (0.819) slightly exceeds centralized (0.803)** due to FL's implicit regularization effect and more data-epochs (20 rounds x 5 clients vs 10 epochs).

5. **Non-IID has minimal impact** on this chest X-ray dataset. Dirichlet alpha=0.1 (extreme) produces similar AUC to IID — the patient-level partitioning creates natural heterogeneity that all strategies handle well.

6. **GPU driver instability** on g6 instances after long Docker runs. Requires periodic reboot.

---

## Appendix A: Terraform Variables

```hcl
region              = "ap-southeast-1"
project_name        = "healthcare-fl"
key_name            = "your-key-pair"
data_s3_bucket      = "your-data-bucket"
num_supernodes      = 5
use_gpu             = true
enable_tls          = true
allowed_ssh_cidrs   = ["YOUR_IP/32"]
```

## Appendix B: Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_PATH` | Path to training data | `/data/flower_data` |
| `MAX_SAMPLES` | Cap dataset size (0=unlimited) | `0` |
| `SYNTHETIC` | Use synthetic data (1/0) | `0` |
| `DATASET_PATH` | Chest X-ray image directory | `/data` |
| `CSV_PATH` | Chest X-ray metadata CSV | `Data_Entry_2017.csv` |

## Appendix C: Tested EC2 Instance Types

| Role | Instance | GPU | vCPU | RAM | Cost/hr |
|------|---------|-----|------|-----|---------|
| SuperLink | t3.large | - | 2 | 8 GB | $0.08 |
| SuperNode (CPU) | t3.xlarge | - | 4 | 16 GB | $0.17 |
| SuperNode (GPU) | g6.4xlarge | L4 24GB | 16 | 64 GB | $1.32 |
| SuperNode (GPU) | g6.8xlarge | L4 24GB | 32 | 128 GB | $2.65 |
