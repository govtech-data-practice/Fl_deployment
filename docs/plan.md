# FL + PET Sandbox — Government Readiness Plan

**Last updated:** 2026-05-15
**Status:** In Progress — 12 models, 10 tasks, secure inference demos, distributed runs active
**Goal:** Close the gaps between current POC and production-grade government deployment

---

## Current State

### Implemented + Validated

| Component | Details | Status |
|-----------|---------|--------|
| HFL (11 strategies) | FedAvg, FedProx, FedAdam, FedYogi, SCAFFOLD, SecAgg+, DP-Central, DP-Local, DP-Local low eps, OneOwner | Tested |
| VFL | 3-bank vertical FL on fraud data | Tested |
| Split Learning | BiLSTM split at LSTM/classifier boundary | Tested |
| Transfer Learning | DenseNet pretrained vs random init | Tested |
| Federated LoRA | Mistral 7B QLoRA, 160 MB adapter | Tested |
| Privacy Attacks | MIA, DLG (gradient inversion), canary extraction | Tested |
| Models (12) | BiLSTM, MLP, DenseNet-121, Split BiLSTM, VFL MLP, Mistral 7B, Autoencoder, LogReg, 1D CNN, TabNet, ResNet-small | **Done** |
| Tasks (10) | sepsis, ecg, fraud, chest_xray, gov_llm, anomaly, mortality, drug, satellite, readmission | **Done** |
| Secure Inference | Paillier HE, TEE simulation, secret sharing, functional encryption demos + resource guide | **Done** |
| Models (target) | 30 models across 14 data types, 12 govt domains — see expanded section below | Phase B/C |
| Distributed Infra | TLS (EC P-256), 1 server (g6.8xlarge) + 5 clients (g6.4xlarge, L4 GPU), Docker, NVIDIA 595.71.05 | **Operational** |
| Orchestration | `run_distributed.sh` — automated deploy, run, monitor, collect results | **Done** |
| Docs | Deployment guide, PET reference, regulatory mapping, govt readiness plan | **Done** |

### EC2 Experiment Runs Completed

| Run | Date | Experiments | Result |
|-----|------|-------------|--------|
| Nitro Enclave (chest X-ray) | 2025-12-10 | 21/21 hyperparameter sweep | All SUCCESS |
| Large benchmark (chest X-ray) | 2026-01-15 to 01-20 | 38 strategies x non-IID levels | All SUCCESS |
| PEFT/LoRA (Mistral 7B) | Validated | FL + MIA + canary extraction | MIA 1.0 -> 0.83 with DP |

### Key Findings

- FedAvg AUC 0.819 > centralized 0.803 on chest X-ray (implicit regularisation)
- DP destroys large models (DenseNet AUC 0.50) but works on small models (BiLSTM 0.65)
- Non-IID has minimal impact on chest X-ray (natural patient-level heterogeneity)
- FedAdam/FedYogi diverge on pretrained models
- LLM canary leakage: 41.7% without DP, 16.7% with DP

---

## Models + Data Types: Comprehensive Government Coverage

### Current Coverage (4 models, 5 data types)

| # | Model | Params | Data Type | Task | Govt Domain | Status |
|---|-------|--------|-----------|------|-------------|--------|
| 1 | BiLSTM | 500K | Structured tabular (vitals/labs) | Sepsis prediction | Healthcare | Tested |
| 2 | BiLSTM | 200K | 1D time series (12-lead ECG) | Arrhythmia classification | Healthcare/Wearable | Tested |
| 3 | MLP | 50K | Tabular (transactions) | Fraud detection | Finance | Tested |
| 4 | DenseNet-121 | 8M | 2D medical images (X-ray) | 14-pathology classification | Healthcare | Tested |
| 5 | Mistral 7B (QLoRA) | 7.2B | Clinical free text | QA / summarisation | Healthcare/NLP | Tested |
| 6 | Split BiLSTM | 500K | Structured tabular | Split learning sepsis | Healthcare | Tested |
| 7 | VFL MLP | 50K | Vertically partitioned tabular | Cross-bank fraud | Finance | Tested |

### Target Coverage (15+ models, 12+ data types)

Expanding to cover all major government domains with appropriate model architectures.

#### Tier 1 — Add to existing tasks (reuse infra, change model/data loader only)

| # | Model | Params | Data Type | Task | Govt Domain | Public Dataset | Effort |
|---|-------|--------|-----------|------|-------------|----------------|--------|
| 8 | **1D CNN** | 200K | ECG waveform | Arrhythmia detection | Healthcare/Wearable | PTB-XL (PhysioNet) | Low — swap model in BiLSTM pipeline |
| 9 | **Temporal CNN (TCN)** | 1M | Multi-variate time series | ICU mortality prediction | Healthcare | MIMIC-IV (PhysioNet) | Low — same pipeline as sepsis |
| 10 | **Logistic Regression** | <10K | Structured tabular | Readmission prediction | Healthcare | Diabetes 130-US (UCI) | Low — simplest baseline, excellent DP tolerance |
| 11 | **Autoencoder** | 500K | Tabular (transactions) | Anomaly detection (unsupervised) | Finance/AML | Elliptic Bitcoin (Kaggle) | Low — swap MLP for autoencoder |
| 12 | **ResNet-50** | 25M | 2D images | General image classification | Defence/Border | CIFAR-100 or ImageNet subset | Low — swap DenseNet backbone |
| 13 | **EfficientNet-B4** | 19M | 2D images | High-res image classification | Healthcare/Pathology | ISIC Skin Lesion (free) | Low — swap DenseNet backbone |
| 14 | **MobileNetV3** | 5M | 2D images | Edge/mobile classification | Smart City/IoT | Any image dataset | Low — swap DenseNet backbone, better DP tolerance |

#### Tier 2 — New data types (new data loaders + model code, reuse FL infra)

| # | Model | Params | Data Type | Task | Govt Domain | Public Dataset | Effort |
|---|-------|--------|-----------|------|-------------|----------------|--------|
| 15 | **3D CNN (ResNet3D)** | 33M | 3D volumetric images (CT/MRI) | Lung nodule detection | Healthcare | LUNA16 (free) | Medium — new data loader for 3D volumes |
| 16 | **U-Net** | 31M | 2D images (segmentation) | Organ/tumour segmentation | Healthcare | Medical Segmentation Decathlon (free) | Medium — new segmentation head + Dice loss |
| 17 | **GNN (GraphSAGE)** | 1-5M | Graph-structured data | Molecular property prediction | Pharma/Drug Discovery | MoleculeNet / ZINC (free) | Medium — new graph data loader |
| 18 | **GNN (GAT)** | 1-5M | Network graph | Illicit transaction detection | Finance/AML/Tax | Elliptic Bitcoin (Kaggle) | Medium — transaction graph |
| 19 | **Transformer (ViT-B/16)** | 86M | 2D images | High-accuracy image classification | Defence/Intel | ImageNet subset | Medium — new model, high param count |
| 20 | **BERT / BioBERT** | 110M | Clinical text (NER/classification) | Named entity recognition | Healthcare/Legal | i2b2 NER (PhysioNet) | Medium — new tokenisation + NER head |
| 21 | **Whisper (small)** | 244M | Audio / speech | Speech-to-text | Defence/Intel/Citizen Services | LibriSpeech (free) | Medium — new audio data pipeline |
| 22 | **YOLO v8 (nano)** | 3M | 2D images (object detection) | Object detection / surveillance | Defence/Border/Smart City | COCO subset (free) | Medium — detection head + bbox loss |
| 23 | **TabNet** | 1-10M | Structured tabular | Interpretable tabular prediction | Tax/Revenue/Healthcare | Any tabular dataset | Medium — attention-based tabular model |
| 24 | **XGBoost (FedXGB)** | ~100K | Structured tabular | Tree-based classification | Tax/Revenue/Finance | IEEE-CIS Fraud (Kaggle) | Medium — federated tree protocol (research-stage) |

#### Tier 3 — New FL paradigms (new FL pipelines, significant engineering)

| # | Model | Params | Data Type | Task | Govt Domain | Public Dataset | Effort |
|---|-------|--------|-----------|------|-------------|----------------|--------|
| 25 | **Llama 3 8B (QLoRA)** | 8B | Government documents | Document classification/QA | All govt | Synthetic (adapt gen_clinical_data.py) | Low — same LoRA pipeline as Mistral |
| 26 | **Stable Diffusion (LoRA)** | 860M | 2D images (generation) | Synthetic image generation | Healthcare/Defence | Domain-specific images | High — diffusion training pipeline |
| 27 | **3D U-Net** | 50M+ | 3D volumetric (segmentation) | Organ segmentation (CT) | Healthcare | KiTS / BTCV (free) | High — 3D segmentation + memory management |
| 28 | **Multi-modal (CLIP-like)** | 150M+ | Image + text pairs | Cross-modal retrieval | Defence/Intel | ROCO medical (free) | High — dual-encoder architecture |
| 29 | **Reinforcement Learning** | Varies | Sequential decisions | Resource allocation / scheduling | Defence/Logistics | Gym environments | High — new FL paradigm for RL |
| 30 | **Federated Recommender** | 1-50M | User-item interactions | Citizen service recommendations | Citizen Services | MovieLens (free, proxy) | High — new rec-sys FL protocol |

### Data Type Coverage Matrix

| # | Data Type | Format | Size Range | Current Models | Target Models | Govt Use Cases |
|---|-----------|--------|-----------|----------------|---------------|----------------|
| 1 | **Structured tabular (EHR)** | CSV/Parquet | 1K-10M rows | BiLSTM, MLP | + LogReg, TabNet, XGBoost | Healthcare, tax, revenue, benefits |
| 2 | **1D time series / waveform** | NumPy arrays | 1K-200K recordings | BiLSTM | + 1D CNN, TCN | ECG, EEG, seismic, network traffic |
| 3 | **2D images (classification)** | PNG/JPEG 224x224 | 10K-200K images | DenseNet-121 | + ResNet-50, EfficientNet, MobileNet, ViT | X-ray, pathology, satellite, surveillance |
| 4 | **2D images (detection)** | PNG/JPEG + bbox | 10K-100K images | None | + YOLO v8 | Border security, smart city, defence |
| 5 | **2D images (segmentation)** | PNG/JPEG + masks | 1K-50K images | None | + U-Net | Tumour segmentation, land use mapping |
| 6 | **3D volumetric images** | NIfTI/DICOM | 1K-10K volumes | None | + 3D CNN, 3D U-Net | CT/MRI diagnosis, geological survey |
| 7 | **Clinical / legal text** | Free text | 1K-100K documents | Mistral 7B | + BERT, BioBERT, Llama 3 | Clinical notes, legal docs, intel reports |
| 8 | **Audio / speech** | WAV/MP3 | 1K-100K clips | None | + Whisper | Transcription, call centres, intel |
| 9 | **Graph / network data** | Edge lists / adjacency | 1K-1M nodes | None | + GNN (GraphSAGE, GAT) | AML networks, social networks, drug interaction |
| 10 | **Transaction sequences** | CSV (time-ordered) | 100K-6M rows | MLP | + LSTM, Autoencoder | Fraud, AML, tax evasion |
| 11 | **Satellite / geospatial** | GeoTIFF / multispectral | 1K-50K patches | None | + ResNet-50, U-Net | Defence, border, environment, agriculture |
| 12 | **Genomic / molecular** | FASTA/VCF/SMILES | 1K-1M sequences | None | + GNN, 1D CNN | Pharma, biodefence, public health |
| 13 | **Multimodal (image+text)** | Paired data | 10K-100K pairs | None | + CLIP-like | Radiology reports + images, intel reports + photos |
| 14 | **Video / CCTV** | MP4 frames | 1K-10K clips | None | + 3D CNN, SlowFast | Smart city, border, defence |

### Model Size vs PET Compatibility (expanded)

| Size Tier | Params | Models | FedAvg | DP | SecAgg | LoRA | Split | Recommended PET Stack |
|-----------|--------|--------|--------|----|--------|------|-------|-----------------------|
| **Micro** | <100K | LogReg, small MLP | Excellent | **Excellent** | Excellent | N/A | N/A | FL + DP (any epsilon) |
| **Small** | 100K-1M | BiLSTM, 1D CNN, TCN, Autoencoder | Excellent | **Good** | Excellent | Optional | Optional | FL + DP (eps 10-50) |
| **Medium** | 1-10M | MobileNet, YOLO nano, TabNet | Good | **Marginal** | Excellent | Optional | Optional | FL + SecAgg preferred |
| **Large** | 10-100M | DenseNet, ResNet, EfficientNet, U-Net, BERT, ViT | Good | **Fails** | Good | Recommended | Recommended | FL + SecAgg + LoRA |
| **XL** | 100M-1B | Whisper, CLIP, 3D U-Net | Slow | Fails | Good | **Required** | Required | FL LoRA + SecAgg |
| **XXL** | 1B+ | Mistral 7B, Llama 3 8B, Stable Diffusion | LoRA only | Adapter DP only | LoRA + SecAgg | **Essential** | Essential | FL LoRA + DP on adapter + SecAgg |

### Government Domain Coverage Matrix

| Domain | Data Types | Recommended Models | Priority |
|--------|-----------|-------------------|----------|
| **Healthcare (hospitals)** | Tabular EHR, X-ray, CT/MRI, clinical notes, ECG | BiLSTM, DenseNet, U-Net, 3D CNN, Mistral/BERT | HIGH — primary use case |
| **Finance (AML/fraud)** | Transactions, network graphs | MLP, LSTM, Autoencoder, GNN | HIGH — cross-bank FL |
| **Defence / Intelligence** | Satellite imagery, text reports, audio, video | ResNet, ViT, BERT, Whisper, YOLO, CLIP | HIGH — multi-modal intelligence |
| **Border Security** | Biometrics (face/iris), travel records, watchlists | ResNet, MLP, GNN | HIGH — cross-nation FL |
| **Tax / Revenue** | Tax returns (tabular), documents | TabNet, XGBoost, BERT | MEDIUM — cross-agency fraud |
| **Public Health / Epidemiology** | Disease surveillance (tabular), genomic | BiLSTM, 1D CNN, GNN | MEDIUM — outbreak prediction |
| **Smart City / Transport** | CCTV video, sensor time series, traffic tabular | YOLO, TCN, MLP | MEDIUM — cross-district FL |
| **Education** | Student records (tabular), essays (text) | MLP, BERT | LOW — simpler models |
| **Energy / Utilities** | Smart meter time series, SCADA | TCN, 1D CNN, Autoencoder | MEDIUM — anomaly detection |
| **Judiciary / Legal** | Case text, structured filings | BERT, Mistral/Llama | LOW — text-heavy |
| **Citizen Services** | Service records, feedback text | MLP, BERT, Recommender | LOW — simpler models |
| **Pharma / Drug Discovery** | Molecular graphs, genomic | GNN (GraphSAGE) | MEDIUM — cross-org drug screening |

### Implementation Priority for New Models

**Phase A — Quick wins (reuse existing FL pipeline, swap model only):**

| Model | Swap for | Files to change | Effort |
|-------|----------|----------------|--------|
| Logistic Regression | MLP | `models/mlp/` — simplify to linear | 1 day |
| 1D CNN | BiLSTM | `models/bilstm/` — swap LSTM for Conv1d | 1 day |
| ResNet-50 | DenseNet-121 | `models/densenet/` — swap backbone | 1 day |
| EfficientNet-B4 | DenseNet-121 | `models/densenet/` — swap backbone | 1 day |
| MobileNetV3 | DenseNet-121 | `models/densenet/` — swap backbone | 1 day |
| Autoencoder | MLP | `models/mlp/` — add decoder + reconstruction loss | 2 days |
| Llama 3 8B | Mistral 7B | `models/mistral/` — swap model ID | 1 day |

**Phase B — New data types (new data loader + model, reuse FL strategies):**

| Model | New capability | Files to create | Effort |
|-------|---------------|----------------|--------|
| U-Net | Image segmentation | `models/unet/`, `tasks/segmentation/` | 1 week |
| YOLO v8 nano | Object detection | `models/yolo/`, `tasks/detection/` | 1 week |
| GNN (GraphSAGE) | Graph data | `models/gnn/`, `tasks/graph/` | 1 week |
| BERT / BioBERT | Text NER/classification | `models/bert/`, `tasks/ner/` | 1 week |
| Whisper small | Audio/speech | `models/whisper/`, `tasks/audio/` | 1-2 weeks |
| 3D CNN | Volumetric images | `models/resnet3d/`, `tasks/volumetric/` | 1-2 weeks |
| TabNet | Interpretable tabular | `models/tabnet/` | 3 days |
| TCN | Long time series | `models/tcn/` | 3 days |

**Phase C — New FL paradigms (significant engineering):**

| Model | New paradigm | Effort |
|-------|-------------|--------|
| Stable Diffusion LoRA | Federated generative model | 2-3 weeks |
| Multi-modal (CLIP) | Federated dual-encoder | 2-3 weeks |
| Federated XGBoost | Tree-based FL protocol | 2-3 weeks |
| Federated RL | RL + FL protocol | 3-4 weeks |
| Federated Recommender | Rec-sys FL protocol | 2-3 weeks |

---

## Phase 1: Integrity + Robustness (Priority: HIGH)

Byzantine fault tolerance and poisoning resistance. Government threat model includes compromised or malicious participants.

### 1.1 Robust Aggregation Strategies

Add to `fl_common/strategies.py`:

| Strategy | What it does | Complexity |
|----------|-------------|------------|
| **Krum** | Selects the update closest to all others (rejects outliers) | Low |
| **Multi-Krum** | Selects top-k closest updates, averages them | Low |
| **Trimmed Mean** | Drops top/bottom beta% of each parameter, averages the rest | Low |
| **Bulyan** | Krum selection + coordinate-wise trimmed mean | Medium |
| **FLTrust** | Server holds small clean dataset, scores clients by cosine similarity to server update | Medium |

**Acceptance criteria:**
- [ ] Each strategy passes the same test harness as existing 9 strategies
- [ ] Poisoning attack test: inject 1 malicious client (label flip or gradient scaling), verify robust strategies recover vs FedAvg degradation
- [ ] Add to `runners/run_ec2.py` STRATEGIES dict and YAML scenario support

### 1.2 Poisoning Attack Suite

Add to `privacy/`:

| Attack | Type | Description |
|--------|------|-------------|
| **Label flipping** | Data poisoning | Malicious client flips labels (e.g., healthy -> sepsis) |
| **Gradient scaling** | Model poisoning | Malicious client scales update by large factor |
| **Backdoor (Trojan)** | Model poisoning | Inject trigger pattern that causes misclassification |

**Acceptance criteria:**
- [ ] `privacy/test_poisoning.py` with `test_label_flip()`, `test_gradient_scaling()`, `test_backdoor()`
- [ ] Each attack tested against FedAvg (vulnerable) and Krum/TrimmedMean (resistant)
- [ ] Results integrated into `print_summary()` and JSON output

---

## Phase 2: Audit + Provenance (Priority: HIGH)

Non-negotiable for regulated government environments (21 CFR Part 11, PDPA, HIPAA).

### 2.1 Round-Level Audit Log

| Field | Description |
|-------|-------------|
| `round_id` | Sequential round number |
| `timestamp` | ISO 8601 |
| `participants` | List of client IDs that contributed |
| `strategy` | Strategy name |
| `num_samples_per_client` | How much data each client used |
| `aggregate_metric` | Loss, accuracy/AUC after aggregation |
| `dp_epsilon_spent` | Cumulative privacy budget consumed |
| `model_hash` | SHA-256 of aggregated model weights |
| `anomalies` | Failed clients, timeout, rejected updates (robust agg) |

**Implementation:**
- [ ] `fl_common/audit.py` — `AuditLogger` class, appends JSONL per round
- [ ] Integrate into `MetricCapture.aggregate_evaluate()` in `runners/run_ec2.py`
- [ ] Each experiment produces `results/audit_<experiment>_<timestamp>.jsonl`
- [ ] Immutable: append-only, include HMAC signature per entry

### 2.2 Model Provenance / Lineage

| Artifact | What to track |
|----------|--------------|
| Model checkpoint | SHA-256, round produced, parent model hash |
| Training data | Dataset version, partition config, num samples per client |
| Configuration | Strategy, hyperparams, DP settings, num rounds |
| Participants | Which clients contributed to which rounds |

**Implementation:**
- [ ] `results/provenance_<experiment>.json` generated alongside results
- [ ] Links model hash -> contributing clients -> data partitions -> config
- [ ] Queryable: "which hospitals contributed to model version X?"

### 2.3 Privacy Budget Tracker

- [ ] Persistent epsilon ledger across experiments (not just per-run)
- [ ] Alert when cumulative epsilon exceeds threshold (e.g., 10.0)
- [ ] Integrates with RDP accountant in `fl_common/dp.py`
- [ ] Output: `results/privacy_ledger.json`

---

## Phase 3: TEE as a PET (Priority: HIGH)

Current state: Nitro Enclave used as experiment runtime. Need: enclave protects aggregation from server operator.

### 3.1 Enclave-Based Aggregation

| Component | Current | Target |
|-----------|---------|--------|
| What runs in enclave | Entire experiment (server + simulated clients) | Only the aggregation server |
| Client verification | None | Clients verify enclave attestation before sending updates |
| Server operator access | Can read enclave console logs | Cannot access model updates (encrypted in enclave memory) |

**Implementation:**
- [ ] Refactor SuperLink to run inside Nitro Enclave
- [ ] Implement attestation verification in client connection handshake
- [ ] Encrypt client updates with enclave public key (from attestation doc)
- [ ] Document threat model: what the enclave protects and what it doesn't

### 3.2 Attestation Flow

```
Client                          Enclave (SuperLink)
  |                                    |
  |  --- request attestation doc --->  |
  |  <-- attestation doc (PCRs) ----   |
  |                                    |
  |  [verify PCRs match expected       |
  |   enclave image hash]              |
  |                                    |
  |  --- TLS + encrypted update --->   |
  |  <-- aggregated model -----------  |
```

- [ ] PCR validation library for clients
- [ ] CI: build enclave image, record expected PCR values
- [ ] Documentation: how to verify enclave integrity

---

## Phase 4: Additional PETs (Priority: MEDIUM)

### 4.1 Federated Analytics (No ML)

Aggregate statistics across hospitals without model training. High value for population health, epidemiology, census.

- [ ] `fl_analytics/` module
- [ ] Supported queries: count, sum, mean, histogram, percentile
- [ ] Privacy: SecAgg + DP on query results
- [ ] Test on eICU: "average LOS by diagnosis across hospitals"

### 4.2 Synthetic Data Generation

- [ ] DP-CTGAN integration (Gretel or SmartNoise SDK)
- [ ] Generate synthetic eICU data with formal DP guarantee
- [ ] Utility validation: train model on synthetic, test on real
- [ ] Fidelity metrics: column distributions, correlations, ML utility gap

### 4.3 Homomorphic Encryption (Paillier)

- [ ] `fl_common/he.py` — Paillier-encrypted FedAvg
- [ ] Clients encrypt updates with shared public key
- [ ] Server sums ciphertexts (homomorphic addition)
- [ ] Decrypt only after aggregation
- [ ] Benchmark: communication overhead, runtime vs plaintext FedAvg

### 4.4 SMPC (Secret Sharing)

- [ ] Shamir secret sharing for model updates (threshold t-of-n)
- [ ] Tolerates up to t-1 dropouts (vs SecAgg all-or-nothing)
- [ ] Integration with CrypTen or MP-SPDZ
- [ ] Benchmark vs SecAgg: dropout tolerance, overhead

---

## Phase 5: Deployment Hardening (Priority: MEDIUM)

### 5.1 Air-Gapped / On-Prem Deployment

| Requirement | Implementation |
|-------------|---------------|
| No internet access | Pre-built Docker images, offline pip wheels |
| No cloud dependency | Terraform for on-prem VMs (vSphere/OpenStack provider) |
| Private registry | Docker save/load workflow, internal registry support |
| Data ingestion | USB/SFTP data pipeline, no S3 dependency |

- [ ] `deploy/offline/` — scripts to package everything for air-gap
- [ ] `deploy/offline/build_offline_bundle.sh` — creates tarball with images + wheels + certs
- [ ] Test: deploy full FL cluster on isolated VMs with no internet

### 5.2 Access Control / RBAC

| Role | Permissions |
|------|------------|
| **Admin** | Create experiments, manage clients, view all results |
| **Operator** | Start/stop experiments, view metrics |
| **Data Owner** | Connect client, view own contributions |
| **Auditor** | Read-only access to audit logs, provenance, privacy ledger |
| **Model Consumer** | Query/download final model (if authorized) |

- [ ] Role definitions in experiment config
- [ ] API-level enforcement (Flower custom authentication)
- [ ] Audit log records who did what

### 5.3 Data Sovereignty Enforcement

- [ ] Geo-tagging of client nodes (region metadata in node config)
- [ ] Policy engine: reject clients from unauthorized regions
- [ ] Cross-border transfer logging (which updates crossed jurisdictions)
- [ ] Configurable per-experiment: `allowed_regions: ["SG", "MY"]`

### 5.4 Multi-Classification Support

For defence/intel — handle different security levels in the same FL pipeline.

- [ ] Data classification labels (UNCLASSIFIED, RESTRICTED, SECRET)
- [ ] Strategy selection based on classification (SECRET -> TEE + SecAgg mandatory)
- [ ] Prevent downgrade: SECRET model cannot be served to UNCLASSIFIED endpoint
- [ ] Separate audit streams per classification level

---

## Phase 6: Testing + Validation (Priority: HIGH — gates all phases)

### 6.1 Expanded Test Matrix

| Test | Current | Target |
|------|---------|--------|
| FL convergence (9 strategies) | Tested | Add robust strategies (Krum, TrimmedMean, Bulyan) |
| Privacy attacks (MIA, DLG, canary) | Tested | Add poisoning attacks |
| Non-IID sweep | Tested (alpha 0.1-10.0) | Same |
| DP utility curve | Tested | Add per-model-size recommendations |
| TEE attestation | Not tested | End-to-end attestation verification |
| Audit log integrity | Not tested | HMAC verification, completeness check |
| Air-gap deployment | Not tested | Full offline deployment test |
| RBAC enforcement | Not tested | Role-based access test suite |

### 6.2 Scenario Files to Add

```
scenarios/
  poisoning_krum.yaml          # Krum vs label flip attack
  poisoning_trimmed_mean.yaml  # TrimmedMean vs gradient scaling
  federated_analytics.yaml     # Aggregate stats on eICU
  synthetic_data.yaml          # DP-CTGAN generation + utility test
  he_fedavg.yaml               # Paillier-encrypted FedAvg
  airgap_smoke.yaml            # Minimal test for offline deployment
```

---

## Priority Summary

| Phase | Priority | Effort | Govt Impact |
|-------|----------|--------|-------------|
| **1. Integrity + Robustness** | HIGH | 1-2 weeks | Addresses #1 govt concern: compromised participants |
| **2. Audit + Provenance** | HIGH | 1-2 weeks | Non-negotiable for regulated deployment |
| **3. TEE as PET** | HIGH | 2-3 weeks | Proves server operator cannot access updates |
| **4. Additional PETs** | MEDIUM | 3-4 weeks | Broadens use cases (analytics, synthetic data, HE) |
| **5. Deployment Hardening** | MEDIUM | 2-3 weeks | Enables classified/air-gapped environments |
| **6. Testing + Validation** | HIGH (gates all) | Ongoing | Confidence for procurement |

**Model/data expansion (additional):**

| Phase | Effort | Models Added |
|-------|--------|-------------|
| **A. Quick wins (swap models)** | ~~1-2 weeks~~ | ~~+7 models~~ **DONE: LogReg, 1D CNN, ResNet-small, Autoencoder, TabNet** |
| **B. New data types** | ~~6-8 weeks~~ Partial | **DONE: anomaly, mortality, drug, satellite, readmission.** Remaining: U-Net, YOLO, GNN, BERT, Whisper, 3D CNN |
| **C. New FL paradigms** | 10-14 weeks | +5 models (Stable Diffusion, CLIP, XGBoost, RL, Recommender) |

**Total estimated scope:** 10-15 weeks (core govt readiness) + 8-12 weeks (Phase A+B model expansion) + 10-14 weeks (Phase C advanced). Phases A and B can run in parallel with core phases.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-15 | Plan created | Gap analysis identified 3 critical gaps: audit, byzantine robustness, TEE-as-PET |
| 2026-05-15 | Phase 1+2 prioritised first | Integrity and audit are non-negotiable for govt; TEE requires more infra work |
| 2026-05-15 | Expanded to 30 models, 14 data types, 12 govt domains | Comprehensive coverage needed for govt procurement across all agencies |
| 2026-05-15 | Distributed infra operational | Installed NVIDIA 595.71.05 + container-toolkit on 5 clients. Created `run_distributed.sh`, `runners/run_client.py`. Fraud 11/11 passed in distributed mode. Full 8-task run launched. |
| 2026-05-15 | Deployment guide started | Comprehensive guide covering infrastructure, distributed deployment, operations, and troubleshooting |
| 2026-05-15 | Phase A complete | Added 5 new models (autoencoder, logreg, cnn1d, tabnet_simple, resnet_small) + 5 new tasks (anomaly, mortality, drug, satellite, readmission) |
| 2026-05-15 | Secure inference module | 4 working demos (Paillier HE, TEE, secret sharing, functional encryption) + resource guide (CrypTFlow2, CrypTen, Concrete ML, DTC protocols) |
| 2026-05-15 | Pushed to GitHub | govtech-data-practice/Fl_deployment — 124 files, no secrets, no Claude mentions |
