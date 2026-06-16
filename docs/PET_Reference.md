# Privacy-Enhancing Technologies — Reference, Checklist & Decision Matrix

**Version:** 1.0
**Last updated:** 2026-05-15

---

## 1. PET Taxonomy

```
Privacy-Enhancing Technologies
|
+-- Input Privacy (protect raw data)
|   +-- Federated Learning (data never leaves)
|   +-- Synthetic Data (release fake data)
|   +-- K-Anonymity / L-Diversity / T-Closeness (transform real data)
|
+-- Computation Privacy (protect during processing)
|   +-- Differential Privacy (noise injection)
|   +-- Secure Aggregation (mask individual updates)
|   +-- Homomorphic Encryption (compute on ciphertext)
|   +-- Secure Multi-Party Computation (secret sharing)
|   +-- Trusted Execution Environments (hardware isolation)
|
+-- Output Privacy (protect model/results)
|   +-- Differential Privacy (formal output guarantee)
|   +-- Model watermarking / fingerprinting
|   +-- Federated LoRA (share adapter, not full model)
|
+-- Verification
    +-- Privacy attacks (MIA, gradient inversion, canary extraction)
    +-- Privacy auditing (epsilon accounting, leakage testing)
```

---

## 2. Differential Privacy (DP)

### 2.1 Formal Definition

A randomised mechanism M satisfies (epsilon, delta)-DP if for all neighbouring datasets D, D' (differing in one record) and all sets S:

```
Pr[M(D) in S] <= exp(epsilon) * Pr[M(D') in S] + delta
```

- **epsilon (privacy budget):** Lower = more private. epsilon=0 is perfect privacy (random output). epsilon=inf is no privacy.
- **delta:** Probability of catastrophic failure (privacy guarantee doesn't hold). Must be << 1/N (dataset size).

### 2.2 DP Variants

| Variant | Where noise added | Trust model | Formal guarantee | Standard |
|---------|------------------|-------------|------------------|----------|
| **Central DP (CDP)** | Server adds noise to aggregate | Trusted server | (epsilon, delta)-DP | Google RAPPOR (2014) |
| **Local DP (LDP)** | Each client adds noise before sending | No trust needed | (epsilon, delta)-LDP | Apple (2017), Google Chrome |
| **Distributed DP (DDP)** | Noise split across clients, reconstructed at server | Semi-honest server | (epsilon, delta)-DP | Google (2023) |
| **Renyi DP (RDP)** | Analysis tool — tighter composition | N/A (accounting) | (alpha, epsilon)-RDP | Mironov (2017) |
| **Zero-Concentrated DP (zCDP)** | Analysis tool — Gaussian mechanism | N/A (accounting) | rho-zCDP | Bun & Steinke (2016) |
| **User-level DP** | Protects all records of one user | Trusted server | Per-user guarantee | Google (2023) |
| **Record-level DP** | Protects single record | Trusted server | Per-record guarantee | Standard |
| **Shuffle DP** | Shuffler anonymises before server | Trusted shuffler | Amplification by shuffling | Erlingsson et al. (2019) |
| **Label DP** | Only label is protected, features public | Trusted server | Weaker guarantee | Ghazi et al. (2021) |

### 2.3 DP Mechanisms

| Mechanism | Noise distribution | Best for | Sensitivity | Standard reference |
|-----------|-------------------|----------|-------------|-------------------|
| **Laplace** | Laplace(0, Delta_f/epsilon) | Low-dimensional queries | L1 | Dwork et al. (2006) |
| **Gaussian** | N(0, sigma^2), sigma = Delta_f * sqrt(2 ln(1.25/delta)) / epsilon | High-dimensional (FL gradients) | L2 | Dwork & Roth (2014) |
| **Discrete Gaussian** | Integer-valued Gaussian | Secure computation compatibility | L2 | Canonne et al. (2020) |
| **Exponential** | Pr[output] ~ exp(epsilon * quality / 2) | Categorical outputs | Quality function | McSherry & Talwar (2007) |
| **Randomised Response** | Flip answer with probability | Binary/categorical LDP | N/A | Warner (1965) |
| **Skellam** | Difference of two Poissons | Distributed DP + SecAgg | Integer L2 | Agarwal et al. (2021) |

### 2.4 State-of-the-Art DP for FL

| Approach | Paper/System | Key innovation | Epsilon achieved | Utility impact |
|----------|-------------|---------------|-----------------|----------------|
| **DP-SGD** | Abadi et al. (2016) | Per-example gradient clipping + Gaussian noise | Depends on rounds | Foundation of all FL DP |
| **DP-FTRL** | Kairouz et al. (2021) | Tree aggregation, no amplification needed | Better than DP-SGD for FL | Google production system |
| **DP-FedAvg** | McMahan et al. (2018) | Central DP on aggregated update | epsilon=8.0 for next-word | Google keyboard |
| **DP-FedSGD** | Noble et al. (2022) | User-level DP with optimal clipping | epsilon=1.0 achievable | Moderate for large N |
| **Adaptive clipping** | Andrew et al. (2021) | Quantile-based clip norm estimation | Reduces hyperparameter tuning | Small improvement |
| **Private selection** | Liu & Talwar (2019) | Private hyperparameter tuning | Pays epsilon cost for tuning | Critical for practice |
| **Poisson subsampling** | Balle et al. (2020) | Privacy amplification by subsampling | sqrt(q) * epsilon | Standard amplification |
| **PATE** | Papernot et al. (2017) | Teacher-student, data-dependent epsilon | Very low epsilon possible | Excellent for structured data |

### 2.5 Industrial DP Implementations

| Organisation | System | DP variant | Epsilon | Use case | Open source |
|-------------|--------|-----------|---------|----------|-------------|
| **Google** | RAPPOR | Local DP (randomised response) | epsilon=2.0-8.0 | Chrome usage statistics | Yes (github.com/google/rappor) |
| **Google** | DP-FTRL (FL) | Central DP (tree aggregation) | epsilon=8.0-18.0 | Gboard next-word prediction | Partial (TF Privacy) |
| **Google** | Differential Privacy Library | Central DP | Configurable | General analytics | Yes (github.com/google/differential-privacy) |
| **Apple** | Private aggregation | Local DP (hash + noise) | epsilon=1.0-8.0 per day | Emoji prediction, Safari | No (proprietary) |
| **Microsoft** | SmartNoise | Central DP | Configurable | SQL queries, synthetic data | Yes (github.com/opendp/smartnoise-sdk) |
| **Meta** | Opacus | Central DP (DP-SGD) | Configurable | PyTorch model training | Yes (github.com/pytorch/opacus) |
| **OpenDP** | OpenDP Library | Multiple | Configurable | General-purpose DP library | Yes (github.com/opendp/opendp) |
| **Tumult Labs** | Tumult Analytics | Central DP + zCDP | Configurable | Census (2020 US Census) | Yes (tmlt.io) |
| **LinkedIn** | DP analytics | Central DP | epsilon~5.0 | Ads measurement | No |
| **Uber** | FLEX/DP SQL | Central DP | Configurable | Internal analytics | Partial |
| **US Census** | TopDown Algorithm | Central DP + post-processing | epsilon=19.61 (2020 Census) | Population counts | Yes (das_decennial) |

### 2.6 DP Guarantees Comparison

| Guarantee level | Epsilon range | What it means | Practical impact | When to use |
|----------------|---------------|---------------|-----------------|-------------|
| **Strong** | 0.1 - 1.0 | Individual records nearly indistinguishable | Significant utility loss on small models | High-risk data (genomic, mental health) |
| **Moderate** | 1.0 - 10.0 | Meaningful privacy protection | Moderate utility loss | Standard PII (medical records, financial) |
| **Weak** | 10.0 - 100.0 | Limited privacy, detectable statistical changes | Minimal utility loss | Low-sensitivity aggregates |
| **Nominal** | >100.0 | Technically DP but practically meaningless | No utility loss | Compliance checkbox only |

**Our validated results:**

| Model | Params | Epsilon | Accuracy/AUC | Verdict |
|-------|--------|---------|-------------|---------|
| BiLSTM (sepsis) | 500K | 50.0 (Central) | 0.822 | Useful DP |
| BiLSTM (sepsis) | 500K | 10.0 (Central) | 0.763 | Moderate loss |
| BiLSTM (sepsis) | 500K | 50.0 (Local) | 0.980 | Minimal loss |
| BiLSTM (sepsis) | 500K | 10.0 (Local) | 0.958 | Acceptable |
| DenseNet-121 (chest) | 8M | 50.0 (Central) | 0.505 | DP destroys model |
| DenseNet-121 (chest) | 8M | Any (Local) | 0.500 | Random — unusable |
| Mistral 7B (LoRA) | 160MB adapter | sigma=0.05 | 0.373 QA | Best trade-off |
| Mistral 7B (LoRA) | 160MB adapter | sigma=0.50 | 0.113 QA | MIA drops to 0.0 |

**Key insight:** DP utility scales inversely with model size. For >10M params, use SecAgg or HE instead.

### 2.7 DP Privacy Accounting

| Accountant | Tightness | Composition | Library |
|-----------|-----------|-------------|---------|
| **Basic composition** | Loose | epsilon_total = sum(epsilon_i) | Manual |
| **Advanced composition** | Better | epsilon_total ~ sqrt(k) * epsilon | Manual |
| **Moments accountant** | Tight | RDP-based, converts to (eps, delta) | TF Privacy, Opacus |
| **RDP accountant** | Tightest (Gaussian) | Renyi divergence tracking | Our `fl_common/dp.py`, Opacus |
| **GDP (Gaussian DP)** | Analytical | f-DP framework, CLT-based | autodp |
| **PRV accountant** | Tightest (general)| Privacy Random Variable | Google dp-accounting |

```python
# Our implementation
from fl_common.dp import PrivacyAccountant
accountant = PrivacyAccountant(noise_multiplier=0.02, sample_rate=1.0, delta=1e-5)
accountant.step(num_steps=30)  # 30 FL rounds
print(f"Total epsilon = {accountant.get_epsilon():.2f}")
```

---

## 3. Secure Aggregation

### 3.1 Approaches

| Approach | Protocol | Trust model | Dropout tolerant | Communication | Libraries |
|----------|----------|-------------|-----------------|---------------|-----------|
| **Pairwise masking (SecAgg)** | Bonawitz et al. (2017) | Honest-but-curious server | No (all must participate) | O(N^2) setup, O(N) per round | Our `fl_common/secagg.py`, Flower |
| **SecAgg+** | Bell et al. (2020) | Honest-but-curious server | Yes (threshold) | O(N log N) setup | Google (TFF) |
| **SecAgg with SS** | Bonawitz et al. (2017) | Honest-but-curious server | Yes (Shamir threshold t-of-N) | O(N^2) | CrypTen |
| **Paillier HE** | Paillier (1999) | Honest-but-curious server | Yes | O(N) + ciphertext expansion | python-paillier, SEAL |
| **CKKS HE** | Cheon et al. (2017) | Honest-but-curious server | Yes | O(N) + large expansion | OpenFHE, TenSEAL |
| **Shamir Secret Sharing** | Shamir (1979) | t-of-N threshold | Yes (up to N-t failures) | O(N^2) | MP-SPDZ, CrypTen |
| **Additive Secret Sharing** | | 2+ servers | Yes | O(N) | ABY3, CrypTen |
| **Garbled Circuits** | Yao (1986) | 2-party | N/A | High constant | EMP-toolkit |
| **Function Secret Sharing** | Boyle et al. (2015) | 2+ servers | Yes | O(1) per query | N/A |

### 3.2 Our SecAgg Implementation (Pairwise Masking)

**Protocol:**
```
For each pair (i, j) where i < j:
  1. Shared seed s_ij (pre-agreed or via Diffie-Hellman)
  2. Generate mask M_ij = PRG(s_ij)  [same shape as model update]
  3. Client i adds +M_ij to their update
  4. Client j adds -M_ij to their update
  5. Server sums: all M_ij cancel, leaving clean aggregate
```

**Guarantee:** Server sees only the aggregate, not individual updates. Information-theoretic security (no computational assumptions).

**Limitations of our implementation:**
- Requires ALL N clients to participate (no dropout)
- Equal-weight averaging (1/N) — required for exact mask cancellation
- Validated AUC: 0.763 (vs FedAvg 0.811) — gap from equal weighting, not SecAgg itself

### 3.3 Paillier Homomorphic Encryption

**What it provides:** Clients encrypt model updates with a shared public key. Server sums ciphertexts (addition is homomorphic). Result is decrypted only after aggregation.

**Protocol:**
```
Setup:
  1. Generate Paillier key pair (pk, sk)
  2. Distribute pk to all clients, sk held by trusted party or threshold-shared

Per round:
  1. Client i encrypts update: c_i = Enc(pk, w_i)
  2. Server computes: c_agg = c_1 * c_2 * ... * c_N  (homomorphic addition)
  3. Trusted party decrypts: w_agg = Dec(sk, c_agg)
  4. Server distributes w_agg / N to clients
```

**Guarantee:** Server never sees individual updates in plaintext. Semantic security under Decisional Composite Residuosity assumption.

| Property | Value |
|----------|-------|
| Key size | 2048-4096 bits |
| Ciphertext expansion | ~32x (2048-bit key) |
| Encryption speed | ~1ms per 2048-bit plaintext |
| Addition (homomorphic) | ~0.1ms per ciphertext pair |
| Decryption | ~1ms per ciphertext |
| Supports | Addition only (partially homomorphic) |
| Does NOT support | Multiplication (use CKKS for both) |

**Libraries:**
| Library | Language | Notes |
|---------|----------|-------|
| python-paillier | Python | Simple, pure Python, slow for large models |
| SEAL (Microsoft) | C++/Python | BFV/CKKS, GPU-accelerated |
| TenSEAL | Python | CKKS on tensors, PyTorch-friendly |
| OpenFHE | C++/Python | CKKS/BFV/BGV, most complete |
| Lattigo | Go | CKKS/BFV, fast |
| HElib | C++ | BGV/CKKS, IBM |

### 3.4 MPC-Based Secure Aggregation

**Shamir Secret Sharing for FL:**

```
Setup:
  Threshold t (need t+1 shares to reconstruct, up to N-t-1 can drop out)

Per round:
  1. Client i splits update w_i into N shares: (s_i1, s_i2, ..., s_iN)
     using degree-t polynomial where w_i = P_i(0)
  2. Client i sends share s_ij to each other client j
  3. Each client j sums received shares: S_j = sum(s_ij for all i)
  4. Server collects t+1 values S_j, reconstructs sum(w_i) via Lagrange interpolation
```

**Guarantee:** Up to t colluding parties learn nothing about individual updates. Information-theoretic security.

| Property | Shamir SS | Additive SS | Garbled Circuits |
|----------|-----------|-------------|-----------------|
| Parties | N (any) | 2+ servers | 2 |
| Dropout tolerance | Up to N-t-1 | None | None |
| Communication | O(N^2) per round | O(N) per round | O(circuit size) |
| Security | Information-theoretic | Information-theoretic | Computational |
| Best for | FL with dropout | Simple aggregation | Complex computation |

### 3.5 SecAgg Comparison Matrix

| Method | Guarantees | Dropout | Communication overhead | Computation overhead | Model size limit | Implementation complexity |
|--------|-----------|---------|----------------------|---------------------|-----------------|--------------------------|
| **Pairwise masking** | Info-theoretic, server learns nothing | None (all required) | O(N^2) setup | Negligible | None | Low (our implementation) |
| **SecAgg+** | Info-theoretic, sparse graph | Yes (threshold) | O(N log N) setup | Low | None | Medium |
| **Paillier HE** | Computational (DCR) | Full (any can drop) | 32x ciphertext expansion | Moderate (encryption) | Practical up to ~10M params | Medium |
| **CKKS HE** | Computational (RLWE) | Full | 10-100x expansion | High | Larger than Paillier | High |
| **Shamir SS** | Info-theoretic (threshold) | Yes (up to N-t-1) | O(N^2) shares | Moderate (polynomial eval) | None | Medium-High |
| **Additive SS + 2 servers** | Info-theoretic | None (both servers needed) | 2x (one share to each) | Negligible | None | Low |
| **TEE (enclave)** | Hardware-based | Full | None (plaintext inside) | None | Limited by enclave memory | High (infra) |

---

## 4. Trusted Execution Environments (TEE)

### 4.1 Platforms

| Platform | Provider | Attestation | Memory limit | Key feature |
|----------|----------|-------------|-------------|-------------|
| **AWS Nitro Enclaves** | AWS | Nitro attestation document (PCRs) | ~16 GB | Isolated from host OS, no network |
| **AMD SEV-SNP** | AWS/GCP/Azure | vTPM + firmware attestation | Full VM memory | Memory encryption, no enclave limit |
| **Intel TDX** | Azure/GCP | DCAP remote attestation | Full VM memory | Successor to SGX for VMs |
| **Intel SGX** | Azure | DCAP remote attestation | ~256 MB (EPC) | Process-level isolation |
| **GCP Confidential Space** | GCP | Attestation verifier token | Full VM | Container-level confidential compute |
| **ARM CCA** | - | Realm attestation | Device-dependent | Mobile/edge TEE |

### 4.2 TEE for FL Aggregation

| Property | Without TEE | With TEE |
|----------|------------|----------|
| Server sees individual updates | Yes | No (encrypted in enclave memory) |
| Server sees aggregate | Yes | Yes (needed for model distribution) |
| Verifiable computation | No (trust server) | Yes (attestation proves code integrity) |
| Side-channel resistance | N/A | Partial (platform-dependent) |
| Performance overhead | None | 5-30% (memory encryption) |

---

## 5. Combined PET Stacks

### 5.1 Recommended Stacks by Threat Model

| Threat model | What you're protecting against | Recommended stack | Epsilon | Validated? |
|-------------|-------------------------------|-------------------|---------|-----------|
| **Honest server, no adversary** | Regulatory compliance only | FL + audit logging | N/A | Yes |
| **Curious server** | Server inspects individual updates | FL + SecAgg (pairwise or Paillier) | N/A | Yes (SecAgg) |
| **Curious server + formal guarantee** | Server inspects + need math proof | FL + DP (Central) + SecAgg | 10-50 | Yes |
| **Untrusted server** | Server is actively malicious | FL + Local DP + SecAgg | 1-10 | Yes |
| **Untrusted server + verification** | Malicious server + prove integrity | FL + TEE + SecAgg | N/A | Partial (Nitro runtime) |
| **Compromised clients** | Malicious participants poison model | FL + Byzantine aggregation (Krum) | N/A | Planned (Phase 1) |
| **Full adversary** | Server + clients collude | FL + Local DP + SMPC + TEE | 1-10 | Planned |
| **Data must never leave** | Strictest (genomic, classified) | Local training only, or FL + HE + TEE | N/A | Planned (HE) |

### 5.2 Stack Compatibility

| | FL | DP (Central) | DP (Local) | SecAgg | Paillier HE | SMPC | TEE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FL** | - | Yes | Yes | Yes | Yes | Yes | Yes |
| **DP (Central)** | Yes | - | No (pick one) | Yes | Yes | Yes | Yes |
| **DP (Local)** | Yes | No | - | Redundant | Redundant | Redundant | Yes |
| **SecAgg** | Yes | Yes | Redundant | - | Alternative | Alternative | Yes |
| **Paillier HE** | Yes | Yes | Redundant | Alt | - | No | Yes |
| **SMPC** | Yes | Yes | Redundant | Alt | No | - | Yes |
| **TEE** | Yes | Yes | Yes | Yes | Yes | Yes | - |

---

## 6. PET Decision Matrix

### 6.1 By Data Type

| Data type | Sensitivity | Model size | Recommended PETs | DP feasible? |
|-----------|-----------|-----------|-----------------|-------------|
| **Tabular (vitals, labs)** | Medium | <1M params | FL + DP (eps 10-50) | Yes |
| **Tabular (financial)** | Medium-High | <1M params | FL + DP + SecAgg | Yes |
| **Tabular (genomic)** | Very High | <1M params | FL + DP (eps 1-5) + TEE | Yes (strong) |
| **Medical images (X-ray)** | High | 8-25M params | FL + SecAgg | No (DP destroys) |
| **Medical images (CT/MRI)** | High | 25-50M params | FL + SecAgg + TEE | No |
| **Clinical text** | Very High | 7B+ (LoRA 160MB) | FL LoRA + DP on adapter | Marginal |
| **ECG/waveforms** | Medium | <2M params | FL + DP (eps 10-50) | Yes |
| **Transaction data** | Medium | <1M params | FL + DP + SecAgg | Yes |
| **Biometrics** | Very High | 5-25M params | FL + SecAgg + TEE | No |
| **Satellite imagery** | Medium-High | 25M+ params | FL + SecAgg | No |
| **Government documents** | High-Very High | 7B+ (LoRA) | FL LoRA + DP + TEE | On adapter only |

### 6.2 By Regulation

| Regulation | Key requirement | Minimum PET stack | Formal DP required? |
|-----------|----------------|-------------------|-------------------|
| **HIPAA** (US healthcare) | De-identification, minimum necessary | FL + de-identification | No (Safe Harbor suffices) |
| **HIPAA** + research | Expert Determination | FL + DP (any epsilon) | Recommended |
| **GDPR** (EU) | Data minimisation, DPIA, right to erasure | FL + DP | Recommended for DPIA |
| **PDPA** (Singapore) | Consent, purpose limitation | FL + SecAgg + audit | No |
| **PIPL** (China) | Data localisation, separate consent | FL (data stays local) + TEE | No |
| **21 CFR Part 11** (FDA) | Audit trail, validation | FL + audit + TEE | No (audit is key) |
| **AI Act** (EU) | Transparency, risk assessment | FL + DP + explainability | Recommended (high-risk AI) |
| **CCPA/CPRA** (California) | Consumer data rights | FL + DP | Recommended |
| **PCI DSS** (payments) | Cardholder data encryption | FL + SecAgg or HE + TLS | No (encryption is key) |
| **Classified/SECRET** | Compartmentalisation | FL + TEE + SMPC | Depends on classification |

### 6.3 By Operational Constraint

| Constraint | Recommendation | Avoid |
|-----------|---------------|-------|
| **Limited bandwidth** | FL LoRA (160MB not 14GB) | Full model FedAvg on LLMs |
| **Client dropout expected** | Shamir SS (threshold) or Paillier HE | Pairwise SecAgg (all required) |
| **No trusted server** | Local DP + SecAgg, or SMPC | Central DP (trusts server) |
| **Real-time / low latency** | FL + lightweight DP | HE (1000x overhead), SMPC (O(N^2)) |
| **Air-gapped network** | FL + Local DP (no external dependency) | TEE attestation (needs verification service) |
| **Heterogeneous models** | Federated Distillation or FedLoRA | Standard FedAvg (requires same architecture) |
| **Must prove computation** | TEE with attestation | Pure software (no verification) |
| **100+ clients** | SecAgg+ (O(N log N)) or Paillier | Shamir SS (O(N^2) communication) |

---

## 7. PET Checklist

### 7.1 Before Deployment

**Data classification:**
- [ ] Identify data sensitivity level (public / internal / confidential / restricted)
- [ ] Identify applicable regulations (HIPAA, GDPR, PDPA, etc.)
- [ ] Determine if formal privacy guarantee (epsilon) is required
- [ ] Assess threat model: who is untrusted? (server, clients, network, all)

**PET selection:**
- [ ] Choose DP variant based on trust model (Central if server trusted, Local if not)
- [ ] Choose SecAgg method based on dropout tolerance (pairwise if no dropout, Shamir/Paillier if dropout expected)
- [ ] Determine if TEE is required (regulated environment, untrusted server operator)
- [ ] Verify PET compatibility with model size (DP fails on >10M params)
- [ ] Set privacy budget (epsilon) appropriate for data sensitivity
- [ ] Determine composition bounds (how many experiments before budget exhausted)

**Infrastructure:**
- [ ] TLS enabled on all FL communication channels
- [ ] Certificates use strong curves (EC P-256 minimum)
- [ ] No `--insecure` flags in any production command
- [ ] TEE attestation verified by clients (if using TEE)
- [ ] Key management plan for HE keys (if using HE)

### 7.2 During Training

**Privacy budget:**
- [ ] RDP accountant tracking cumulative epsilon per round
- [ ] Alert threshold set (e.g., epsilon > 10 triggers review)
- [ ] Per-experiment epsilon logged in audit trail
- [ ] Composition across experiments tracked (not just within)

**SecAgg / HE:**
- [ ] All clients participating (pairwise SecAgg) or threshold met (Shamir/Paillier)
- [ ] Mask/share verification (if protocol supports)
- [ ] No individual updates visible in server logs

**Integrity:**
- [ ] Client updates within expected magnitude (detect poisoning)
- [ ] Byzantine aggregation active if threat model includes malicious clients
- [ ] Model hash recorded after each aggregation round

### 7.3 After Training

**Privacy verification:**
- [ ] Run MIA (Membership Inference Attack) on final model
- [ ] Run canary extraction test (if text/LLM model)
- [ ] Run gradient inversion test (if concerned about update leakage)
- [ ] Compare attack success with/without DP: expect significant drop
- [ ] Record final epsilon spend in privacy ledger

**Compliance:**
- [ ] Privacy budget within approved threshold
- [ ] Audit trail complete (rounds, participants, epsilon per round)
- [ ] Model provenance documented (which data sources, which rounds)
- [ ] Attestation logs archived (if TEE used)
- [ ] Results JSON + markdown summary generated

**Acceptance criteria:**

| Attack | Without DP (baseline) | With DP (target) | Status |
|--------|---------------------|------------------|--------|
| MIA advantage | Measure | < 0.1 (near random) | |
| Canary extraction | Measure | < 10% leaked | |
| Gradient cosine similarity | Measure | < 0.1 | |
| 5-gram text overlap | Measure | < 1% | |

---

## 8. Formal Guarantee Summary

| PET | Guarantee type | Formal statement | Assumptions | Breakable by |
|-----|---------------|-----------------|-------------|-------------|
| **(eps,delta)-DP** | Mathematical | Pr[M(D) in S] <= e^eps * Pr[M(D') in S] + delta | Correct implementation | Implementation bugs, floating-point errors |
| **Local DP** | Mathematical | Same as DP, but per-client | No trust in server | Implementation bugs |
| **SecAgg (pairwise)** | Information-theoretic | Server learns only sum(w_i), nothing about individual w_i | All clients honest, all participate | Client dropout, collusion of N-1 clients |
| **Paillier HE** | Computational | IND-CPA under DCR assumption | Key pair secure | Quantum computers (poly-time factoring) |
| **CKKS HE** | Computational | IND-CPA under RLWE assumption | Key pair secure, noise budget managed | Post-quantum attacks (RLWE believed quantum-resistant) |
| **Shamir SS (t-of-N)** | Information-theoretic | Up to t colluding parties learn nothing | Fewer than t+1 collude | t+1 or more parties collude |
| **TEE (Nitro)** | Hardware-based | Code + data isolated from host OS | Hardware not compromised | Physical access, side-channel attacks |
| **TEE (SEV-SNP)** | Hardware + firmware | Memory encrypted, VM isolated | AMD firmware trusted | Firmware vulnerabilities |
| **Federated Learning** | None (architectural) | Data stays local | Clients honest, no model inversion | Gradient inversion, MIA, memorisation |

### Post-Quantum Status

| PET | Quantum-safe? | Notes |
|-----|-------------|-------|
| DP | Yes | Information-theoretic, no crypto |
| SecAgg (pairwise masks) | Yes | PRG can use quantum-safe primitives |
| Paillier HE | **No** | Based on factoring — broken by Shor's algorithm |
| CKKS/BFV HE | Believed yes | Based on lattice problems (RLWE) |
| Shamir SS | Yes | Information-theoretic |
| TLS (EC P-256) | **No** | ECC broken by quantum. Migrate to ML-KEM / ML-DSA |
| TEE attestation | Depends | Attestation signatures may use quantum-vulnerable crypto |

---

## 9. Implementation Status in This Sandbox

| PET | Status | File | Notes |
|-----|--------|------|-------|
| FL (FedAvg, FedProx) | Implemented + validated | `fl_common/strategies.py` | 11 strategies |
| SCAFFOLD | Implemented + validated | `fl_common/strategies.py`, `fl_common/scaffold.py` | Control variates |
| DP (Central) | Implemented + validated | `fl_common/dp.py` | Gaussian mechanism, RDP accountant |
| DP (Local) | Implemented + validated | `fl_common/dp.py` | Per-client clipping + noise |
| SecAgg (pairwise) | Implemented + validated | `fl_common/secagg.py` | No dropout tolerance |
| Federated LoRA | Implemented + validated | `models/mistral/` | Mistral 7B QLoRA |
| MIA attack | Implemented + validated | `privacy/test_privacy.py` | Shadow model approach |
| Gradient inversion (DLG) | Implemented + validated | `privacy/test_privacy.py` | Cosine similarity metric |
| Canary extraction | Implemented + validated | `privacy/attack_suite.py` | Planted MRN/SSN recovery |
| TEE (Nitro runtime) | Tested (as runtime) | `deploy/` | Not yet as PET (no attestation) |
| Paillier HE | **Planned** | `fl_common/he.py` | Phase 4 |
| SMPC (Shamir) | **Planned** | `fl_common/smpc.py` | Phase 4 |
| SecAgg+ (dropout) | **Planned** | `fl_common/secagg.py` | Phase 4 |
| TEE (attestation) | **Planned** | `fl_common/attestation.py` | Phase 3 |
| Krum / TrimmedMean | **Planned** | `fl_common/strategies.py` | Phase 1 |
| Synthetic data (DP-CTGAN) | **Planned** | TBD | Phase 4 |
| Federated Analytics | **Planned** | `fl_analytics/` | Phase 4 |
