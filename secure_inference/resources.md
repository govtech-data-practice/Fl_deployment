# Secure Inference — Resource Guide

A comprehensive reference for privacy-preserving machine learning inference,
covering frameworks, cloud services, standards, and research.

---

## 1. Industrial Frameworks & Libraries

### Homomorphic Encryption (HE)

| Project | Organization | Description | Language | Link |
|---------|-------------|-------------|----------|------|
| **TenSEAL** | OpenMined | CKKS/BFV on PyTorch tensors; easiest on-ramp for ML+HE | Python/C++ | https://github.com/OpenMined/TenSEAL |
| **OpenFHE** | Duality Technologies | Most complete HE library (BGV, BFV, CKKS, TFHE, threshold HE) | C++ | https://github.com/openfheorg/openfhe-development |
| **Concrete ML** | Zama | Compile scikit-learn / PyTorch models to run under TFHE | Python | https://github.com/zama-ai/concrete-ml |
| **EVA** | Microsoft | HE compiler — translates tensor programs to optimized CKKS circuits | Python/C++ | https://github.com/microsoft/EVA |
| **SEAL** | Microsoft | BFV and CKKS schemes; foundation for many higher-level tools | C++ | https://github.com/microsoft/SEAL |
| **HElib** | IBM | BGV and CKKS with bootstrapping support | C++ | https://github.com/homenc/HElib |
| **Lattigo** | Tune Insight | Go-native CKKS and BGV with multiparty extensions | Go | https://github.com/tuneinsight/lattigo |

### Secure Multi-Party Computation (MPC)

| Project | Organization | Description | Link |
|---------|-------------|-------------|------|
| **CrypTen** | Meta | PyTorch-based MPC framework; secret sharing behind familiar tensor API | https://github.com/facebookresearch/CrypTen |
| **MP-SPDZ** | KU Leuven / Bristol | General MPC framework supporting 30+ protocols (SPDZ, MASCOT, semi-honest, malicious) | https://github.com/data61/MP-SPDZ |
| **ABY** | TU Darmstadt | Mixed-protocol 2PC (Arithmetic, Boolean, Yao) with automatic conversions | https://github.com/encryptogroup/ABY |
| **ABY3** | Visa Research | 3-party mixed-protocol MPC for ML (honest majority) | https://github.com/ladnir/aby3 |
| **SecretFlow** | Ant Group | Full-stack privacy platform: MPC, FL, TEE, DP, PSA | https://github.com/secretflow/secretflow |
| **Rosetta** | Ant Group | TensorFlow-integrated MPC (3PC semi-honest) | https://github.com/LatticeX-Foundation/Rosetta |

### 2PC Inference Systems (purpose-built for neural network inference)

| Project | Organization | Description | Paper |
|---------|-------------|-------------|-------|
| **CrypTFlow2** | Microsoft Research | 2PC inference with OT-based protocols; semi-honest; handles ReLU efficiently | CCS 2020 |
| **DELPHI** | UC Berkeley | Preprocessing + online phase; client-aided garbled circuits for non-linear layers | USENIX Security 2020 |
| **Cheetah** | Alibaba DAMO | Fast 2PC CNN inference using function secret sharing for ReLU | USENIX Security 2022 |
| **EzPC** | Microsoft Research | High-level language that compiles to 2PC protocols (OT, GC, HE) | IEEE S&P 2019 |
| **GAZELLE** | Microsoft / MIT | HE for linear layers + garbled circuits for non-linear; pioneered the hybrid approach | USENIX Security 2018 |
| **MiniONN** | — | Oblivious neural network inference via mixed protocols | CCS 2017 |
| **XONN** | — | Binary neural networks with garbled circuits (very fast for binarized models) | USENIX Security 2019 |
| **Iron** | UC Berkeley | Function secret sharing for fast non-linear operations | NeurIPS 2022 |
| **Piranha** | — | GPU-accelerated MPC for neural network training and inference | USENIX Security 2022 |

---

## 2. Cloud TEE Services

### AWS Nitro Enclaves
- **What**: Isolated VMs (no persistent storage, no networking) attached to EC2 instances
- **Attestation**: Nitro Security Module (NSM) produces attestation documents signed by AWS
- **SDK**: `aws-nitro-enclaves-sdk` (Rust, C, Python via KMS integration)
- **Use case**: Key management, tokenization, secure inference
- **Docs**: https://docs.aws.amazon.com/enclaves/

### Azure Confidential Computing
- **Hardware**: Intel SGX (application enclaves), Intel TDX and AMD SEV-SNP (confidential VMs)
- **Services**: Confidential VMs, Confidential Containers (AKS), Confidential Ledger
- **Attestation**: Microsoft Azure Attestation (MAA) service
- **SDK**: Open Enclave SDK, Intel SGX SDK, Gramine (library OS for SGX)
- **Docs**: https://learn.microsoft.com/azure/confidential-computing/

### GCP Confidential Space
- **Hardware**: AMD SEV-SNP (confidential VMs)
- **Services**: Confidential VMs, Confidential GKE Nodes, Confidential Space
- **Attestation**: vTPM-based attestation tokens verified via Google's OIDC
- **Use case**: Multi-party data collaboration without a trusted third party
- **Docs**: https://cloud.google.com/confidential-computing/

### IBM Hyper Protect
- **Hardware**: IBM Secure Execution on IBM Z (s390x) and LinuxONE
- **Services**: Hyper Protect Virtual Servers, Hyper Protect Crypto Services
- **Feature**: Hardware-encrypted memory, no admin access (even IBM operators)
- **Docs**: https://www.ibm.com/cloud/hyper-protect-services

### Comparison Matrix

| Feature | AWS Nitro | Azure SGX | Azure SEV/TDX | GCP Conf. Space | IBM Hyper Protect |
|---------|-----------|-----------|----------------|-----------------|-------------------|
| Memory encryption | Yes (VM-level) | Yes (enclave) | Yes (VM) | Yes (VM) | Yes (hardware) |
| Attestation | NSM | DCAP/EPID | MAA | vTPM+OIDC | Attestation Service |
| GPU support | No | No | Limited | Limited | No |
| Max enclave memory | Parent instance | 256 GB (EPC) | VM memory | VM memory | 64 GB+ |
| Side-channel resistance | Hardware | Hardware | Hardware | Hardware | Hardware |

---

## 3. Standards & Regulations

### Privacy-Enhancing Technologies Standards

| Standard | Title | Relevance |
|----------|-------|-----------|
| **ISO/IEC 20889:2018** | Privacy-enhancing data de-identification techniques | Taxonomy of de-identification methods including encryption, perturbation |
| **ISO/IEC 27559:2022** | Privacy-enhancing data de-identification framework | Framework for selecting and applying de-identification techniques |
| **NIST SP 800-188** | De-Identifying Government Datasets | Guidelines for de-identification risk assessment |
| **HomomorphicEncryption.org Standard** | Homomorphic Encryption Standard | Community standard for HE parameter selection (security levels, encoding) |
| **ISO/IEC 4922-1:2023** | Secure multiparty computation — Part 1: General | Framework and terminology for MPC |

### Healthcare-Specific

| Standard | Relevance |
|----------|-----------|
| **HIPAA** (45 CFR 164) | PHI de-identification Safe Harbor / Expert Determination |
| **GDPR** (Art. 25, 32) | Data protection by design; pseudonymization as a safeguard |
| **FDA 21 CFR Part 11** | Electronic records — relevant when secure inference is part of a regulated SaMD |
| **HITECH Act** | Strengthens HIPAA; breach notification for PHI |

---

## 4. Research Benchmarks & Landmark Papers

### Benchmarks

| Benchmark | Description | Link |
|-----------|-------------|------|
| **MLPerf Inference** | Industry-standard latency/throughput benchmarks for ML inference (not privacy-specific, but the baseline) | https://mlcommons.org/en/inference-datacenter-11/ |
| **PPML Benchmarks** (CrypTFlow2) | End-to-end latency for ResNet-50, DenseNet-121, SqueezeNet on ImageNet under 2PC | CCS 2020 paper |
| **CryptoNets** | First demonstration of neural network inference on CKKS-encrypted data (Microsoft) | ICML 2016 |
| **SoK: Efficient Privacy-Preserving ML** | Systematization of knowledge — compares HE, GC, SS approaches across many models | IEEE S&P 2023 |

### Performance Reference Points (approximate, from literature)

| Model | Method | Latency | Communication | Paper |
|-------|--------|---------|---------------|-------|
| ResNet-32 (CIFAR-10) | CrypTFlow2 (2PC) | ~6s | ~1.2 GB | CCS 2020 |
| ResNet-50 (ImageNet) | Cheetah (2PC) | ~3.2s | ~2.3 GB | USENIX Sec 2022 |
| MNIST MLP | CryptoNets (HE) | ~250s | ~370 KB | ICML 2016 |
| MNIST MLP | Concrete ML (TFHE) | ~5s | N/A (single party) | Zama benchmarks |
| SqueezeNet | DELPHI (2PC) | ~15s | ~2 GB | USENIX Sec 2020 |

---

## 5. Decision Tree Classification (DTC) for Secure Inference

Secure evaluation of decision trees and ensemble methods is an active area
because DTs are widely used in healthcare (interpretable, tabular data).

### Approaches

#### Oblivious Decision Trees (ODT)
- **Idea**: Evaluate ALL paths in the tree; use oblivious selection to pick the
  correct leaf.  The server does not learn which path was taken.
- **Techniques**: Oblivious RAM (ORAM), PIR (private information retrieval)
- **Paper**: Bost et al., "Machine Learning Classification over Encrypted Data,"
  NDSS 2015.

#### HE-Friendly Decision Trees
- **Idea**: Encode comparisons (x_i < threshold) as polynomial operations
  compatible with HE.  Requires comparison protocols (e.g., using TFHE's
  programmable bootstrapping).
- **Paper**: Lu & Bhatt, "Poster: Efficient FHE-based Privacy-Enhanced
  Neural Network for AI-as-a-Service," CCS 2021.
- **Implementation**: Concrete ML can compile sklearn decision trees to FHE.

#### PDTE (Private Decision Tree Evaluation)
- **Protocol**: Client holds input x (encrypted), server holds decision tree T.
  Protocol reveals T(x) to client, nothing else.
- **Approaches**:
  - **Additive HE + OT**: Tai et al., "Privacy-Preserving Decision Trees
    Evaluation via Linear Functions," ESORICS 2017.
  - **Garbled circuits**: Kiss et al., "SoK: Modular and Efficient Private
    Decision Tree Evaluation," PoPETS 2019.
  - **Function secret sharing**: Boyle et al., applied to DT evaluation.
- **Complexity**: O(d * n) where d = depth, n = number of features.

#### XGBoost Secure Inference
- **Challenge**: XGBoost ensembles have hundreds of trees; naive per-tree
  evaluation is expensive.
- **Approaches**:
  - **Batched HE evaluation**: Encrypt all feature comparisons, evaluate all
    trees in parallel under HE.  Akavia et al., "Privacy-Preserving Decision
    Trees Training and Prediction," TOPS 2022.
  - **Function secret sharing for comparisons**: Each comparison x_i < t_j is
    evaluated via a distributed comparison gate.  Recent work from CryptoLab.
  - **SecureBoost** (FATE framework): Vertical FL for XGBoost training; can be
    adapted for inference.
  - **Concrete ML**: Compiles XGBoost models to TFHE circuits.
- **Reference**: Chen et al., "When Homomorphic Encryption Marries Secret
  Sharing: Secure Large-Scale Sparse Logistic Regression and Applications
  in Risk Control," KDD 2021.

#### Random Forest Secure Inference
- Similar to XGBoost but simpler (no gradient boosting); majority vote
  can be done with secure comparison protocols.
- **Paper**: Joye & Salehi, "Private yet Efficient Decision Tree Evaluation,"
  DBSec 2018.

### Summary Table

| Method | Tree Types | Non-Interactive? | Communication | Key Advantage |
|--------|-----------|-------------------|---------------|---------------|
| Concrete ML (TFHE) | DT, RF, XGB | Yes (FHE) | Ciphertext only | Easy to use, sklearn compatible |
| PDTE (HE + OT) | DT | No (2 rounds) | O(d) ciphertexts | Efficient for single trees |
| GC-based | DT, RF | No (1 round after setup) | O(d * n) gates | Low online latency |
| FSS-based | DT, RF, XGB | No (2 rounds) | O(n) keys | Scales well with ensemble size |
| SecureBoost (MPC) | XGB | No (multi-round) | Varies | Integrates with FATE FL |

---

## 6. Choosing the Right Approach

### Decision Flowchart

```
Is the model linear (logistic regression, linear SVM)?
  YES -> Paillier / CKKS homomorphic encryption (simple, non-interactive)
  NO  -> Continue...

Is the model a decision tree / ensemble?
  YES -> Concrete ML (TFHE) for easy deployment, or PDTE protocols
  NO  -> Continue...

Is the model a deep neural network?
  YES ->
    Is latency critical (< 100ms)?
      YES -> TEE (Nitro Enclaves, SGX) — fastest option
      NO  ->
        Is the model small (< 10M params)?
          YES -> 2PC (CrypTFlow2, Cheetah) or CrypTen
          NO  -> TEE or hybrid (HE for linear layers + TEE for non-linear)

Do you need multi-party input?
  YES -> MPC (CrypTen, MP-SPDZ, SecretFlow)
  NO  -> HE or TEE

Is the threat model malicious (not just honest-but-curious)?
  YES -> Malicious-secure MPC (SPDZ) or TEE with attestation
  NO  -> Semi-honest protocols (faster)
```

### Quick Comparison

| Approach | Latency Overhead | Communication | Trust Assumption | Non-Linear Support |
|----------|-----------------|---------------|------------------|-------------------|
| **HE (CKKS/BFV)** | 1000-10000x | Low (one-way) | Crypto hardness (RLWE) | Limited (polynomial approx.) |
| **HE (TFHE)** | 100-1000x | Low (one-way) | Crypto hardness (LWE) | Good (bootstrapping) |
| **2PC (GC+OT)** | 10-100x | High (GB-scale) | OT correlation robustness | Good (any boolean circuit) |
| **MPC (3PC)** | 5-50x | Medium | Honest majority | Good |
| **TEE** | 1-2x | Low | Hardware vendor trust | Full (native execution) |
| **Functional Encryption** | Varies | Low | Crypto hardness | Very limited (inner product) |

---

## 7. Getting Started — Recommended Path

1. **Easiest entry point**: [CrypTen](https://github.com/facebookresearch/CrypTen)
   — if you already use PyTorch, CrypTen is the least friction.

2. **For sklearn models**: [Concrete ML](https://github.com/zama-ai/concrete-ml)
   — compile decision trees, logistic regression, or small NNs to FHE.

3. **For production TEE**: Start with AWS Nitro Enclaves
   — good docs, well-integrated with KMS for key management.

4. **For research / flexibility**: [MP-SPDZ](https://github.com/data61/MP-SPDZ)
   — supports the widest range of protocols and security models.

5. **For healthcare FL**: Combine with this repository's FL framework:
   - Use `fl_common/secagg.py` for secure aggregation during training.
   - Use `fl_common/dp.py` for differential privacy guarantees.
   - Deploy trained models with one of the above secure inference methods.

---

## 8. Key References

1. Gentry, C. "Fully Homomorphic Encryption Using Ideal Lattices." STOC 2009.
2. Goldreich, O., Micali, S., Wigderson, A. "How to Play ANY Mental Game." STOC 1987.
3. Yao, A. "How to Generate and Exchange Secrets." FOCS 1986.
4. Dwork, C. "Differential Privacy." ICALP 2006.
5. Cheon, J.H. et al. "Homomorphic Encryption for Arithmetic of Approximate Numbers (CKKS)." ASIACRYPT 2017.
6. Mohassel, P., Zhang, Y. "SecureML: A System for Scalable Privacy-Preserving Machine Learning." IEEE S&P 2017.
7. Rathee, D. et al. "CrypTFlow2: Practical 2-Party Secure Inference." CCS 2020.
8. Huang, Z. et al. "Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference." USENIX Security 2022.
9. Mishra, P. et al. "DELPHI: A Cryptographic Inference Service for Neural Networks." USENIX Security 2020.
10. Abdalla, M. et al. "Simple Functional Encryption Schemes for Inner Products." PKC 2015.
