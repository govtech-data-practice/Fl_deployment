# Federated Learning Platform -- Comprehensive Deployment Guide

**Version:** 3.0
**Date:** 2026-06-02
**Platform:** AWS EC2 (configurable region), Amazon Linux 2023, NVIDIA L4 GPU

---

## About This Document

This guide is written for technical staff in government agencies, healthcare organizations, and regulated industries who need to deploy a production federated learning (FL) system. It assumes you understand IT infrastructure (Linux, Docker, AWS, networking) but does not assume prior knowledge of machine learning, federated learning, or cryptography.

Every concept is explained before it is used. Every command is accompanied by an explanation of what it does and why. Every architecture decision includes the reasoning behind it and the alternatives that were considered.

If you follow this guide from start to finish, you will have a working federated learning cluster that trains machine learning models on distributed data without ever centralizing that data.

**How to read this document:**

- **Sections 1-2** (How Federated Learning Works, Privacy Guarantees) provide conceptual foundations. Read these first if you are new to federated learning.
- **Sections 3-10** (Architecture through Deployment) are the step-by-step deployment instructions. Follow these in order for your first deployment.
- **Sections 11-13** (Tasks, Adapters, Secure Inference) describe advanced capabilities. Read these when you need to understand specific features.
- **Sections 14-21** (Monitoring through Troubleshooting) are operational guides. Reference these during day-to-day operations.
- **Appendices** provide reference material: a glossary, file index, and procedures for adding new sites and tasks.

**What changed from Version 2.1 to Version 3.0:**

- Added Sections 1, 2, 12, and 13 (previously absent): comprehensive explanations of FL concepts, privacy analysis, the federated adapter framework, and secure inference.
- Added explanatory context to every section, table, and code block.
- Added a strategy selection guide (Section 11.3) and privacy attack testing guide (Section 11.4).
- Added a glossary (Appendix E) and quick start checklist (Appendix F).
- All technical content from Version 2.1 has been preserved in full.

---

## Table of Contents

1. [How Federated Learning Works](#1-how-federated-learning-works)
2. [Privacy Guarantees and Their Limits](#2-privacy-guarantees-and-their-limits)
3. [Architecture](#3-architecture)
4. [Prerequisites and Dependencies](#4-prerequisites-and-dependencies)
5. [Configuration Management](#5-configuration-management)
6. [Infrastructure Setup](#6-infrastructure-setup)
7. [Docker Image](#7-docker-image)
8. [Data Pipeline](#8-data-pipeline)
9. [TLS and PKI](#9-tls-and-pki)
10. [Deployment](#10-deployment)
11. [Tasks and Strategies](#11-tasks-and-strategies)
12. [Federated Adapter Framework](#12-federated-adapter-framework)
13. [Secure Inference](#13-secure-inference)
14. [Monitoring and Observability](#14-monitoring-and-observability)
15. [Security](#15-security)
16. [Backup and Recovery](#16-backup-and-recovery)
17. [Capacity Planning](#17-capacity-planning)
18. [Version Management and Rollback](#18-version-management-and-rollback)
19. [Incident Response Runbook](#19-incident-response-runbook)
20. [Cost Management](#20-cost-management)
21. [Troubleshooting](#21-troubleshooting)
22. [Appendices](#appendices)

---

## 1. How Federated Learning Works

### 1.1 The Problem: Data That Cannot Move

Many organizations hold sensitive data -- patient health records, financial transactions, classified documents -- that cannot be copied to a central location. Regulations like HIPAA, GDPR, and national data sovereignty laws prohibit it. Even when regulations allow data movement, organizations may not trust each other enough to share raw data.

Yet these organizations often want to collaborate. A model trained on data from five hospitals is better than a model trained on data from one hospital, because it has seen more examples of rare diseases and more diversity in patient populations. A fraud detection model trained on transactions from multiple banks catches more fraud patterns.

Federated learning solves this problem. Instead of bringing data to the model, it brings the model to the data.

### 1.2 What Is a Machine Learning Model?

Before explaining federated learning, it helps to understand what a machine learning model is. If you already know this, skip to Section 1.3.

A machine learning model is a mathematical function with adjustable parameters (also called "weights"). The function takes an input (e.g., a patient's vital signs) and produces a prediction (e.g., the probability that the patient will develop sepsis). The parameters determine how the function maps inputs to outputs.

**Training** is the process of adjusting these parameters so the model makes good predictions. During training, the model sees many examples of known input-output pairs (e.g., past patients whose sepsis outcomes are known). For each example, it makes a prediction, compares it to the known answer, and adjusts its parameters slightly to reduce the error. After seeing thousands or millions of examples, the parameters converge to values that produce accurate predictions on new, unseen inputs.

The number of parameters varies by model complexity. A simple logistic regression for hospital readmission prediction might have 10,000 parameters. A neural network for sepsis prediction might have 500,000 parameters. A language model for document classification might have 1 billion parameters. More parameters generally means the model can capture more complex patterns, but also requires more data to train effectively.

### 1.3 The Core Loop: Train Locally, Aggregate Globally

Federated learning works through a repeated cycle of four steps:

1. **Distribution.** A central server sends a copy of the current model to each participating site (called "clients" or "nodes"). In round one, this is an untrained model with random weights. In subsequent rounds, it is the model that was improved in the previous round.

2. **Local training.** Each site trains the model on its own private data. This is standard machine learning training -- the model sees examples, makes predictions, measures errors, and adjusts its internal parameters (called "weights") to reduce those errors. The site's raw data never leaves the site.

3. **Update transmission.** Each site sends its updated model weights (or the difference between the old weights and new weights, called a "gradient") back to the central server. These updates are numerical arrays -- not the raw data itself. However, as we discuss in Section 2, these updates do carry some information about the training data.

4. **Aggregation.** The server combines the updates from all sites into a single improved model. The simplest method, called FedAvg (Federated Averaging), takes a weighted average of all the updates. More sophisticated methods exist for handling situations where sites have very different data distributions (see Section 11.2 for the full strategy reference).

This cycle repeats for a configured number of "rounds" (typically 3-20 rounds, depending on the task). After all rounds complete, the final aggregated model is the output. Each site can then use this model for predictions on their own data.

### 1.4 Why This Matters for Government and Healthcare

For government agencies and healthcare organizations, federated learning offers a way to:

- **Comply with data sovereignty requirements.** Patient data, classified documents, and citizen records stay within the organization's infrastructure at all times.
- **Collaborate across jurisdictions.** Agencies in different countries or states can jointly train a model without establishing data-sharing agreements for raw data.
- **Improve model quality.** Rare conditions, unusual fraud patterns, and edge cases that one organization rarely encounters may be common at another. Federated learning captures this diversity.
- **Maintain audit trails.** Each organization controls its own data and can independently verify what information left its network (only model updates, not raw data).

### 1.5 Types of Federated Learning

This platform supports three types of federated learning, each suited to different organizational relationships:

**Horizontal Federated Learning (HFL)** is the most common type. All sites have the same type of data (same columns/features) but different records (different patients or transactions). Example: five hospitals each have patient vital signs and lab results, but for different patients. This is the default mode for most tasks in this platform.

**Vertical Federated Learning (VFL)** applies when sites have data about the same entities but different features. Example: a hospital has patient diagnoses and a pharmacy has prescription histories, and they want to jointly predict drug interactions. In VFL, each site trains part of the model (the part that processes its features), and the partial results are combined without revealing raw features. The `vfl` task in this platform demonstrates this approach.

**Split Learning** is a variant where the model itself is split between sites. The bottom layers run at the client (processing raw data into abstract representations), and the top layers run at the server (making final predictions from those representations). Only the intermediate representations (called "activations") cross the network boundary. The `split` task demonstrates this approach.

**Comparing HFL, VFL, and Split Learning:**

| Aspect | HFL | VFL | Split Learning |
|--------|-----|-----|---------------|
| Data structure | Same features, different records | Same records, different features | Same features, different records |
| What crosses network | Full model updates | Partial model outputs | Activations (intermediate representations) |
| Privacy of raw data | Raw data stays local | Raw features stay local | Raw data stays local |
| Privacy risk | Gradient inversion on model updates | Feature inference from partial outputs | Activation inference (intermediate layers may leak) |
| Typical use case | Multiple hospitals, same schema | Hospital + pharmacy, different schemas | Limited-compute clients with a powerful server |
| Platform support | All 11 strategies | 4 strategies (`vfl` task) | 3 strategies (`split` task) |

Choose HFL (the default) unless your organizational structure specifically calls for VFL or split learning. Most government and healthcare collaborations involve organizations with the same type of data (same schema) but different populations, which maps directly to HFL.

### 1.6 A Concrete Example

Consider fraud detection across five banks:

1. Each bank has its own transaction records: amounts, timestamps, merchant categories, customer histories.
2. The FL server sends a small neural network (50,000 parameters -- about 200 KB of data) to each bank.
3. Each bank trains this network on its transactions for a few epochs (passes through the data). The network learns patterns like "transactions at 3 AM from a foreign country following a large purchase are often fraudulent."
4. Each bank sends back its updated model weights (~200 KB) to the server. No transaction records are transmitted.
5. The server averages the five sets of weights into one improved model.
6. This cycle repeats for 5 rounds.
7. The final model has learned fraud patterns from all five banks' data, even though no bank ever shared a single transaction with another bank or with the server.

In this platform, this exact scenario is the `fraud` task. It completes in about 68 seconds with 5 clients.

### 1.7 What Makes Federated Learning Different from Distributed Training

It is important to distinguish federated learning from distributed training, which is a related but different concept:

**Distributed training** splits a large dataset across multiple GPUs or machines to speed up training. The data is owned by a single organization and could theoretically be processed on a single machine (it would just take longer). There are no privacy constraints -- all machines have access to all data. Examples: training GPT-4 across thousands of GPUs, or scaling a model across a GPU cluster in a single data center.

**Federated learning** keeps data at separate organizations that do not share data with each other. The data cannot be centralized -- not because of technical limitations, but because of legal, regulatory, or trust constraints. Each organization only sees its own data and the aggregated model. The goal is not speed (FL is actually slower than centralized training) but privacy and compliance.

This platform implements federated learning. It is designed for scenarios where multiple organizations want to collaboratively train a model but cannot share their raw data.

### 1.8 Limitations of Federated Learning

No technology is a silver bullet. Understanding FL's limitations helps you decide whether it is appropriate for your use case:

- **Communication overhead.** Model updates must travel over the network each round. For large models (millions of parameters), this can be hundreds of megabytes per round per client. The adapter framework (Section 12) mitigates this for language models.
- **Synchronization delays.** In synchronous FL, the server waits for all clients before aggregating. The slowest client determines the round time. Asynchronous FL algorithms exist but introduce their own trade-offs.
- **Non-IID challenges.** When clients have very different data distributions (which is common in practice), the federated model may perform poorly for all clients. Advanced strategies (FedProx, SCAFFOLD) help, but cannot fully overcome extreme data heterogeneity.
- **Privacy is not absolute.** As discussed in Section 2, model updates leak some information. Differential privacy and secure aggregation reduce but do not eliminate this leakage.
- **Requires coordination.** All participating organizations must agree on the model architecture, training schedule, and data format. This organizational coordination can be more challenging than the technical deployment.

---

## 2. Privacy Guarantees and Their Limits

### 2.1 What Federated Learning Protects

The fundamental guarantee of federated learning is that **raw data never leaves the site where it was collected**. The server and other clients never see individual patient records, transaction details, or document contents. They only see model updates -- numerical arrays representing how the model's parameters changed during training.

This is a meaningful protection. An eavesdropper monitoring the network sees only floating-point numbers, not structured data. A compromised server has model weights, not patient records. Regulatory audits can confirm that no raw data left the organization's network.

### 2.2 What Federated Learning Does NOT Protect

Research has demonstrated that model updates are not perfectly opaque. They carry information about the training data, and sophisticated attackers can extract some of that information. Two attack categories are particularly important to understand:

**Membership Inference Attacks (MIA).** Given a trained model and a specific data record, an attacker can determine with some confidence whether that record was used to train the model. The attack works because models tend to "memorize" their training data slightly -- they produce lower prediction errors on data they were trained on compared to data they were not trained on. In our testing (see `privacy/test_privacy.py`), a trained BiLSTM model shows a measurable loss gap between training members and non-members, confirming that this attack is real.

**Gradient Inversion Attacks (also called Deep Leakage from Gradients, or DLG).** Given the model weights and a gradient update from a single training step, an attacker can attempt to reconstruct the original training input that produced that gradient. The attacker starts with random data and iteratively adjusts it until its gradient matches the observed gradient. In our testing, this attack achieves a cosine similarity of approximately 0.4-0.8 on unprotected gradients for small models -- meaning the attacker recovers a recognizable approximation of the original input.

These attacks prove that **federated learning alone is not sufficient for strong privacy**. Additional protections are needed, and this platform provides several.

### 2.3 Defense: Differential Privacy (DP)

Differential privacy (DP) adds carefully calibrated random noise to model updates before they leave a client (Local DP) or after the server aggregates them (Central DP). The noise is strong enough to mask individual contributions but weak enough that the aggregated model still learns useful patterns.

**The formal guarantee:** A mechanism satisfies (epsilon, delta)-differential privacy if the inclusion or exclusion of any single individual's data in the training set changes the probability of any particular output by at most a factor of exp(epsilon), with a failure probability of at most delta. In plain language: an attacker looking at the model cannot determine whether any specific person's data was used to train it.

**How it works in this platform:**

1. **Gradient clipping:** Before adding noise, each client's gradient (the model update) is clipped to a maximum norm. This bounds the maximum influence any single data point can have on the update.
2. **Noise injection:** Gaussian noise is added to the clipped gradient. The noise magnitude is calibrated to the clipping norm and the target epsilon.
3. **Privacy accounting:** The platform uses Renyi Differential Privacy (RDP) accounting to track the cumulative privacy cost across multiple rounds. Each round consumes some of the privacy budget; when the budget is exhausted, training should stop.

**Epsilon values explained:**

The strength of the privacy guarantee is measured by epsilon. Lower epsilon means more noise and stronger privacy, but also more degradation to model accuracy:

- **Epsilon = 1.0:** Very strong privacy. Used by Apple for emoji prediction. Significant accuracy degradation.
- **Epsilon = 10.0:** Meaningful privacy. The platform's stronger DP setting. Moderate accuracy impact.
- **Epsilon = 50.0:** Moderate privacy. The platform's default DP setting. Minimal accuracy impact.
- **Epsilon = infinity:** No privacy (no noise added). This is the non-DP baseline.

The platform supports epsilon values of 10.0 (stronger privacy, more noise) and 50.0 (moderate privacy, less noise). These values were chosen as a practical balance: epsilon=10 provides meaningful protection against membership inference while maintaining model utility; epsilon=50 provides moderate protection with negligible accuracy loss.

**Central DP vs. Local DP:**

- **Central DP** (`DP_Central_*` strategies): The server adds noise after aggregating all client updates. This requires trusting the server -- the server sees individual client updates in plaintext before adding noise. The advantage is better model accuracy, because noise is added once to the aggregate rather than separately to each client's update.
- **Local DP** (`DP_Local_*` strategies): Each client adds noise to its own update before sending it. The server never sees the unnoised update. This does not require trusting the server, but degrades accuracy more because each of the N clients adds independent noise.

In our gradient inversion testing, DP with sigma=0.5 and clip_norm=1.0 reduces the attacker's reconstruction quality significantly -- cosine similarity drops and the reconstructed data becomes unrecognizable.

### 2.4 Defense: Secure Aggregation (SecAgg)

Secure Aggregation uses cryptographic masking so that the server can compute the average of all clients' updates without seeing any individual client's update.

**How it works:**

1. Before training begins, each pair of clients (e.g., Client A and Client B) agrees on a random "mask" -- a random vector of the same size as the model update. This agreement uses a key exchange protocol (similar to the Diffie-Hellman key exchange used in HTTPS).
2. Client A adds the mask to its update; Client B subtracts the same mask from its update.
3. When the server sums all updates, the masks cancel out: `(update_A + mask) + (update_B - mask) = update_A + update_B`. The server gets the correct sum but never sees the individual updates.
4. With N clients, each client adds masks agreed with every other client and subtracts masks agreed with every other client. The combinatorics ensure that all masks cancel in the aggregate.

This protects against a **curious server** -- even if the server operator wants to inspect individual updates, they cannot. The server only ever sees the sum. SecAgg is critical in scenarios where the server is operated by a third party or a potentially adversarial organization.

**Limitations of SecAgg:**

- SecAgg requires all participating clients to complete the round. If a client drops out mid-round, its mask does not get subtracted, corrupting the aggregate. The platform handles this by requiring a minimum number of clients per round.
- SecAgg protects individual updates but not the aggregate. If a colluding server and N-1 clients work together, they can recover the remaining client's update by subtracting their own updates from the aggregate.
- SecAgg adds communication overhead (pairwise key exchange). For small numbers of clients (2-20), this overhead is negligible.

### 2.5 Defense: Homomorphic Encryption (for Inference)

For inference (using a trained model to make predictions on new data), the platform supports homomorphic encryption via TenSEAL CKKS. This allows a data owner to encrypt their input, send it to a model owner, receive an encrypted prediction, and decrypt it -- all without the model owner ever seeing the input data or the prediction. See Section 13 for full details.

### 2.6 Privacy Is a Spectrum

There is no single "private enough" threshold. The right combination of protections depends on your threat model:

| Threat | Protection | Platform Feature |
|--------|-----------|-----------------|
| Network eavesdropping | TLS encryption | Section 9 |
| Curious server inspecting updates | Secure Aggregation | `SecAgg_Alpha_0.5` strategy |
| Gradient inversion attacks | Differential Privacy | `DP_Central_*` and `DP_Local_*` strategies |
| Membership inference | Differential Privacy + regularization | DP strategies + model architecture |
| Malicious server modifying aggregation | Split learning / VFL | `split` and `vfl` tasks |
| Inference-time data exposure | Homomorphic encryption | Secure inference module (Section 13) |

For most government deployments, we recommend starting with **TLS + Secure Aggregation** and adding Differential Privacy if the data sensitivity warrants the accuracy trade-off. Run the privacy attack tests (`privacy/test_privacy.py`) on your trained models to measure actual leakage.

### 2.7 A Note on Regulatory Compliance

This platform provides technical privacy protections, but technical measures alone do not ensure regulatory compliance. Consult your organization's legal and compliance teams to determine:

- Whether federated learning satisfies data localization requirements in your jurisdiction. Some regulations require not just that data stays local, but that no information derived from the data (including model updates) crosses certain boundaries.
- Whether differential privacy with your chosen epsilon value provides a sufficient formal privacy guarantee for your regulatory context. Some regulations are satisfied by any epsilon; others may require specific thresholds.
- Whether the audit trail produced by this platform (CloudTrail logs, Docker audit logs, data manifests) satisfies your record-keeping requirements.
- Whether a Data Protection Impact Assessment (DPIA) is required before deploying FL on personal data.

---

## 3. Architecture

This section describes the system architecture: what components exist, how they communicate, and why the system is designed this way.

### 3.1 Overview

The platform follows a hub-and-spoke architecture. One central server (the "aggregator") coordinates the training process. Multiple clients (one per participating site) train on their local data and send updates to the server. All communication flows through encrypted gRPC channels.

The following diagram shows a typical deployment with five client sites:

```
                      +-------------------------------+
                      |     FL Server (aggregator)     |
                      |     start_server() :9092       |
                      |     TLS (EC P-256)             |
                      |     32 vCPU / 128 GB / L4      |
                      +---------------+----------------+
                                      | gRPC + TLS
              +-----------------------+-----------------------+
              |            |          |          |             |
        +-----+-----+ +---+------+ +-+-------+ ++--------+ +-+--------+
        | Client 0  | | Client 1 | | Client 2| | Client 3| | Client 4 |
        | L4 24GB   | | L4 24GB  | | L4 24GB | | L4 24GB | | L4 24GB  |
        | Site A    | | Site B   | | Site C  | | Site D  | | Site E   |
        +-----------+ +----------+ +---------+ +---------+ +----------+
```

**Why hub-and-spoke?** We chose this topology because it is simple to deploy, easy to secure (only one server address to protect), and well-supported by the Flower framework.

Alternatives considered:

- **Peer-to-peer (decentralized) FL:** Each client communicates directly with every other client. This removes the single point of failure (the server) and eliminates the need for a trusted aggregator. However, it dramatically increases network complexity (N^2 connections instead of N) and makes TLS certificate management much harder. It also requires each client to have a public endpoint, which is unacceptable for many government networks.
- **Hierarchical FL:** Regional aggregators collect updates from nearby clients, then forward aggregated updates to a central server. This reduces the central server's load and can improve latency for geographically distributed deployments. However, it adds infrastructure complexity and requires trust in the regional aggregators. This architecture becomes relevant when scaling beyond 20 clients.

For deployments with fewer than 20 clients, hub-and-spoke is the standard industry choice.

### 3.2 Components

The following table lists every major component in the system, its role, and the key files that implement it. Subsequent sections explain each component in detail.

| Component | Role | Key Files |
|-----------|------|-----------|
| **FL Server** | Aggregation, strategy execution, results collection | `runners/run_ec2.py --distributed` |
| **FL Client** | Local training, model updates, pre-flight data check | `runners/run_client.py` |
| **Docker Image** | Unified runtime for server + clients | `Dockerfile` (all deps pinned) |
| **TLS / mTLS** | Server + per-client certificates (EC P-256) | `deploy/gen_mtls_certs.sh` |
| **Orchestrator** | Automated deploy, run, collect | `run_server_side.sh` in `docker:cli` |
| **Data Pipeline** | Client-side ingestion, validation, manifests | `tools/ingest.py`, `fl_common/data.py` |
| **Adapter Framework** | Federated LoRA for any HuggingFace model | `fl_common/federated_adapter.py` |
| **Secure Inference** | CKKS homomorphic encryption (TenSEAL) | `secure_inference/tenseal_inference.py` |

**Design decision: unified Docker image.** Both the server and all clients run the same Docker image. We chose this approach (instead of separate server and client images) because it simplifies versioning -- there is only one image to build, distribute, and track. The entry point determines the role: `runners/run_ec2.py` for the server, `runners/run_client.py` for clients.

### 3.3 Communication Flow

This is the step-by-step sequence of events during a complete training run. Understanding this flow is essential for debugging issues where clients fail to connect or training hangs between strategies.

1. The orchestrator stops any existing containers on all nodes, then starts client containers on each client node. Clients enter a reconnect loop, waiting for the server to become available.
2. The orchestrator starts the server container, which calls `start_server()` and binds to port 9092 with TLS encryption.
3. Clients connect to the server using the server's private IP address and authenticate using the CA certificate.
4. The server sends the strategy configuration to each client, including the strategy name (e.g., FedAvg), learning rate, number of local epochs, and how to partition the data.
5. Each client trains the model on its local data partition and sends the updated model weights back to the server.
6. The server aggregates all client updates, evaluates the aggregated model, and advances to the next round.
7. After all rounds for a given strategy complete, the server moves to the next strategy in the queue. Clients detect the disconnection, wait briefly, and reconnect for the next strategy.
8. After all strategies have been executed, the server exits. The orchestrator then moves to the next task (e.g., from fraud detection to sepsis prediction).

### 3.4 The Flower Framework

This platform is built on **Flower** (flwr), an open-source federated learning framework maintained by Flower Labs. Flower provides the gRPC communication layer, the client-server protocol, and the aggregation infrastructure. The platform extends Flower with custom strategies (FedProx, SCAFFOLD, DP variants), custom data pipelines, and the adapter framework.

Understanding Flower is not necessary to deploy the platform -- the deployment scripts and orchestrator handle all Flower-specific details. However, if you need to debug communication issues or extend the platform, familiarity with Flower's concepts (clients, servers, strategies, and rounds) will be helpful. The Flower documentation is at https://flower.ai/docs.

### 3.5 Important: No Separate SuperLink

This is a common source of confusion and deployment errors. The Flower framework includes a component called "SuperLink" that can act as a standalone gRPC server for coordinating FL training. However, in this platform, `runners/run_ec2.py --distributed` calls `start_server()` directly, which binds port 9092 and acts as the gRPC server.

**Do not run a separate SuperLink container.** If a SuperLink is already running, it will hold port 9092, and the training server will fail with: `Port in server address 0.0.0.0:9092 is already in use`.

The SuperLink is only needed for idle monitoring between runs and must be stopped before any training starts. The orchestrator handles this automatically by killing the SuperLink container before starting each task.

### 3.6 Network Configuration

Only two network ports are used by the platform. This minimal footprint simplifies security group configuration and audit.

| Port | Protocol | Direction | Purpose |
|------|----------|-----------|---------|
| 9092 | gRPC + TLS | Client -> Server | FL model updates (the core FL protocol) |
| 22 | SSH | Orchestrator -> All | Deployment, container management, log collection |

**Why gRPC?** gRPC is a high-performance RPC framework built on HTTP/2. It supports bidirectional streaming, which is necessary for FL where the server needs to push model weights to clients and clients need to push updates back. It also has native TLS support and is the default transport for the Flower FL framework. The alternative -- REST over HTTPS -- would add serialization overhead and lack streaming support. For model updates that can be tens of megabytes, gRPC's binary serialization (Protocol Buffers) is significantly more efficient.

---

## 4. Prerequisites and Dependencies

Before deploying the platform, ensure that all software versions and AWS resources are in place. Version mismatches -- especially between GPU drivers, CUDA, and PyTorch -- are one of the most common causes of deployment failures.

### 4.1 Software Versions

The following table lists every software component, the minimum version that is known to work, and the exact version we have validated in production. We strongly recommend using the validated versions to avoid compatibility issues.

| Component | Minimum Version | Validated Version |
|-----------|----------------|-------------------|
| Docker | 24.0+ | 25.0.14 |
| NVIDIA Driver | 550+ | 570.124.06 |
| CUDA (container) | 12.4+ | 12.4 (PyTorch cu124) |
| PyTorch | 2.5+ | 2.5.1+cu124 |
| Flower (flwr) | 1.30+ | 1.30.0 |
| Python | 3.12+ | 3.12.13 |
| NumPy | 1.26+ | 1.26.4 |
| Pandas | 3.0+ | 3.0.3 |
| Scikit-learn | 1.8+ | 1.8.0 |
| TenSEAL | 0.3.16 | 0.3.16 |
| OS | Amazon Linux 2023 | Amazon Linux 2023 |
| nvidia-container-toolkit | 1.19+ | 1.19.0 |

**Why pin exact versions?** In machine learning systems, even minor version differences can produce different numerical results, causing model divergence across clients. For example, a PyTorch patch release might change the default random number generator seed, causing different weight initializations on different nodes. NumPy version differences can affect floating-point accumulation order. All dependencies are pinned to exact versions in the Dockerfile to ensure that every node in the cluster runs identical software.

**Version upgrade policy:** Before upgrading any dependency, run the full test suite on a staging cluster to verify that results are numerically consistent with the previous version. This is especially important for PyTorch, NumPy, and Flower, which directly affect model computation and FL communication.

### 4.2 AWS Prerequisites

The platform runs on AWS EC2 instances within a Virtual Private Cloud (VPC). The following AWS resources must be provisioned before deployment:

- **VPC with private subnets** for inter-node communication. All FL traffic (port 9092) should flow over private subnets, never over the public internet. This provides network-level isolation in addition to TLS encryption.
- **IAM role with EC2 describe/start/stop permissions** for cost management scripts that automatically stop instances when training completes.
- **S3 bucket for result archival** (optional but recommended). Training results, data manifests, and audit logs should be stored durably in S3 with server-side encryption.
- **CloudWatch agent installed on all nodes** (recommended) for centralized monitoring and alerting.
- **KMS key for secret encryption** (see Section 15). TLS private keys and SSH keys should be encrypted at rest using a customer-managed KMS key.
- **SSH key pair** created in the EC2 console (`.pem` format). This key is used for deployment, orchestration, and log collection.

### 4.3 Operator Prerequisites

The person performing the deployment needs:

- **AWS CLI** configured with credentials that have permissions to manage EC2 instances, ECR repositories, S3 buckets, and Secrets Manager.
- **Terraform >= 1.5** (if using infrastructure-as-code, which is recommended for reproducible deployments).
- **OpenSSL >= 3.0** for generating TLS certificates. Older versions of OpenSSL may not support the EC P-256 curve parameters used by the platform.
- **Access to a private Docker registry or ECR repository** for distributing the Docker image to cluster nodes.

---

## 5. Configuration Management

All deployment scripts read their configuration from a single file called `cluster.env`. This section explains how to create this file, what each setting controls, and how to manage configurations for different environments (staging, production).

### 5.1 Cluster Configuration File

Create `cluster.env` on the operator workstation. **This file contains sensitive information (IP addresses, key paths, infrastructure identifiers) and must not be committed to version control.** Add `cluster.env` to your `.gitignore`.

The following template shows every configuration parameter with explanatory comments:

```bash
# cluster.env -- Cluster configuration
# Copy to cluster.env.local and fill in values

# --- Infrastructure ---
AWS_REGION=ap-southeast-1
VPC_ID=vpc-XXXXXXXXXXXXXXXXX
SUBNET_ID=subnet-XXXXXXXXXXXXXXXXX
KEY_PAIR_NAME=<your-key-pair>
KEY_PATH=~/.ssh/<your-key>.pem
SECURITY_GROUP_ID=sg-XXXXXXXXXXXXXXXXX

# --- Cluster Nodes ---
SERVER_IP=<server-private-ip>
SERVER_PUBLIC_IP=<server-public-ip>
CLIENT_IPS="<client1-ip> <client2-ip> <client3-ip> <client4-ip> <client5-ip>"
NUM_CLIENTS=5

# --- Image ---
FL_IMAGE=healthcare-fl
FL_IMAGE_TAG=latest
REGISTRY=<account-id>.dkr.ecr.<region>.amazonaws.com

# --- TLS ---
CERTS_DIR=~/fl-deploy/certs
CA_CERT=${CERTS_DIR}/ca.pem
SERVER_CERT=${CERTS_DIR}/server.pem
SERVER_KEY=${CERTS_DIR}/server.key

# --- Data ---
DATA_DIR=~/fl-deploy/data
RESULTS_DIR=~/fl-deploy/results

# --- Timeouts (seconds) ---
TIMEOUT_DEFAULT=3600
TIMEOUT_DENSENET=54000
TIMEOUT_MEDIUM=7200
```

Key parameters explained:

- `SERVER_IP` is the **private** IP of the server. Clients use this to connect over the VPC internal network. Never use the public IP for FL traffic.
- `SERVER_PUBLIC_IP` is used only for SSH access from the operator workstation.
- `CLIENT_IPS` is a space-separated list of client private IPs. The order matters: client 0 gets `PARTITION_ID=0`, client 1 gets `PARTITION_ID=1`, and so on.
- `CERTS_DIR` is where TLS certificates are stored on each node. The deployment scripts copy certificates to this directory.
- Timeout values control how long the orchestrator waits before killing a hung training run. DenseNet tasks require much longer timeouts because they process image data through a large model.

### 5.2 Loading Configuration

All deployment scripts source this file at the start, then validate that required variables are set. If any required variable is missing, the script exits with an error rather than proceeding with potentially broken configuration.

```bash
source cluster.env

# Validate required variables
for var in SERVER_IP CLIENT_IPS KEY_PATH CERTS_DIR; do
  [ -z "${!var}" ] && echo "ERROR: $var not set in cluster.env" && exit 1
done
```

### 5.3 Per-Environment Overrides

For organizations that maintain separate staging and production environments (which is strongly recommended for regulated deployments), create separate configuration files:

```
cluster.env.staging
cluster.env.production
```

Symlink the active environment to `cluster.env`:

```bash
ln -sf cluster.env.production cluster.env
```

This approach ensures that you never accidentally run production commands against a staging cluster or vice versa. Before any deployment, verify which environment is active by checking the symlink target.

---

## 6. Infrastructure Setup

This section walks through provisioning EC2 instances, installing GPU drivers, and setting up Docker with GPU support. These steps only need to be performed once per instance; they are not repeated between training runs.

### 6.1 EC2 Instance Sizing

Choosing the right instance type involves balancing GPU memory (needed for model training), CPU and RAM (needed for data preprocessing), and cost. The following configurations have been validated.

#### Recommended Configuration

This configuration provides a GPU on every node, which is required for image-based tasks (chest X-ray, satellite) and significantly accelerates all other tasks.

| Role | Instance Type | Count | GPU | vCPU | RAM | Storage |
|------|--------------|-------|-----|------|-----|---------|
| Server | g6.8xlarge | 1 | L4 24GB | 32 | 128 GB | 500 GB gp3 |
| Client | g6.4xlarge | N | L4 24GB | 16 | 64 GB | 1 TB gp3 |

**Why g6 instances?** The g6 family uses NVIDIA L4 GPUs, which provide 24 GB of VRAM at a lower cost than A10G (g5) or A100 (p4d) instances. The L4 is sufficient for all models in this platform, including DenseNet-121 (which peaks at ~12 GB). For larger language models (e.g., Llama-3-8B with QLoRA), you would need g5 or p4d instances.

**Why different server and client sizes?** The server performs aggregation (averaging model weights from all clients), which is CPU-intensive but not GPU-intensive. The extra CPU and RAM on g6.8xlarge (32 vCPU, 128 GB) handles aggregation for up to 20 clients. Clients are g6.4xlarge (16 vCPU, 64 GB) because they only need to train a single model on their local data.

#### Budget Configuration (CPU-only clients)

For tasks that use small models (MLP, logistic regression), clients do not strictly need GPUs. This configuration significantly reduces cost.

| Role | Instance Type | Count | GPU |
|------|--------------|-------|-----|
| Server | g6.4xlarge | 1 | L4 24GB |
| Client | t3.xlarge | N | None |

**When to choose this:** Only for tabular data tasks (fraud, sepsis, anomaly, mortality, readmission, drug). Image tasks and large models require GPU clients. Note that even for tabular tasks, GPU clients train faster (seconds vs. minutes), so the budget configuration extends training time.

### 6.2 Provisioning

You can provision instances using Terraform (recommended for reproducibility) or the AWS CLI. Both approaches are shown below. Always enable EBS encryption -- for regulated workloads, use a customer-managed KMS key (CMK) rather than the default AWS key.

```bash
source cluster.env

# Using Terraform (recommended)
cd deploy/terraform
terraform init && terraform apply -var-file=production.tfvars

# Or AWS CLI
aws ec2 run-instances \
  --image-id <ami-id> \
  --instance-type g6.4xlarge \
  --count ${NUM_CLIENTS} \
  --key-name ${KEY_PAIR_NAME} \
  --security-group-ids ${SECURITY_GROUP_ID} \
  --subnet-id ${SUBNET_ID} \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":1000,"VolumeType":"gp3","Encrypted":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fl-client},{Key=Environment,Value=production}]'
```

**Note:** Always enable EBS encryption. Use a KMS CMK for regulated workloads. Unencrypted EBS volumes mean that anyone with physical access to the underlying hardware could potentially read your data -- a risk that is unacceptable for healthcare and government workloads.

### 6.3 GPU Driver Installation

Every instance that has a GPU needs the NVIDIA driver installed before it can use the GPU. This must be done on the host operating system, not inside Docker containers. The driver provides the kernel module that allows user-space programs to communicate with the GPU hardware.

Run the following on each GPU instance:

```bash
ssh -i ${KEY_PATH} ec2-user@<NODE_IP>

# Add NVIDIA CUDA repo
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo

# Install driver
sudo dnf install -y nvidia-driver nvidia-driver-cuda nvidia-driver-libs

# Load kernel module
sudo modprobe nvidia

# Verify
nvidia-smi
# Expected: NVIDIA L4, Driver 595.x, CUDA 13.x
```

If `nvidia-smi` shows the GPU name, driver version, and CUDA version, the installation was successful. If it fails, check `dmesg | tail -20` for kernel errors. The most common cause is a kernel update that invalidated the driver module -- reinstalling the driver resolves this.

### 6.4 Docker Setup

Docker provides the containerized runtime environment. The NVIDIA Container Toolkit is an additional component that allows Docker containers to access the host GPU.

```bash
# Install Docker
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

The verification command runs a minimal CUDA container and checks that `nvidia-smi` works inside it. If this fails but `nvidia-smi` works on the host, the Container Toolkit is not configured correctly. The most common fix is restarting Docker after configuring the toolkit.

---

## 7. Docker Image

The Docker image is the packaged runtime environment that runs on every node in the cluster. It contains Python, PyTorch, Flower, all model implementations, all task pipelines, and the server/client entry points. This section covers building, distributing, and securing the image.

### 7.1 Build

Build the image from the repository root. The Dockerfile pins every dependency to an exact version to ensure all nodes run identical software.

```bash
cd /path/to/healthcare-fl
docker build -t ${FL_IMAGE}:${FL_IMAGE_TAG} -f Dockerfile .

# Image includes:
#   - Python 3.12, PyTorch 2.5.1+cu124, Flower 1.29
#   - All models: bilstm, mlp, densenet, autoencoder, logreg, cnn1d,
#     tabnet, resnet_small, vfl_mlp, split_bilstm, generic, mistral
#   - All tasks + generic config-driven pipeline
#   - FL strategies, privacy mechanisms, secure inference, scenarios
#   - runners/run_ec2.py (server), runners/run_client.py (client)
#
# Orchestrator uses docker:cli (Alpine + Docker CLI + SSH)
```

The build takes approximately 10-15 minutes and produces an image of about 3.2 GB compressed. Most of this size comes from PyTorch with CUDA support.

### 7.2 Push to Registry (Recommended)

Amazon Elastic Container Registry (ECR) is the recommended distribution method. It integrates with IAM for access control and stores images encrypted at rest.

```bash
# Tag and push to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${REGISTRY}

docker tag ${FL_IMAGE}:${FL_IMAGE_TAG} ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG}
docker push ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG}

# Pull on each node
for ip in ${SERVER_IP} ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "aws ecr get-login-password --region ${AWS_REGION} | \
     docker login --username AWS --password-stdin ${REGISTRY} && \
     docker pull ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG} && \
     docker tag ${REGISTRY}/${FL_IMAGE}:${FL_IMAGE_TAG} ${FL_IMAGE}:${FL_IMAGE_TAG}"
done
```

The `for` loop SSHs into each node, authenticates with ECR, pulls the image, and tags it with the local name that deployment scripts expect. This ensures every node has an identical image.

### 7.3 Distribute via SCP (Alternative)

When ECR is unavailable (e.g., air-gapped environments or networks without internet access), you can distribute the image as a compressed tarball. **Always distribute from the server, not from a local machine.** VPC internal network transfers at gigabit speeds (~30 seconds per client), whereas transferring from an external machine would be much slower and would route through the public internet.

```bash
# On server: save image
docker save ${FL_IMAGE}:${FL_IMAGE_TAG} | gzip > /tmp/fl-image.tar.gz
# Size: ~3.2 GB compressed

# Distribute to all clients in parallel
for ip in ${CLIENT_IPS}; do
  (
    scp -i ${KEY_PATH} -o StrictHostKeyChecking=no \
      /tmp/fl-image.tar.gz ec2-user@${ip}:/tmp/
    ssh -i ${KEY_PATH} -o StrictHostKeyChecking=no ec2-user@${ip} \
      "sudo docker load < /tmp/fl-image.tar.gz && rm /tmp/fl-image.tar.gz"
    echo "$ip: DONE"
  ) &
done
wait
```

The `&` at the end of each iteration runs the transfer in parallel, and `wait` blocks until all transfers complete. This reduces total distribution time from 5x (serial) to 1x (parallel) the single-transfer time.

### 7.4 Container Security

In a federated learning deployment, containers run code that processes sensitive data and communicates over the network. A compromised container could leak data, tamper with model updates, or be used as a pivot point for attacking other nodes. The following Docker flags harden containers against these threats.

```bash
# Production container flags
--read-only \                           # Read-only root filesystem
--tmpfs /tmp:rw,noexec,nosuid,size=2g \ # Writable tmp with limits
--security-opt no-new-privileges \      # Prevent privilege escalation
--cap-drop ALL \                        # Drop all capabilities
--cap-add SYS_NICE \                    # Only add what's needed (GPU scheduling)
--memory 60g \                          # Memory limit (adjust per instance)
--memory-swap 60g \                     # No swap
--cpus 14 \                             # CPU limit
--pids-limit 512 \                      # Prevent fork bombs
--log-opt max-size=100m \               # Log rotation
--log-opt max-file=5 \                  # Keep 5 rotated log files
```

What each flag protects against:

- `--read-only` prevents an attacker from writing persistent malware to the container filesystem. Legitimate writes go to the tmpfs mount.
- `--no-new-privileges` prevents a process inside the container from gaining additional Linux capabilities (e.g., via setuid binaries).
- `--cap-drop ALL` removes all Linux kernel capabilities except `SYS_NICE`, which is needed for GPU scheduling. This limits what a compromised process can do at the kernel level.
- `--memory` and `--memory-swap` prevent a runaway process from exhausting host memory and crashing other containers or the host itself.
- `--pids-limit` prevents fork bomb attacks.
- `--log-opt` prevents logs from filling the disk (a common denial-of-service vector).

**Note:** `--user` (non-root) is recommended but requires the Dockerfile to support it. If the base image runs as root, add a `USER` directive to the Dockerfile.

---

## 8. Data Pipeline

### 8.1 Why Data Stays at Each Site

The fundamental principle of federated learning is that data stays where it was collected. This is not just a technical design choice -- it is often a legal requirement.

In healthcare, regulations such as HIPAA (United States), GDPR (European Union), and PIPL (China) place strict controls on where patient data can be stored and processed. Moving patient records to a central server would require data processing agreements, potentially cross-border data transfer mechanisms, and expose the centralized data to a single point of breach.

In government, data sovereignty laws often require that citizen data remains within national borders. Intelligence and defense data may have classification restrictions that prohibit any form of centralization.

By keeping data at each site, federated learning eliminates the legal complexity of data sharing agreements. Each organization maintains full control over its data, can audit access independently, and is responsible only for its own data security.

### 8.2 Data Ingestion

Each participating organization ingests its own data locally using the `tools/ingest.py` command-line tool. The server never sees raw data -- only metadata manifests that describe the shape and characteristics of the data without revealing its contents.

Run the following on each client node:

```bash
# Ingest local data
python tools/ingest.py --task sepsis --input /mnt/ehr/sepsis_cohort.csv --client-id site_a

# Validate existing ingested data
python tools/ingest.py --task sepsis --validate-only

# View manifest
python tools/ingest.py --show-manifest ~/fl-deploy/data/sepsis
```

The `--client-id` parameter is a human-readable identifier for the site (e.g., "hospital_a", "bank_singapore"). It is recorded in the manifest for audit purposes but does not affect training.

### 8.3 What tools/ingest.py Does

The ingestion pipeline performs five steps, each designed to catch data problems before they cause training failures:

1. **Reads input data** (CSV, NPZ, or image directory). The tool auto-detects the format based on file extension.
2. **Validates** -- checks shape (correct number of features), data types (numeric where expected), NaN/Inf values, and label distribution (are there at least some examples of each class?). If validation fails, it blocks ingestion and reports the errors.
3. **Converts to standardized format** (`data.npz` with `X` for features and `y` for labels). This ensures that all clients produce data in the same format regardless of their input source.
4. **Generates `manifest.json`** -- a metadata file containing the schema, sample count, SHA-256 checksum of the data file, and label distribution. This manifest is used for pre-flight validation before training and can optionally be shared with the server for coordination.
5. **Outputs to `~/fl-deploy/data/<task>/`** in the standard directory layout.

### 8.4 Data Directory Layout

Every task follows the same directory structure. This consistency allows the training code to find data without per-task configuration.

```
~/fl-deploy/data/<task>/
  manifest.json       -- DataManifest (schema, counts, checksums)
  data.npz            -- features (X) and labels (y)
  OR data.csv         -- tabular data with header row
  OR images/          -- image directory (chest_xray, satellite)
      metadata.csv    -- image paths and labels
```

For tabular data (most tasks), the data is stored as a compressed NumPy archive (`.npz`). For image data (chest X-ray, satellite imagery), the images are stored in a directory with a metadata CSV that maps file paths to labels.

### 8.5 Data Validation Gates

Data validation happens at two points: during ingestion (when you run `tools/ingest.py`) and at training startup (when `runners/run_client.py` begins). This defense-in-depth approach catches problems even if data files are modified between ingestion and training.

Before any training run, you can manually verify data integrity:

```bash
# Verify manifest checksums match data files
python tools/ingest.py --task <TASK> --validate-only

# Pre-flight check in runners/run_client.py logs:
# Data: npz, 12000 samples, checksum a1b2c3d4e5f6
# Starting FL client: task=sepsis ... data=real
```

`runners/run_client.py` performs these checks at startup:

- Manifest exists and is valid JSON
- SHA-256 checksum of data file matches the checksum recorded in the manifest (detects data corruption or tampering)
- Feature dimensions match the task configuration (e.g., sepsis expects 14 features)
- Label values are within expected range (e.g., binary classification labels must be 0 or 1)
- If any check fails, the client logs the error and falls back to synthetic data (randomly generated data that matches the expected format). This fallback allows testing and development without real data, but production deployments should investigate and fix validation failures rather than relying on synthetic data.

### 8.6 Data Versioning

Every data release should be tagged with a version in the manifest to support reproducibility. If you need to reproduce a training run six months later, you need to know exactly which data was used.

```json
{
  "version": "2026-05-01",
  "task": "sepsis",
  "samples": 12000,
  "sha256": "a1b2c3...",
  "schema": {"features": 34, "labels": 2},
  "created": "2026-05-01T00:00:00Z"
}
```

Keep previous data versions in `~/fl-deploy/data/<task>/archive/` for reproducibility. When investigating model behavior or responding to audits, you can re-run training with the exact same data version.

### 8.7 Data Retention Policy

The following retention periods are recommended. Adjust based on your organization's data governance policies and regulatory requirements.

| Data Type | Retention | Storage |
|-----------|-----------|---------|
| Raw ingested data | Per organization policy | Local EBS |
| Processed NPZ/CSV | Duration of engagement + 90 days | Local EBS |
| Manifests | Indefinite | S3 archive |
| Synthetic data | Ephemeral (regenerated per run) | tmpfs |
| Training results | 1 year minimum | S3 archive |

Manifests are retained indefinitely because they are small (a few KB each) and essential for auditing and reproducibility. Processed data has a limited retention period because it can be regenerated from raw data if needed.

### 8.8 Understanding Data Partitioning

In a real federated learning deployment, each site naturally has its own data -- Hospital A has its patients, Hospital B has its patients. But during development and testing, you may be working with a single dataset that needs to be split across simulated clients. The platform handles this automatically using partition IDs.

**IID (Independent and Identically Distributed) partitioning:** The dataset is shuffled randomly and split into N equal parts. Each client gets a representative sample of the full data distribution. This simulates the ideal case where all sites have similar data.

**Non-IID partitioning (Dirichlet distribution):** The dataset is split using a Dirichlet distribution parameterized by alpha. Lower alpha values create more extreme imbalance:

- **Alpha = 1.0:** Moderate heterogeneity. Each client gets all classes but in different proportions.
- **Alpha = 0.5:** Significant heterogeneity. Some clients may have very few examples of certain classes.
- **Alpha = 0.1:** Extreme heterogeneity. Some clients may have almost entirely one class. This simulates a specialist hospital that sees mostly one type of patient.

The partitioning is deterministic (given the same random seed, number of clients, and alpha value, the same partition is produced every time), which is important for reproducibility. The `PARTITION_ID` environment variable tells each client which partition to use.

**Why non-IID matters:** Real-world federated learning is almost always non-IID. Different hospitals serve different communities with different disease prevalence. Different banks operate in different markets with different fraud patterns. If your model only works well on IID data, it will underperform in production. Testing with non-IID strategies (FedProx, SCAFFOLD) is essential.

---

## 9. TLS and PKI

### 9.1 Why Federated Learning Needs TLS

In federated learning, model updates travel over the network between clients and the server. These updates are numerical arrays -- the weights of a neural network -- but as discussed in Section 2, they carry information about the training data. Research has shown that an attacker who intercepts model updates can perform gradient inversion attacks to partially reconstruct the original training data.

TLS (Transport Layer Security) encrypts all network traffic between clients and the server, preventing eavesdroppers from inspecting model updates in transit. Without TLS, anyone with access to the network path between a client and the server (e.g., a compromised router, a cloud provider's network equipment, or a co-tenant on the same physical host) could capture and analyze model updates.

The platform uses Elliptic Curve Cryptography (EC P-256) for TLS certificates. We chose EC P-256 over RSA because it provides equivalent security (128-bit) with much shorter keys (256 bits vs. 3072 bits for RSA), resulting in faster handshakes and smaller certificate sizes. EC P-256 is widely supported, recommended by NIST, and is the standard for modern TLS deployments.

### 9.2 Certificate Authority

A Certificate Authority (CA) is the root of trust for the TLS infrastructure. The CA's certificate is installed on all nodes, and each node's individual certificate is signed by the CA. When a client connects to the server, it verifies the server's certificate against the CA -- confirming that it is connecting to the legitimate server and not an impersonator.

**For production deployments, use an organizational CA or AWS Private CA (ACM PCA).** A managed CA provides automatic renewal, audit logging, and integration with your organization's PKI infrastructure. The self-signed CA procedure below is for development environments or situations where a managed PKI is unavailable.

#### Self-Signed CA (when managed PKI is unavailable)

```bash
cd deploy/distributed
mkdir -p certs

# Generate CA key and certificate
openssl ecparam -genkey -name prime256v1 -out certs/ca.key
openssl req -new -x509 -key certs/ca.key -out certs/ca.pem \
  -days 365 -subj "/CN=FL-Platform-CA/O=<YOUR_ORG>"
```

The first command generates an EC P-256 private key for the CA. The second command creates a self-signed X.509 certificate valid for 365 days. The `-subj` parameter sets the certificate's Common Name and Organization fields -- replace `<YOUR_ORG>` with your organization's name.

#### AWS Private CA (recommended)

```bash
# Create a subordinate CA in ACM PCA for FL platform use
aws acm-pca create-certificate-authority \
  --certificate-authority-type SUBORDINATE \
  --certificate-authority-configuration \
    "KeyAlgorithm=EC_prime256v1,SigningAlgorithm=SHA256WITHECDSA,Subject={CommonName=FL-Platform-CA,Organization=<YOUR_ORG>}"
```

**Why a subordinate CA?** Using a subordinate (rather than root) CA limits the blast radius if the CA is compromised. The subordinate CA can only issue certificates for the FL platform, not for your entire organization.

### 9.3 Server Certificate

The server certificate identifies the FL server to connecting clients. It must include Subject Alternative Names (SANs) that match the server's hostname and IP addresses. Without correct SANs, clients will reject the connection with a TLS handshake error.

```bash
# Generate server cert with SAN
cat > certs/san.cnf << EOF
[req]
distinguished_name=dn
req_extensions=v3
prompt=no
[dn]
CN=fl-server
[v3]
subjectAltName=DNS:fl-server,DNS:localhost,IP:127.0.0.1,IP:${SERVER_IP}
EOF

openssl ecparam -genkey -name prime256v1 -out certs/server.key
openssl req -new -key certs/server.key -out certs/s.csr -config certs/san.cnf
openssl x509 -req -in certs/s.csr -CA certs/ca.pem -CAkey certs/ca.key \
  -CAcreateserial -out certs/server.pem -days 365 \
  -extfile certs/san.cnf -extensions v3

# Clean up signing artifacts
rm -f certs/{s.csr,san.cnf,ca.srl}

# Secure the CA key -- store it offline or in a secrets manager
# Do NOT leave ca.key on any cluster node
chmod 400 certs/ca.key certs/server.key

# Distribution:
#   certs/ca.pem      -> ALL nodes (public, read-only)
#   certs/server.pem  -> server only (read-only)
#   certs/server.key  -> server only (secret, 0400 permissions)
```

**Critical:** The `IP:${SERVER_IP}` in the SAN must match the server's private IP address -- the IP that clients use to connect. If the server is replaced and gets a new private IP, you must regenerate the server certificate with the new IP. This is the most common cause of TLS handshake failures.

### 9.4 mTLS (Per-Client Certificates)

Standard TLS (server certificates only) ensures that clients can verify the server's identity. Mutual TLS (mTLS) adds client certificates so the server can also verify each client's identity. This is important in production because it prevents unauthorized nodes from joining the federated learning cluster.

```bash
# Generate CA + server cert + per-client certs in one step
./deploy/gen_mtls_certs.sh --full

# Output:
#   certs/ca.pem              -> all nodes
#   certs/server.pem/key      -> server only
#   certs/client_0.pem/key    -> client 0 only
#   certs/client_1.pem/key    -> client 1 only
#   ...

# Add a new client later
./deploy/gen_mtls_certs.sh --add-client 5

# Verify all certs
./deploy/gen_mtls_certs.sh --verify
```

Client certificates are auto-detected by `runners/run_client.py` -- if `client_N.pem` and `client_N.key` exist in the certs directory, mTLS is enabled automatically. No additional configuration is needed.

**Known limitation:** Flower's deprecated `start_server()` API does not enforce client certificate verification on the server side. The client certs are generated and transmitted but server-side validation requires migrating to Flower's `ServerApp` API. This means mTLS currently provides client-side server verification but not server-side client verification. For environments where unauthorized client access is a serious concern, use network-level controls (security groups restricting port 9092) as a compensating measure.

### 9.5 Certificate Rotation

TLS certificates have expiration dates (365 days in our configuration). Expired certificates cause immediate connection failures. Certificate rotation should be planned and automated, not performed as emergency maintenance when certificates expire.

```bash
# Check certificate expiry
./deploy/rotate_certs.sh --check

# Full rotation: generate new certs, distribute, restart, verify TLS
./deploy/rotate_certs.sh --full

# Generate only (no distribution)
./deploy/rotate_certs.sh --generate
```

Rotation backs up old certs before overwriting, so you can roll back if the new certificates cause issues.

Automate expiry monitoring by adding a weekly check to your operator workstation's crontab:

```bash
# Add to cron on operator workstation (check weekly)
0 9 * * 1 /path/to/deploy/rotate_certs.sh --check | grep -q "EXPIRING\|EXPIRED" && \
  echo "FL cert rotation needed" | mail -s "CERT EXPIRY WARNING" ops@example.com
```

This runs every Monday at 9 AM and sends an email alert if any certificate is expiring within 30 days or has already expired.

---

## 10. Deployment

This section covers the two deployment methods: automated (recommended) and manual. The automated method uses an orchestrator container that manages the entire training lifecycle, including starting/stopping containers, running multiple tasks in sequence, and handling timeouts.

### 10.1 Server-Side Orchestrator (Recommended)

The orchestrator runs as a Docker container on the server node. It manages the entire training pipeline: stopping old containers, starting clients, starting the server, monitoring progress, and moving to the next task. Running the orchestrator on the server (rather than on your workstation) ensures that training continues even if your laptop goes to sleep or your SSH session disconnects.

**Why a Docker container for the orchestrator?** The orchestrator needs the Docker CLI (to manage containers on the server) and SSH (to manage containers on clients). Rather than installing these on the host, we run a lightweight Alpine container (`docker:cli`) with SSH added. This keeps the host clean and makes the orchestrator reproducible.

```bash
source cluster.env

# Upload orchestrator script and SSH key to server (one-time setup)
scp -i ${KEY_PATH} run_server_side.sh ec2-user@${SERVER_PUBLIC_IP}:~/
scp -i ${KEY_PATH} ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP}:~/.ssh/$(basename ${KEY_PATH})
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "chmod 600 ~/.ssh/$(basename ${KEY_PATH})"

# Launch orchestrator container
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "
docker run -d --name fl-orchestrator \
  --restart on-failure:3 \
  --network host \
  --memory 4g \
  --cpus 2 \
  --pids-limit 256 \
  --log-opt max-size=200m \
  --log-opt max-file=5 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/.ssh/$(basename ${KEY_PATH}):/keys/deploy.pem:ro \
  -v ~/fl-deploy/certs:${CERTS_DIR}:ro \
  -v ~/fl-deploy/results:${RESULTS_DIR} \
  -v ~/run_server_side.sh:/run.sh:ro \
  docker:cli \
  sh -c 'apk add --no-cache openssh-client bash >/dev/null 2>&1 && bash /run.sh all'
"

# Monitor progress (can disconnect and reconnect anytime)
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-orchestrator --tail 20"
```

The orchestrator mounts the Docker socket (`/var/run/docker.sock`) to manage containers on the server, and uses SSH (via the mounted key) to manage containers on clients. The `--restart on-failure:3` flag automatically restarts the orchestrator if it crashes, up to 3 times.

**Available targets:**

- `all` -- run all tasks in sequence (~6-8 hours for the full suite)
- `failed` -- re-run only tasks that failed in the previous run
- `fraud` / `sepsis` / `ecg` / etc. -- run a single task (useful for testing or re-running a specific task)

**What the orchestrator does per task:**

1. Kills any running `fl-training`, `fl-superlink`, and client containers across the cluster
2. Starts `fl-client` containers on all client nodes, each with a reconnect loop
3. Starts `fl-training` on the server (`runners/run_ec2.py --distributed <task>`)
4. Monitors the server container every 30 seconds, checking if it is still running
5. Enforces a per-task timeout. If the server container runs longer than the timeout, it is killed and the task is marked as failed.
6. On completion or timeout, stops all client containers and prints a summary
7. Moves to the next task

**Per-task timeouts:**

Different tasks require vastly different amounts of time. Image-based tasks with large models (DenseNet-121 at 8 million parameters processing chest X-ray images) can take hours, while small tabular tasks (MLP at 50,000 parameters processing fraud data) complete in about a minute. The orchestrator uses these timeouts to prevent hung tasks from blocking the entire pipeline.

| Task | Timeout | Reason |
|------|---------|--------|
| chest, transfer | 15 hours | DenseNet-121 (8M params, image data) |
| ecg, satellite | 2 hours | Medium models with 11 strategies |
| All others | 1 hour | Small models, fast convergence |

### 10.2 Manual Deployment

If you need fine-grained control over the deployment (e.g., for debugging or running a single task with custom parameters), you can start the server and clients manually. This is also useful for understanding what the orchestrator does under the hood.

#### Start Server

The server container runs `runners/run_ec2.py --distributed <task>`, which starts the gRPC server on port 9092 and waits for clients to connect.

```bash
source cluster.env

ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP}

docker run -d --name fl-training --network host \
  --restart on-failure:3 \
  --memory 120g \
  --cpus 30 \
  --log-opt max-size=100m \
  --log-opt max-file=5 \
  --health-cmd "python3 -c 'import socket; s=socket.socket(); s.connect((\"127.0.0.1\",9092)); s.close()'" \
  --health-interval=30s \
  --health-timeout=5s \
  --health-retries=3 \
  -v ~/fl-deploy/certs:/certs:ro \
  -v ~/fl-deploy/results:/app/results \
  -e FL_DISTRIBUTED=1 \
  -e SUPERLINK_ADDRESS=0.0.0.0:9092 \
  -e CERTS_DIR=/certs \
  -e PYTHONUNBUFFERED=1 \
  ${FL_IMAGE}:${FL_IMAGE_TAG} \
  python3 runners/run_ec2.py --distributed fraud
```

Key details:

- `--network host` uses the host's network stack directly, avoiding Docker's network translation overhead. This is simpler and faster than bridge networking for single-container-per-host deployments.
- The health check verifies that port 9092 is accepting connections. Docker uses this to report container health status.
- `-v ~/fl-deploy/certs:/certs:ro` mounts the certificate directory as read-only inside the container.
- `-e PYTHONUNBUFFERED=1` ensures that Python log output appears immediately in `docker logs` rather than being buffered.

#### Start Clients

Each client runs `runners/run_client.py` with a unique `PARTITION_ID` that determines which slice of the data it trains on. In horizontal federated learning, the data is split into N partitions (one per client) to simulate different organizations having different subsets of data.

```bash
source cluster.env
PARTITION=0

for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "
    sudo docker run -d --name fl-client --network host --gpus all \
      --restart on-failure:3 \
      --memory 56g \
      --cpus 14 \
      --log-opt max-size=100m \
      --log-opt max-file=5 \
      --health-cmd 'pgrep -f runners/run_client.py' \
      --health-interval=30s \
      --health-timeout=5s \
      --health-retries=3 \
      -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro \
      -v ~/fl-deploy/data:/data:ro \
      -e PARTITION_ID=${PARTITION} \
      -e NUM_CLIENTS=${NUM_CLIENTS} \
      -e FL_TASK=fraud \
      -e FL_SERVER=${SERVER_IP}:9092 \
      -e CERTS_DIR=/certs \
      -e PYTHONUNBUFFERED=1 \
      ${FL_IMAGE}:${FL_IMAGE_TAG} \
      python3 runners/run_client.py
  "
  PARTITION=$((PARTITION + 1))
done
```

Note that:

- `--gpus all` passes the host GPU through to the container. Without this flag, PyTorch will not find any GPU and will fall back to CPU training, which is much slower.
- `-v ~/fl-deploy/data:/data:ro` mounts the data directory as read-only. The container cannot modify the training data.
- `-v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro` mounts only the CA certificate, not the server's private key. Clients only need the CA cert to verify the server's identity.

### 10.3 Server-to-Client SSH Key Setup

The server needs SSH access to client nodes so the orchestrator can manage client containers remotely. This is a one-time setup step.

```bash
source cluster.env

# Copy key to server (one-time)
scp -i ${KEY_PATH} ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP}:~/.ssh/deploy.pem
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "chmod 600 ~/.ssh/deploy.pem"

# Verify connectivity to each client
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
    "ssh -i ~/.ssh/deploy.pem -o StrictHostKeyChecking=no ec2-user@${ip} 'echo OK'"
done
```

Each client should print "OK". If any client fails, check that the security group allows SSH (port 22) from the server's security group, and that the SSH key is correct.

### 10.4 Environment Variables

The following tables list every environment variable recognized by the server and client containers. These are passed via `-e` flags in the `docker run` command.

#### Server

| Variable | Description | Default |
|----------|-------------|---------|
| `FL_DISTRIBUTED` | Enable distributed mode (1/0). Set to 1 for real multi-node training. | `0` (simulation) |
| `SUPERLINK_ADDRESS` | gRPC listen address. `0.0.0.0` means listen on all interfaces. | `0.0.0.0:9092` |
| `CERTS_DIR` | Path to TLS certificates directory inside the container. | `/certs` |
| `SYNTHETIC` | Use synthetic (randomly generated) data instead of real data (1/0). | `0` |
| `DATA_PATH` | Path to data directory inside the container. | `/data` |

#### Client

| Variable | Description | Default |
|----------|-------------|---------|
| `PARTITION_ID` | Client index, from 0 to N-1. Determines which data partition this client trains on. | `0` |
| `NUM_CLIENTS` | Total number of clients in the cluster. Used for data partitioning. | `5` |
| `FL_TASK` | Task name (e.g., fraud, sepsis, ecg). Determines which model and data pipeline to use. | `fraud` |
| `FL_SERVER` | Server address in the format `private_ip:port`. | Required |
| `CERTS_DIR` | Path to CA certificate directory inside the container. | `/certs` |
| `DATA_PATH` | Path to local data inside the container. | `/data` |
| `MAX_SAMPLES` | Cap dataset size (0 = unlimited). Useful for testing with large datasets. | `0` |
| `SYNTHETIC` | Use synthetic data (1/0). | `0` |
| `DATASET_PATH` | Directory containing image data (for chest_xray, satellite tasks). | `/data/chest_xray` |
| `CSV_PATH` | Image metadata CSV filename. | `Data_Entry_2017.csv` |

---

## 11. Tasks and Strategies

This section describes what the platform can train (tasks) and how it trains them (strategies). A "task" is a specific machine learning problem (e.g., predicting sepsis from vital signs). A "strategy" is a federated learning algorithm that governs how model updates are aggregated and what privacy protections are applied.

### 11.1 Task Matrix

The platform includes 14 pre-built tasks covering different data types (tabular, time series, images, text), model architectures (MLP, BiLSTM, DenseNet, TabNet, ResNet, OLMo), and FL paradigms (horizontal, vertical, split, transfer). These tasks serve two purposes:

1. **Demonstration and validation.** Each task proves that federated learning works correctly for a specific combination of data type, model architecture, and FL paradigm. They serve as reference implementations that you can use as starting points for your own tasks.

2. **Production use.** Several tasks (fraud detection, sepsis prediction, ECG analysis, mortality prediction, hospital readmission) directly address common healthcare and financial use cases. If your use case matches one of these tasks, you can use it directly -- just ingest your own data using the data pipeline (Section 8).

Each task has been validated in distributed mode with the indicated number of strategies. "PASS" means all strategies completed without errors and produced valid model metrics. The "Last Verified" column shows the date of the most recent successful distributed run.

| Task | Model | Params | Strategies | Distributed Time | Last Verified |
|------|-------|--------|-----------|-----------------|---------------|
| **fraud** | MLP | 50K | 11 | 68s | 2026-05-27, 11/11 PASS |
| **sepsis** | BiLSTM | 500K | 11 | 98s | 2026-05-28, 11/11 PASS |
| **ecg** | BiLSTM | 200K | 11 | ~5 min | 9/11 PASS (DP strategies use CPU fallback) |
| **anomaly** | Autoencoder | 500K | 11 | 69s | 11/11 PASS |
| **mortality** | TabNet | 1M | 11 | 68s | 11/11 PASS |
| **drug** | Generic MLP | 50K | 11 | ~3 min | Needs re-test |
| **readmission** | LogReg | 10K | 11 | 69s | 11/11 PASS |
| **satellite** | ResNet-small | 5M | 7 | 339s | 7/7 PASS |
| **chest** | DenseNet-121 | 8M | 7 | 429s (synthetic) | 7/7 PASS |
| **vfl** | VFL MLP | 50K | 4 | 38s | 4/4 PASS |
| **split** | Split BiLSTM | 500K | 3 | 38s | 3/3 PASS |
| **transfer** | DenseNet-121 | 8M | 2 | 69s (synthetic) | 2/2 PASS |
| **olmo** | OLMo-1B QLoRA | 2.1M (LoRA) | 3 | 96s | 2026-05-28, perplexity 1.13 |
| **privacy** | BiLSTM + MLP | - | 3 attacks | ~10 min | PASS |

Reading this table:

- **Params** is the number of trainable parameters in the model. More parameters generally means a more capable model but also more data needed, more compute time, and larger network transfers.
- **Strategies** is how many different FL strategies are tested for each task. The 11-strategy tasks run the full suite: IID, FedProx, SCAFFOLD, SecAgg, DP variants, and OneOwner. Tasks with fewer strategies are limited by their FL paradigm (e.g., VFL only supports 4 strategies).
- **Distributed Time** is the wall-clock time for a complete run with 5 clients. This includes all strategies, not just one.
- **DP strategies use CPU fallback** means that differential privacy noise injection triggers numerical issues on GPU for that particular model, so the code automatically falls back to CPU for those strategies. Training is slower but still completes correctly.

**Understanding the "privacy" task:** The `privacy` task is special -- it does not train a model for prediction. Instead, it runs the three privacy attack tests (gradient inversion, membership inference, and canary extraction) on trained models to empirically measure how much information the models leak. Run this task after training to verify that your privacy protections are working as expected.

### 11.2 Strategy Reference

Strategies control how model updates are aggregated and what privacy protections are applied. The following table describes each strategy. Choosing the right strategy depends on your data distribution and privacy requirements.

| Strategy Name | Description | Non-IID | Privacy |
|--------------|-------------|---------|---------|
| `IID` | FedAvg with uniform data split | No | No |
| `FedProx_Mu0.1_Alpha_0.5` | FedProx, moderate non-IID | Moderate | No |
| `FedProx_Mu0.1_Alpha_0.1` | FedProx, extreme non-IID | Extreme | No |
| `SCAFFOLD_Alpha_0.5` | SCAFFOLD with control variates | Moderate | No |
| `SCAFFOLD_Alpha_0.1` | SCAFFOLD, extreme non-IID | Extreme | No |
| `SecAgg_Alpha_0.5` | Secure Aggregation (pairwise masks) | Moderate | Server can't see updates |
| `DP_Central_Eps50.0_Alpha_0.5` | Central DP, epsilon=50 | Moderate | Formal guarantee |
| `DP_Central_Eps10.0_Alpha_0.5` | Central DP, epsilon=10 | Moderate | Stronger guarantee |
| `DP_Local_Eps50.0_Alpha_0.5` | Local DP, epsilon=50 | Moderate | Client-side noise |
| `DP_Local_Eps10.0_Alpha_0.5` | Local DP, epsilon=10 | Moderate | Strong client noise |
| `OneOwner_Alpha_0.5` | Single owner, all contribute | Moderate | Access control |

Understanding the key parameters:

- **Non-IID** refers to how different each client's data is from the others. "IID" (Independent and Identically Distributed) means all clients have similar data distributions. In practice, this rarely holds -- Hospital A may see mostly elderly patients while Hospital B sees mostly young patients. "Alpha" controls the degree of non-IID-ness: alpha=0.5 is moderate (clients have somewhat different data), alpha=0.1 is extreme (clients have very different data).

- **FedProx** adds a "proximal term" that penalizes client models for diverging too far from the global model. This stabilizes training when clients have very different data. The `Mu` parameter controls the strength of this penalty.

- **SCAFFOLD** uses "control variates" to correct for the drift caused by non-IID data. Each client maintains a correction term that adjusts its gradient to account for its local data bias. SCAFFOLD generally converges faster than FedProx on non-IID data.

- **SecAgg (Secure Aggregation)** uses cryptographic masking so the server can compute the aggregate of all client updates without seeing any individual update. This is the primary defense against a curious server operator.

- **DP (Differential Privacy)** adds calibrated random noise to model updates. "Central" DP adds noise at the server after aggregation (requires trusting the server). "Local" DP adds noise at each client before sending (does not require trusting the server, but degrades accuracy more).

- **Epsilon** is the privacy budget. Lower epsilon = stronger privacy = more noise = less accuracy. Epsilon=10 provides meaningful privacy protection. Epsilon=50 provides moderate protection with less accuracy loss.

### 11.3 Choosing a Strategy

Choosing the right strategy depends on three factors: how different the data is across sites (data heterogeneity), what privacy guarantees are required, and how much accuracy degradation is acceptable. Here is a decision guide:

**If your data is similar across sites (IID):**
Start with `IID` (FedAvg). This is the simplest strategy and produces the best accuracy when data distributions are similar. Example: all participating hospitals serve similar patient populations.

**If your data differs moderately across sites (non-IID, alpha=0.5):**
Use `FedProx_Mu0.1_Alpha_0.5` or `SCAFFOLD_Alpha_0.5`. SCAFFOLD generally converges faster than FedProx because it uses control variates to correct for data heterogeneity, but it requires more memory on each client. If memory is constrained, use FedProx. Example: hospitals in different cities serve somewhat different patient populations.

**If your data differs dramatically across sites (extreme non-IID, alpha=0.1):**
Use `SCAFFOLD_Alpha_0.1`. Extreme non-IID is common when sites specialize (e.g., a cardiac hospital vs. a pediatric hospital). SCAFFOLD handles this better than FedProx because the control variates explicitly compensate for each site's data bias.

**If you do not trust the server operator:**
Add `SecAgg_Alpha_0.5` to your strategy list. Secure Aggregation ensures the server cannot inspect individual client updates. This is essential when the server is operated by a third party. You can combine SecAgg with non-IID strategies by modifying the strategy configuration.

**If you need formal privacy guarantees (regulatory requirement):**
Use `DP_Central_Eps10.0_Alpha_0.5` (if you trust the server) or `DP_Local_Eps10.0_Alpha_0.5` (if you do not trust the server). The choice between epsilon=10 and epsilon=50 depends on the sensitivity of the data and the accuracy requirements. Run both and compare accuracy metrics.

**If you want to test everything:**
Run the full 11-strategy suite. The orchestrator runs all strategies in sequence and produces a comparative results file. This is recommended for initial deployments to understand how each strategy performs on your specific data.

### 11.4 Running the Privacy Attack Tests

After training, run the privacy attack suite to empirically measure how much information the trained model leaks:

```bash
# Run all privacy tests (MIA + gradient inversion)
python -m privacy.test_privacy

# Output shows:
#   - Gradient inversion: cosine similarity without DP vs. with DP
#   - Membership inference: loss gap between members and non-members
#   - Reduction in attack effectiveness with DP enabled
```

If the MIA attack shows a large loss gap between members and non-members, the model is memorizing training data. Consider using a stronger DP epsilon, reducing the number of training epochs, or adding regularization (dropout, weight decay) to the model.

---

## 12. Federated Adapter Framework

### 12.1 The Challenge of Federated Learning with Large Language Models

Large language models (LLMs) like OLMo-1B, Llama-3-8B, or Mistral-7B have billions of parameters. Sending an entire LLM's weights from each client to the server every round would require transmitting gigabytes of data per round per client, making federated learning impractical. A 7-billion parameter model in 32-bit precision is approximately 28 GB -- sending this from 5 clients for 10 rounds means 1.4 TB of network transfer.

Additionally, training an entire LLM requires enormous amounts of data and compute. Individual sites in a federated learning scenario typically have small, domain-specific datasets that are insufficient to retrain all parameters.

### 12.2 The Solution: Freeze the Base, Train Only Adapters

The federated adapter framework, inspired by the FlexOLMo approach from the Allen Institute for AI (AI2), solves both problems by applying a technique called LoRA (Low-Rank Adaptation):

1. **Freeze the base model.** The pre-trained LLM weights are frozen -- they are not modified during training. Every site downloads the same base model (e.g., OLMo-1B from HuggingFace) and keeps it unchanged.

2. **Attach small adapter layers.** LoRA inserts small trainable matrices into the model's attention layers. These adapter matrices typically have 0.1-1% as many parameters as the full model. For OLMo-1B (1 billion parameters), the LoRA adapters have about 2.1 million parameters -- 0.2% of the total.

3. **Train only the adapters.** Each site trains the adapter weights on its local data. Only the adapter weights (2.1M parameters = ~8 MB) are sent to the server each round, not the full model (1B parameters = ~4 GB).

4. **Aggregate adapters on the server.** The server averages the adapter weights from all sites, producing a single improved set of adapters.

5. **Combine for inference.** To use the model, combine the frozen base model with the aggregated adapters. The result is a full LLM that has been fine-tuned on the combined knowledge from all sites.

This approach reduces network transfer by 99.8% (from ~4 GB per round to ~8 MB per round) while still allowing each site to contribute its domain-specific knowledge to the model.

### 12.3 Supported Models

The adapter framework is generic -- it works with any HuggingFace model that supports LoRA. The following presets are pre-configured:

| Preset | Model | Adapter/Round | Base Size (4-bit) | Use Case |
|--------|-------|--------------|-------------------|----------|
| `olmo-1b` | allenai/OLMo-1B-hf | 32 MB | 0.5 GB | Gov documents (tested) |
| `llama-3-8b` | meta-llama/Meta-Llama-3-8B | 256 MB | 4.0 GB | General LLM |
| `mistral-7b` | mistralai/Mistral-7B-v0.3 | 224 MB | 3.5 GB | General LLM |
| `phi-3-mini` | microsoft/Phi-3-mini-4k-instruct | 16 MB | 0.2 GB | Lightweight LLM |
| `bert-base` | bert-base-uncased | 8 MB | 2.0 GB | Text classification/NER |
| `biobert` | dmis-lab/biobert-v1.1 | 8 MB | 2.0 GB | Medical NER |
| `vit-base` | google/vit-base-patch16-224 | 8 MB | 2.0 GB | Image classification |
| `whisper-small` | openai/whisper-small | 8 MB | 2.0 GB | Speech-to-text |

**Base Size (4-bit)** refers to the model size when quantized to 4-bit precision using QLoRA. Quantization reduces the memory required to load the model by approximately 4x at the cost of a small accuracy reduction. The `olmo-1b` preset at 0.5 GB fits comfortably on a single L4 GPU (24 GB).

### 12.4 Switching Models

To use a different base model, set the `ADAPTER_PRESET` environment variable. No code changes are needed:

```bash
# Switch model by setting environment variable
ADAPTER_PRESET=llama-3-8b FL_TASK=olmo python runners/run_client.py
```

The framework automatically downloads the base model from HuggingFace (on first use), configures the LoRA adapters with the preset's parameters, and handles the adapter weight extraction and injection during FL rounds.

### 12.5 The FlexOLMo Inspiration

The federated adapter framework is inspired by the FlexOLMo approach developed by the Allen Institute for AI (AI2). The key insight from FlexOLMo is that for domain-specific fine-tuning, you do not need to modify the entire base model. The base model already contains general language understanding (grammar, facts, reasoning). Domain-specific knowledge (medical terminology, legal language, financial jargon) can be captured in small adapter layers that modify the base model's behavior without changing its core parameters.

In the federated context, this insight is particularly powerful:

- **Each agency trains adapters on their private documents.** A healthcare agency trains adapters on medical records. A finance agency trains adapters on financial reports. A research agency trains adapters on scientific papers.
- **Only adapters cross the network.** The base model (potentially gigabytes) stays local. Only the adapters (megabytes) are sent to the server.
- **The server aggregates adapters, not full models.** Averaging adapter weights is equivalent to averaging the "domain knowledge" from all agencies, without needing to see any agency's raw data.
- **The base model is public.** Since the base model is a publicly available pre-trained model (e.g., from HuggingFace), sharing it is not a privacy concern. Only the adapters contain information derived from private data.

### 12.6 When to Use the Adapter Framework

Use the federated adapter framework when:

- Your task involves natural language (document classification, named entity recognition, summarization) or image classification using a pre-trained model.
- You want to leverage a large pre-trained model but cannot centralize the fine-tuning data.
- Network bandwidth between sites is limited (adapter updates are 100-1000x smaller than full model updates).

Do not use the adapter framework when:

- Your task is simple enough for a small custom model (e.g., tabular fraud detection with 50K parameters). The overhead of loading a billion-parameter base model is unnecessary.
- You need to modify the model's core architecture (LoRA only adapts existing layers; it cannot add new ones).
- The base model does not cover your domain at all (e.g., using a language model for audio without a speech-to-text head).

### 12.7 Performance

The `olmo` task with the `olmo-1b` preset completes in 96 seconds for 3 strategies with 5 clients. The final model achieves a perplexity of 1.13 on the test set, indicating excellent language modeling quality. The adapter weights per round are approximately 32 MB, compared to approximately 4 GB for the full model -- a 99.2% reduction in network transfer.

---

## 13. Secure Inference

### 13.1 What Problem Secure Inference Solves

After training a model via federated learning, you have a trained model that can make predictions. But using this model raises a new privacy question: what if the organization that hosts the model (the "model owner") should not see the data being submitted for prediction, and the organization submitting data (the "data owner") should not see the model's weights?

Example: a central government agency has trained a disease prediction model using federated learning. A hospital wants to submit a patient's records to get a prediction, but regulations prohibit sharing patient data with the government agency. The agency, in turn, does not want to share the model weights with the hospital (the model may be classified or proprietary).

Secure inference solves this by allowing predictions to be computed on encrypted data. The data owner encrypts their input, sends the encrypted input to the model owner, the model owner runs inference on the encrypted data (without ever decrypting it), and the encrypted result is sent back to the data owner, who decrypts it. At no point does the model owner see the input data or the prediction result.

### 13.2 How CKKS Homomorphic Encryption Works

The platform uses TenSEAL, a Python library that wraps Microsoft SEAL, to implement the CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme. Here is a simplified explanation:

**Homomorphic encryption** allows mathematical operations (addition, multiplication) to be performed on encrypted data. If you encrypt the number 3 and the number 5, you can add the encrypted values and get an encryption of 8 -- without ever decrypting the individual values.

**CKKS** is a specific homomorphic encryption scheme designed for approximate arithmetic on real numbers. It is well-suited for machine learning because neural network computations are inherently approximate (floating-point operations have rounding errors anyway).

The key limitation of CKKS is **multiplicative depth**. Each multiplication on encrypted data introduces noise in the ciphertext. After a certain number of multiplications (determined by the encryption parameters), the noise overwhelms the signal and decryption produces garbage. For the platform's configuration (poly_modulus_degree=16384), approximately 8 levels of multiplicative depth are available. This is enough for a 3-layer MLP with polynomial activation functions but not for deep networks like DenseNet-121.

### 13.3 Supported Models and the Hybrid Approach

Because of the multiplicative depth limitation, different models require different approaches:

**MLP (fraud detection) -- Full encrypted inference.** The MLP has 3 linear layers with polynomial activation functions (square activation instead of ReLU, because ReLU is non-polynomial and cannot be computed on encrypted data). The entire forward pass runs on encrypted data. This provides the strongest privacy guarantee: the model owner never sees the input or the output.

**BiLSTM (sepsis/ECG) -- Hybrid approach.** LSTM gates involve multiple multiplications per time step, quickly exhausting the multiplicative depth budget. The practical approach is to split the computation: the data owner runs the LSTM layers locally (they have the data anyway) to produce an "embedding" (a compressed representation), encrypts the embedding, and sends it to the model owner. The model owner runs only the final classification layer on the encrypted embedding. The data owner never shares raw data; the model owner never sees the embedding in plaintext.

**DenseNet-121 (chest X-ray) -- Hybrid with encrypted input/output.** DenseNet-121 has 121 layers and would require hundreds of multiplicative levels for full encrypted inference. The practical approach is similar to BiLSTM: encrypt the input image and the output prediction, but perform the intermediate computation in a trusted execution environment or with other privacy controls.

### 13.4 Activation Function Approximations

Standard neural network activation functions (ReLU, sigmoid, tanh) cannot be computed on encrypted data because they are non-polynomial. CKKS only supports addition and multiplication. The platform uses polynomial approximations:

- **Sigmoid:** approximated as `0.5 + 0.197*x - 0.004*x^3`. Accurate to ~0.01 within the range [-4, 4].
- **ReLU:** approximated as `x^2` (square activation). This is less accurate than true ReLU but only requires one multiplication, preserving multiplicative depth.
- **Tanh:** approximated as `x - x^3/3` (first two terms of the Taylor series). Accurate within [-1, 1].

These approximations mean that encrypted inference produces slightly different results than plaintext inference. In our benchmarks, the maximum error for the MLP model is less than 0.02 (on a 0-1 scale), which is acceptable for classification tasks.

### 13.5 Performance Numbers

Encrypted inference is significantly slower than plaintext inference due to the computational cost of homomorphic operations. Here are honest performance numbers from our benchmarks:

| Model | Plaintext (per sample) | Encrypted (per sample) | Slowdown | Max Error |
|-------|----------------------|----------------------|----------|-----------|
| MLP (30 features) | ~0.1 ms | ~50-200 ms | ~500-2000x | < 0.02 |
| BiLSTM classifier (128-dim embedding) | ~0.2 ms | ~100-500 ms | ~500-2500x | < 0.03 |

The slowdown is substantial but acceptable for batch inference scenarios where privacy is paramount. For real-time applications (e.g., < 10ms latency required), encrypted inference is not yet practical. In those cases, consider trusted execution environments (TEEs) or on-premises model deployment instead.

### 13.6 When to Use Secure Inference

Use CKKS homomorphic encryption when:

- The data owner and model owner are different organizations that do not trust each other.
- Latency requirements are relaxed (batch processing, overnight jobs).
- The model is small (MLP, logistic regression, small classifiers).

Consider alternatives when:

- Low latency is required (use TEEs or on-premises deployment).
- The model is very deep (use the hybrid approach or TEEs).
- Both parties are within the same trust boundary (plain inference is fine).

### 13.7 Security Levels

The TenSEAL CKKS implementation supports two security levels, corresponding to different polynomial modulus degrees:

**128-bit security (default):** Uses `poly_modulus_degree=16384`. Provides 128-bit equivalent security -- meaning an attacker would need approximately 2^128 operations to break the encryption. This is the standard security level recommended by NIST and is considered secure against all known attacks, including quantum computing attacks up to approximately 2040. Provides 8 levels of multiplicative depth, sufficient for a 3-layer MLP.

**192-bit security:** Uses `poly_modulus_degree=32768`. Provides 192-bit equivalent security with 12 levels of multiplicative depth. This is recommended only for classified or extremely sensitive data. The trade-off is approximately 4x slower computation compared to 128-bit security.

**Why not 256-bit?** Higher polynomial modulus degrees (required for 256-bit security) dramatically increase computation time and ciphertext size. For current threat models, 128-bit security is considered sufficient for all non-classified government data. Consult your organization's cryptography standards for specific requirements.

### 13.8 The Encryption Workflow in Detail

A complete secure inference interaction follows this sequence:

1. **Key generation (data owner, one-time setup):** The data owner generates a public key and a secret key. The public key is shared with the model owner. The secret key is kept private -- it is the only key that can decrypt results.

2. **Encryption (data owner):** The data owner encrypts their input data (e.g., a patient's 30 lab values for fraud detection) using their public key. The resulting ciphertext is approximately 100-500x larger than the plaintext, depending on the encryption parameters.

3. **Encrypted computation (model owner):** The model owner receives the ciphertext and runs the neural network forward pass on it. Each linear layer computes matrix multiplication on encrypted vectors. Each activation function uses a polynomial approximation. The model owner never decrypts the input -- they cannot, because they do not have the secret key.

4. **Result transmission (model owner to data owner):** The model owner sends the encrypted output (the model's prediction, still encrypted) back to the data owner.

5. **Decryption (data owner):** The data owner decrypts the result using their secret key. They now have the model's prediction without the model owner ever having seen their input data or the prediction.

### 13.9 Running the Benchmarks

To verify secure inference on your deployment:

```bash
# Run all benchmarks (MLP, BiLSTM, DenseNet)
python -m secure_inference.tenseal_inference

# Run a specific model benchmark
python -m secure_inference.tenseal_inference --model mlp
python -m secure_inference.tenseal_inference --model bilstm
```

The benchmark output shows plaintext predictions, encrypted predictions, the difference between them, and timing breakdowns for encryption, computation, and decryption. It also proves that ciphertexts are real by displaying the ciphertext size and showing that the raw bytes are unreadable.

---

## 14. Monitoring and Observability

Monitoring a distributed FL cluster is more complex than monitoring a single application because failures can be partial and silent. Unlike a web server where downtime is immediately visible (users see errors), a federated learning system can appear to be running normally even when things are going wrong:

- **A failed client silently degrades model quality.** If one of five clients crashes, the remaining four continue training. The server aggregates updates from four clients instead of five. Training completes successfully, but the final model is less accurate because it missed data from the fifth site. Without monitoring, this goes undetected.
- **A straggler client extends training time.** In synchronous FL (the default), the server waits for all clients before aggregating. If one client is slow (due to a smaller GPU, thermal throttling, or a larger dataset), it delays every round. A 2x slowdown on one client can double total training time.
- **A hung server wastes compute hours.** If the server enters a deadlock (e.g., waiting for a client that already crashed), all clients sit idle, consuming GPU hours with no training progress.

This section provides the commands and configurations needed to maintain visibility into cluster health, detect these issues early, and respond appropriately.

### 14.1 Health Checks

#### Container Health

Docker health checks are configured in the `docker run` commands (Section 10). They run periodically inside each container and report the container's health status. The following commands query health status across the cluster.

```bash
source cluster.env

# Check container health status across cluster
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker inspect --format='{{.Name}}: {{.State.Health.Status}}' fl-training 2>/dev/null || echo 'fl-training: not running'"

for ip in ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "sudo docker inspect --format='{{.Name}}: {{.State.Health.Status}}' fl-client 2>/dev/null || echo 'fl-client: not running'"
done
```

Possible health statuses:

- `healthy` -- the health check passed. The container is running and responsive.
- `unhealthy` -- the health check failed 3 consecutive times. The container may be hung or crashed.
- `starting` -- the container just started and health checks have not yet completed.

#### Readiness Probe

To check if the server is ready to accept client connections:

```bash
# Server readiness: verify gRPC port is listening
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "ss -tlnp | grep 9092 && echo 'READY' || echo 'NOT READY'"
```

### 14.2 Cluster Status

The following command provides a quick overview of all containers and GPU utilization across the cluster:

```bash
source cluster.env

# Server status
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker ps --format '{{.Names}}: {{.Status}}'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"

# All clients
for ip in ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "sudo docker ps --format '{{.Names}}: {{.Status}}'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
done
```

### 14.3 Training Monitoring

During training, the server logs detailed information about each round: which clients participated, the aggregated metric, and timing. The following commands help you monitor training progress.

```bash
source cluster.env

# Live log stream (server)
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training -f --tail 20"

# Round progress
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training 2>&1 | grep 'Round [0-9]' | tail -5"

# Strategy completion
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training 2>&1 | grep 'Final'"

# Client logs
ssh -i ${KEY_PATH} ec2-user@<CLIENT_IP> "sudo docker logs fl-client --tail 10"
```

The "Round" lines show which round is currently in progress. The "Final" lines appear when a strategy completes, showing the final aggregated metric. These are the key indicators of training health.

### 14.4 GPU Monitoring

GPU utilization and temperature are important to monitor, especially during long training runs. Sustained high temperatures can cause thermal throttling, which reduces GPU clock speed and slows training.

```bash
source cluster.env

# GPU usage across cluster
for ip in ${SERVER_PUBLIC_IP} ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
done

# Continuous watch on server
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "watch -n 5 nvidia-smi"
```

### 14.5 Log Aggregation

For production deployments, centralized log aggregation is essential. When troubleshooting a failed training run, you need logs from the server and all clients in one place.

#### CloudWatch (Recommended)

Configure Docker to ship logs directly to CloudWatch Logs. This provides automatic retention, search, and alerting.

```bash
# /etc/docker/daemon.json on each node
{
  "log-driver": "awslogs",
  "log-opts": {
    "awslogs-region": "<REGION>",
    "awslogs-group": "/fl-platform/containers",
    "awslogs-create-group": "true",
    "tag": "{{.Name}}/{{.ID}}"
  }
}
```

After adding this configuration, restart Docker (`sudo systemctl restart docker`). All subsequent container logs will be shipped to the specified CloudWatch log group. The `tag` option labels each log stream with the container name, making it easy to filter for specific containers.

#### File-Based (Alternative)

For environments without CloudWatch access, collect logs manually after each run:

```bash
# Collect logs from all nodes
source cluster.env
mkdir -p logs/$(date +%Y%m%d)

ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker logs fl-training" > logs/$(date +%Y%m%d)/server.log 2>&1

IDX=0
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "sudo docker logs fl-client" > logs/$(date +%Y%m%d)/client_${IDX}.log 2>&1
  IDX=$((IDX + 1))
done
```

### 14.6 Alerting

Set up CloudWatch alarms to be notified of issues before they become outages. The following table lists the recommended alerts:

| Metric | Threshold | Action |
|--------|-----------|--------|
| GPU memory utilization | > 95% for 5 min | Notify ops (may indicate memory leak or model too large) |
| Container restart count | > 3 in 10 min | Notify ops, check logs (container is crash-looping) |
| Disk usage | > 85% | Notify ops, prune images (Docker images and logs can fill disks) |
| Training round duration | > 2x baseline | Investigate straggler (one slow client delays all others) |
| gRPC port 9092 unreachable | > 60s | Page on-call (server is down, training is halted) |

Example CloudWatch alarm for disk usage:

```bash
# Example: CloudWatch alarm for disk usage
aws cloudwatch put-metric-alarm \
  --alarm-name fl-server-disk-usage \
  --metric-name disk_used_percent \
  --namespace CWAgent \
  --statistic Average \
  --period 300 \
  --threshold 85 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions <SNS_TOPIC_ARN>
```

---

## 15. Security

Security in a federated learning system has unique considerations beyond standard IT security. Understanding the threat model is essential for making informed security decisions.

### The FL-Specific Threat Model

In a traditional centralized ML system, you protect one data store and one compute cluster. In federated learning, the attack surface is larger:

- **Network eavesdropping.** Model updates transit the network between clients and server. As discussed in Section 2, these updates carry information about training data. An eavesdropper could perform gradient inversion attacks. TLS encryption (Section 9) mitigates this.

- **Malicious server (honest-but-curious).** The server operator follows the protocol correctly but tries to extract information from the individual client updates it receives. This is a realistic threat when the server is operated by a different organization than the data owners. Secure Aggregation (Section 11.2) mitigates this by ensuring the server only sees the aggregate, not individual updates.

- **Malicious server (active attacker).** The server operator sends crafted model updates to clients designed to cause them to leak more information in their responses. For example, the server could send a model where only one neuron has non-zero weights, and the client's gradient for that neuron would directly reveal information about a specific feature. This is a harder threat to defend against. Differential privacy provides some protection because the noise masks the leaked signal regardless of the server's strategy.

- **Compromised client.** One of the participating organizations (or an attacker who has compromised their node) submits malicious updates designed to poison the model (make it produce incorrect predictions) or extract information about other clients' data through the aggregated model. Byzantine-robust aggregation algorithms can partially mitigate this, though they are not yet implemented in this platform.

- **Container escape.** An attacker who compromises the application inside a container attempts to break out of the container and access the host system, other containers, or the network. Container hardening (Section 7.4) mitigates this.

This section covers network segmentation, secret management, audit logging, and a comprehensive security checklist that addresses each of these threats.

### 15.2 Network Segmentation

Network segmentation limits the blast radius of a compromised node. If an attacker gains access to a client node, they should not be able to reach the server's management interface or other clients' data.

The following zone structure separates traffic by function:

| Zone | Nodes | Allowed Traffic |
|------|-------|----------------|
| **Server subnet** | FL Server, Orchestrator | Inbound 9092 from client subnet only; SSH from bastion only |
| **Client subnet** | FL Clients | Outbound 9092 to server only; SSH from bastion only |
| **Bastion / VPN** | Operator access | SSH to server/client subnets |

Security group rules implement this segmentation:

```bash
# Server security group
aws ec2 authorize-security-group-ingress \
  --group-id ${SG_SERVER} \
  --protocol tcp --port 9092 \
  --source-group ${SG_CLIENTS}

aws ec2 authorize-security-group-ingress \
  --group-id ${SG_SERVER} \
  --protocol tcp --port 22 \
  --source-group ${SG_BASTION}

# Client security group
aws ec2 authorize-security-group-ingress \
  --group-id ${SG_CLIENTS} \
  --protocol tcp --port 22 \
  --source-group ${SG_BASTION}
```

Key points:

- Port 9092 (FL traffic) is only allowed from the client security group to the server security group. No external access.
- SSH (port 22) is only allowed from the bastion/VPN. Operators must connect through the bastion to reach any cluster node.
- Clients have no inbound rules for port 9092 -- they only make outbound connections to the server. This prevents clients from being used as rogue servers.

### 15.3 Secret Management

TLS private keys and SSH keys are the most sensitive assets in the deployment. If an attacker obtains the server's TLS private key, they can impersonate the server and intercept all model updates. If they obtain the SSH key, they can access any node in the cluster.

**Do not store SSH keys or TLS private keys on disk unprotected.** Use AWS Secrets Manager or SSM Parameter Store:

```bash
# Store SSH key in Secrets Manager
aws secretsmanager create-secret \
  --name fl-platform/ssh-key \
  --secret-binary fileb://${KEY_PATH}

# Store TLS server key
aws secretsmanager create-secret \
  --name fl-platform/server-tls-key \
  --secret-binary fileb://certs/server.key

# Retrieve at deployment time
aws secretsmanager get-secret-value \
  --secret-id fl-platform/ssh-key \
  --query SecretBinary --output text | base64 --decode > /tmp/deploy.pem
chmod 600 /tmp/deploy.pem
# Use /tmp/deploy.pem, then shred it after deployment
shred -u /tmp/deploy.pem
```

The `shred -u` command overwrites the file multiple times before deleting it, preventing recovery of the key from disk. This is important because standard file deletion does not erase the data -- it only removes the directory entry, leaving the key recoverable with forensic tools.

### 15.4 Audit Logging

For regulated environments, you need a complete audit trail of who accessed the cluster, what containers were started, and what data was processed. Enable both CloudTrail (for AWS API calls) and Docker audit logging (for container operations).

```bash
# Docker daemon audit logging
# /etc/docker/daemon.json
{
  "log-level": "info",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  }
}

# Linux audit rules for Docker socket
sudo auditctl -w /var/run/docker.sock -k docker-socket
sudo auditctl -w /etc/docker -k docker-config
```

The `auditctl` rules monitor the Docker socket (used for all Docker API calls) and the Docker configuration directory. Any access to these resources is logged in the Linux audit log (`/var/log/audit/audit.log`), which can be shipped to a SIEM for analysis.

### 15.5 Security Checklist

Use this checklist before every production deployment. Each item addresses a specific threat.

#### Pre-Deployment

- [ ] TLS certificates generated with correct SANs (server private IP)
- [ ] SSH keys stored in Secrets Manager, not on local disk
- [ ] Security groups restrict port 9092 to client security group only
- [ ] Security groups restrict SSH (22) to bastion/VPN only
- [ ] No `--insecure` flags in any command
- [ ] EBS volumes encrypted with KMS CMK
- [ ] IMDSv2 enforced on all instances (prevents SSRF attacks from extracting instance credentials)
- [ ] Docker containers run with resource limits

#### Data

- [ ] Patient data only on client machines, never on server
- [ ] Data directories mounted read-only (`:ro`) in Docker
- [ ] Data manifests validated before training

#### Operations

- [ ] Instances stopped when not in use
- [ ] Docker logs have size limits and rotation
- [ ] GPU driver matches CUDA version in Docker image
- [ ] Results collected and backed up before instance termination
- [ ] Certificate expiry monitored

#### Post-Training

- [ ] MIA attack run on final model (check for data leakage using `privacy/test_privacy.py`)
- [ ] Privacy budget (epsilon) reviewed if DP was used
- [ ] Results JSON archived to S3
- [ ] Temporary files cleaned up (`/tmp/fl-image.tar.gz`)
- [ ] Audit logs reviewed

---

## 16. Backup and Recovery

Hardware failures, accidental deletions, and configuration errors can destroy training results that took hours or days to produce. This section covers what to back up, how to automate backups, and how to recover from common failure scenarios.

### 16.1 What to Back Up

The following table lists every asset that should be backed up, where it lives, how often it changes, and where to store the backup.

| Asset | Location | Frequency | Destination |
|-------|----------|-----------|-------------|
| Training results | `~/fl-deploy/results/*.json` | After each run | S3 bucket |
| TLS certificates | `~/fl-deploy/certs/` | On rotation | Secrets Manager |
| Data manifests | `~/fl-deploy/data/*/manifest.json` | On change | S3 bucket |
| Orchestrator config | `cluster.env`, `run_server_side.sh` | On change | Version control |
| Docker image | `healthcare-fl:latest` | On rebuild | ECR |

Training results are the most important asset to back up. They represent hours of compute time and cannot be regenerated without re-running the entire training pipeline. TLS certificates and Docker images can be regenerated, but regeneration requires downtime.

### 16.2 Automated Backup

Run the following after each training run to archive results and manifests to S3:

```bash
source cluster.env

# Back up results to S3
aws s3 sync ~/fl-deploy/results/ s3://${BACKUP_BUCKET}/results/$(date +%Y%m%d)/ \
  --sse aws:kms --sse-kms-key-id ${KMS_KEY_ID}

# Back up manifests
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "find ~/fl-deploy/data -name 'manifest.json' -exec cat {} \;" \
    > manifests/client_${ip}.json
done
aws s3 cp manifests/ s3://${BACKUP_BUCKET}/manifests/$(date +%Y%m%d)/ --recursive
```

The `--sse aws:kms` flag encrypts the backup at rest using your KMS key. This is essential for regulated data -- even backups must be encrypted.

### 16.3 Recovery Procedures

#### Scenario: Server Instance Lost

This can happen due to hardware failure, accidental termination, or spot instance reclamation.

1. Launch new server instance (same type, same VPC/subnet)
2. Install GPU driver and Docker (Section 6.3-6.4)
3. Restore TLS certs from Secrets Manager
4. Pull Docker image from ECR
5. Regenerate server TLS cert with new private IP (Section 9.3) -- the new instance will have a different private IP
6. Distribute new `ca.pem` if CA was rotated
7. Update `cluster.env` with new server IP
8. Resume training from last checkpoint

Recovery time: approximately 30-60 minutes.

#### Scenario: Client Instance Lost

1. Launch replacement instance
2. Install GPU driver and Docker
3. Copy CA cert to new node
4. Re-ingest data from source (data stays at each site -- the source organization must provide the data again)
5. Pull Docker image from ECR
6. Update `cluster.env` with new client IP
7. Client will join on next orchestrator cycle

Recovery time: approximately 30-60 minutes plus data re-ingestion time.

#### Scenario: Corrupted Training Results

1. Check S3 backup for latest valid results
2. Restore: `aws s3 sync s3://${BACKUP_BUCKET}/results/<date>/ ~/fl-deploy/results/`
3. Re-run affected tasks if backup is stale

#### Scenario: TLS Certificate Expired During Training

If certificates expire during an active training run, clients will fail to reconnect between strategies.

1. Generate new certificates using Section 9.2-9.4
2. Distribute to all nodes
3. Restart all containers (the orchestrator will handle this on the next task)
4. Resume training from the current task (previous tasks' results are preserved)

#### Scenario: Complete Cluster Rebuild

If the entire cluster needs to be rebuilt from scratch (e.g., moving to a new AWS region):

1. Ensure all results are backed up to S3
2. Ensure all TLS certificates are stored in Secrets Manager
3. Provision new instances (Section 6)
4. Install GPU drivers and Docker on all nodes (Section 6.3-6.4)
5. Pull Docker image from ECR on all nodes
6. Restore certificates from Secrets Manager (regenerate server cert with new IPs)
7. Copy data to new client nodes from the source organizations
8. Create new `cluster.env` with new IP addresses
9. Run a smoke test with the `fraud` task before running the full suite

---

## 17. Capacity Planning

This section helps you estimate the compute, memory, and storage resources needed for your deployment. Under-provisioning leads to out-of-memory errors and slow training; over-provisioning wastes money.

### 17.1 GPU Memory by Task

GPU memory is the most common bottleneck in machine learning systems. Each model requires a certain amount of GPU memory to store its parameters, the training data batch, intermediate computations, and optimizer state. If the model requires more memory than the GPU has, training fails with a "CUDA out of memory" error.

The following table shows how much GPU memory each task requires per client during training. The batch size column shows the default setting -- reducing batch size reduces memory usage but may affect training quality.

| Task | Model | GPU Memory (per client) | Batch Size |
|------|-------|------------------------|------------|
| fraud, drug, readmission | MLP/LogReg | < 1 GB | 256 |
| sepsis, ecg | BiLSTM | ~2 GB | 128 |
| anomaly | Autoencoder | ~2 GB | 128 |
| mortality | TabNet | ~3 GB | 64 |
| satellite | ResNet-small | ~6 GB | 32 |
| chest, transfer | DenseNet-121 | ~12 GB | 16 |

The L4 GPU (24 GB) is sufficient for all current models with room to spare. DenseNet-121 uses the most GPU memory at ~12 GB, leaving 12 GB of headroom.

### 17.2 Scaling Clients

Adding more clients increases the diversity of data the model sees but also increases network overhead and aggregation time. The following table summarizes the trade-offs:

| Clients | Impact on Training Time | Network Overhead |
|---------|------------------------|------------------|
| 2-5 | Baseline | Negligible |
| 5-10 | ~same per round, more aggregation | Low |
| 10-20 | Aggregation becomes bottleneck | Moderate |
| 20+ | Requires async aggregation or hierarchical FL | High |

When scaling beyond 10 clients:

- Consider asynchronous aggregation strategies, where the server does not wait for all clients before aggregating.
- Use hierarchical FL with regional aggregators, where clients report to a regional server that reports to the central server.
- Monitor server CPU during aggregation -- the recommended server (32 vCPU) handles approximately 20 clients.

### 17.3 Storage Planning

| Component | Size | Growth Rate |
|-----------|------|-------------|
| Docker image | ~3.2 GB | Per rebuild |
| Results per task | ~1-5 MB | Per run |
| Full run results | ~50 MB | Per full run |
| Chest X-ray data | ~43 GB | Static |
| EBS root volume | 500 GB (server) / 1 TB (client) | Monitor monthly |

Set up disk usage alerts at 85% (Section 14.6). Docker images and logs are the primary consumers of disk space. Run `docker image prune -f` periodically to reclaim space from old images.

---

## 18. Version Management and Rollback

Production deployments need the ability to track which code version is running and quickly roll back to a previous version if a new release introduces problems. In a distributed system, version management is particularly critical because all nodes must run the same code version. A version mismatch between the server and clients can cause subtle bugs: model weight shapes may differ, strategy behavior may be inconsistent, or data pipeline processing may produce incompatible results.

This section covers image versioning, how to perform a rollback, and how to manage configuration versions.

### 18.1 Image Versioning

Tag images with semantic versions based on the build date and git commit hash. Never rely solely on the `latest` tag, which is ambiguous and makes it impossible to determine which code is running.

```bash
# Build with version tag
VERSION=$(date +%Y%m%d)-$(git rev-parse --short HEAD)
docker build -t ${FL_IMAGE}:${VERSION} -f Dockerfile .
docker tag ${FL_IMAGE}:${VERSION} ${FL_IMAGE}:latest

# Push both tags to registry
docker push ${REGISTRY}/${FL_IMAGE}:${VERSION}
docker push ${REGISTRY}/${FL_IMAGE}:latest
```

The `latest` tag is updated for convenience (deployment scripts default to it), but the version-specific tag is the authoritative identifier for audit and reproducibility.

### 18.2 Rollback Procedure

If a new image version causes training failures or produces incorrect results, roll back to the previous version across all nodes:

```bash
source cluster.env
ROLLBACK_VERSION=<previous-version-tag>

# Pull previous version on all nodes
for ip in ${SERVER_PUBLIC_IP} ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "docker pull ${REGISTRY}/${FL_IMAGE}:${ROLLBACK_VERSION} && \
     docker tag ${REGISTRY}/${FL_IMAGE}:${ROLLBACK_VERSION} ${FL_IMAGE}:latest"
done

# Restart orchestrator to pick up the rolled-back image
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker rm -f fl-orchestrator && docker rm -f fl-training"
# Re-launch orchestrator (Section 10.1)
```

This pulls the previous version from ECR, tags it as `latest` (so deployment scripts use it), and restarts the orchestrator to run with the rolled-back image.

### 18.3 Configuration Rollback

Keep `cluster.env` versions in a private repository. If a configuration change causes problems, revert to the previous version:

```bash
git log --oneline cluster.env
git checkout <commit> -- cluster.env
```

---

## 19. Incident Response Runbook

This section provides structured procedures for responding to incidents of different severity levels. Having these procedures documented and rehearsed before an incident occurs reduces response time and prevents ad-hoc decisions under pressure.

In a federated learning system, incident response is complicated by the multi-party nature of the deployment. A security incident may require notifying multiple data-owning organizations, each with their own incident response policies. A training failure may require coordinating with multiple sites to diagnose the root cause. Establishing clear communication channels and escalation procedures before going to production is essential.

### 19.1 Severity Levels

| Level | Definition | Response Time | Example |
|-------|-----------|---------------|---------|
| **P1** | Training halted, data at risk | 15 min | TLS compromise, unauthorized access |
| **P2** | Training degraded or failing | 1 hour | Client crash, GPU OOM, port conflict |
| **P3** | Non-blocking issue | 4 hours | Slow training, disk warning |

### 19.2 P1: Security Incident

A P1 security incident means there is evidence of unauthorized access to the cluster or compromise of cryptographic material. The priority is containment, then investigation, then remediation.

1. **Isolate:** Remove affected instances from security group. This immediately cuts network access without destroying evidence.
2. **Preserve:** Snapshot EBS volumes for forensics. Snapshots capture the disk state at the time of the incident.
3. **Rotate:** Regenerate all TLS certificates immediately. Even if you are unsure whether the TLS keys were compromised, rotating them eliminates the risk.
4. **Rotate:** Create new SSH key pair, update all nodes.
5. **Audit:** Review CloudTrail and Docker audit logs to determine the scope of the compromise.
6. **Notify:** Inform data owners per your organization's incident response policy. For healthcare data, this may trigger breach notification requirements.
7. **Remediate:** Patch the vulnerability, redeploy, and verify that the compromise vector is eliminated.

### 19.3 P2: Training Failure

#### Server container exits unexpectedly

```bash
source cluster.env

# Check exit code and logs
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker inspect fl-training --format='{{.State.ExitCode}}'"
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} \
  "docker logs fl-training --tail 50"

# Common causes:
# Exit 137 -> OOM killed. Increase --memory or reduce model size.
# Exit 1   -> Python exception. Check logs for traceback.
# Exit 0   -> Normal completion. Check if all strategies ran.
```

Exit code 137 is particularly common and means the Linux OOM killer terminated the process because it exceeded the container's memory limit. The fix is to increase the `--memory` flag or reduce model size / batch size.

#### Client can't connect

```bash
# Verify server is listening
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "ss -tlnp | grep 9092"

# Verify TLS cert SANs include server private IP
openssl x509 -in certs/server.pem -noout -text | grep -A1 "Subject Alternative Name"

# Verify network path
ssh -i ${KEY_PATH} ec2-user@<CLIENT_IP> "nc -zv ${SERVER_IP} 9092"
```

The most common cause of connection failures is a mismatch between the server's actual IP and the IP in the TLS certificate's SAN. If the server was replaced (new instance), the IP changed, and the certificate must be regenerated.

#### GPU errors

```bash
# CUDA device-side assert: DP noise corrupted model weights
# All models include prediction clamping (.clamp(1e-7, 1-1e-7)) to prevent this.
# If it recurs, check that the latest image is deployed.

# CUDA OOM: reduce batch size or MAX_SAMPLES
# L4 has 24GB -- sufficient for all current models

# nvidia-smi fails after long runs
sudo reboot  # Wait 60s
nvidia-smi --query-gpu=name --format=csv,noheader
```

The "CUDA device-side assert" error is specific to differential privacy strategies. When DP adds noise to model weights, it can push predictions outside the valid [0, 1] range, causing `log(0)` in the loss function. The platform handles this by clamping predictions to [1e-7, 1-1e-7], but older image versions may not have this fix.

### 19.4 P3: Performance Degradation

```bash
# Identify straggler clients
source cluster.env
for ip in ${CLIENT_IPS}; do
  echo "=== ${ip} ==="
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "sudo docker logs fl-client 2>&1 | grep 'Round' | tail -1"
done

# Check for thermal throttling
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} \
    "nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader"
done
```

In federated learning, the slowest client determines the round time. If one client is consistently slower than others (a "straggler"), it delays training for everyone. Common causes are: thermal throttling (GPU overheating), smaller instance type, or a larger local dataset.

---

## 20. Cost Management

AWS charges for EC2 instances whether they are training or idle. A GPU cluster of 6 instances costs approximately $222 per day. Stopping instances when not in use is the single most impactful cost management action.

### 20.1 Running Costs

| Configuration | Hourly | Daily (24h) | Monthly (730h) |
|--------------|--------|-------------|----------------|
| Full GPU cluster (1 server + 5 clients) | ~$9.25 | ~$222 | ~$6,753 |
| Stopped instances (EBS only) | ~$0.21 | ~$5 | ~$150 |

**EC2 charges whether training or not. Always stop instances when idle.**

The cost difference between running and stopped is approximately 45x. A cluster left running over a weekend wastes approximately $444.

### 20.2 Instance Lifecycle

Use the following commands to stop and start the cluster. Tag your instances consistently so these commands work reliably.

```bash
source cluster.env

# Stop all instances
aws ec2 describe-instances \
  --filters "Name=tag:Environment,Values=production" "Name=tag:Project,Values=fl-platform" \
  --query 'Reservations[].Instances[].InstanceId' --output text | \
  xargs aws ec2 stop-instances --instance-ids

# Start all instances
aws ec2 describe-instances \
  --filters "Name=tag:Environment,Values=production" "Name=tag:Project,Values=fl-platform" \
  --query 'Reservations[].Instances[].InstanceId' --output text | \
  xargs aws ec2 start-instances --instance-ids

# Note: Public IPs change on restart unless Elastic IPs are assigned.
# Update cluster.env after restart.
```

**Important:** When instances are stopped and restarted, they receive new public IP addresses (unless Elastic IPs are assigned). Private IPs within a VPC typically remain the same, but verify and update `cluster.env` after every restart.

### 20.3 Cost Optimization

Several strategies can reduce costs significantly:

- **Spot Instances** for clients: ~70% savings. Spot instances can be reclaimed by AWS with 2 minutes notice, but FL is naturally fault-tolerant -- clients can rejoin after the next round starts. Do not use spot instances for the server, as losing the server halts all training.
- **Reserved Instances** for predictable workloads: ~40% savings if you commit to 1-year usage.
- **Schedule heavy tasks** (DenseNet) during off-peak hours when spot prices are lower.
- **Right-size instances:** use `t3.xlarge` (CPU-only) clients for small models (MLP, LogReg) that do not benefit from GPU acceleration.

### 20.4 Time Estimates

Use this table to estimate how long a training run will take and its approximate cost:

| Scope | Tasks | Est. Time | Est. Cost (full GPU) |
|-------|-------|-----------|---------------------|
| Smoke test | fraud | 5 min | < $1 |
| Light tasks | fraud + sepsis + ecg | 3 hours | ~$28 |
| All except chest | 12 tasks | 5 hours | ~$46 |
| Full run | All 13 tasks | 20 hours | ~$185 |
| Chest X-ray only | chest | 14 hours | ~$130 |

The chest X-ray task dominates total cost because DenseNet-121 is the largest model and image data is the most compute-intensive to process. If chest X-ray is not needed for your deployment, excluding it cuts total run time and cost by approximately 70%.

### 20.5 Cost Comparison with Centralized ML

A common question is whether federated learning is more expensive than centralized machine learning. The answer depends on the scenario:

**Federated learning is more expensive in compute terms.** N clients each train the full model, whereas centralized training runs once on the combined dataset. For N=5 clients, FL uses approximately 5x the GPU hours of centralized training. However, this comparison is misleading because centralized training is often not an option -- the data cannot be centralized.

**Federated learning can be cheaper in total cost.** When you factor in the legal, regulatory, and organizational costs of data centralization -- data sharing agreements, privacy impact assessments, data transfer infrastructure, breach liability insurance, compliance audits -- federated learning is often cheaper even with higher compute costs. A single data breach can cost millions; the incremental compute cost of FL is typically thousands.

**Federated learning has lower ongoing data costs.** Each organization stores and manages its own data using its existing infrastructure. There is no central data lake to provision, secure, and maintain.

---

## 21. Troubleshooting

This section catalogs known issues, their symptoms, and resolutions. If you encounter a problem not listed here, check the server and client logs (Section 14.3) for error messages.

### 21.1 Common Issues

The following table covers the most frequently encountered issues. Each row describes a specific error, its root cause, and how to fix it.

| Issue | Symptom | Resolution |
|-------|---------|------------|
| **Port in use** | `Port 0.0.0.0:9092 is already in use` | A SuperLink or previous `fl-training` is still running. Run `docker rm -f fl-superlink fl-training` before starting. The orchestrator does this automatically. |
| **Client can't connect** | `Connection refused` or TLS errors | Check server is running on 9092, certs match, private IP correct in SAN |
| **CUDA device-side assert** | `CUDA error: device-side assert triggered` | DP noise corrupts model weights, causing `BCELoss(log(0))`. All models clamp predictions `.clamp(1e-7, 1-1e-7)`. Verify latest image is deployed. |
| **Server hangs** | Server stuck waiting for clients | Clients hit CUDA errors and exited. Orchestrator enforces per-task timeouts. |
| **GPU not found** | `nvidia-smi: not found` | Install driver: `sudo dnf install -y nvidia-driver` then `sudo modprobe nvidia` |
| **GPU not in Docker** | `CUDA not available` inside container | Install nvidia-container-toolkit, restart Docker (Section 6.4) |
| **Partition KeyError** | `KeyError: 0` in client logs | Extreme non-IID can leave partitions empty. Client code falls back to nearest partition. |
| **Docker needs sudo** | `permission denied` on client | Add user to docker group: `sudo usermod -aG docker ec2-user` |
| **Image not found** | `Unable to find image` | Pull from ECR or distribute from server (Section 7.2-7.3) |
| **TLS handshake fail** | `WRONG_VERSION_NUMBER` | Ensure server cert SAN includes the private IP. Regenerate certs (Section 9.3). |
| **CUDA OOM** | `RuntimeError: CUDA out of memory` | Reduce batch size or `MAX_SAMPLES`. L4 has 24GB -- sufficient for all current models. |

### 21.2 Client Reconnection

`runners/run_client.py` has built-in reconnection logic designed to handle the multi-strategy workflow. When one strategy completes, the server disconnects all clients and starts the next strategy. Clients must reconnect for each new strategy.

The reconnection behavior:

- After each strategy completes, the client waits 2 seconds then attempts to reconnect.
- On connection failure, retries every 5 seconds.
- After 12 consecutive failures (~60 seconds), assumes the server has finished all strategies and exits.

To restart a failed client manually:

```bash
source cluster.env

ssh -i ${KEY_PATH} ec2-user@<CLIENT_IP> "
  sudo docker rm -f fl-client
  sudo docker run -d --name fl-client --network host --gpus all \
    --restart on-failure:3 \
    --memory 56g --cpus 14 \
    --log-opt max-size=100m --log-opt max-file=5 \
    -v ~/fl-deploy/certs/ca.pem:/certs/ca.pem:ro \
    -v ~/fl-deploy/data:/data:ro \
    -e PARTITION_ID=<N> -e NUM_CLIENTS=${NUM_CLIENTS} -e FL_TASK=<TASK> \
    -e FL_SERVER=${SERVER_IP}:9092 -e CERTS_DIR=/certs \
    ${FL_IMAGE}:${FL_IMAGE_TAG} python3 runners/run_client.py
"
```

Replace `<N>` with the client's partition ID and `<TASK>` with the current task name. The client will connect to the server and join the current round.

### 21.3 GPU Driver Recovery

The NVIDIA driver can occasionally become unstable after long training runs (many hours). Symptoms include `nvidia-smi` hanging or returning errors.

```bash
# If nvidia-smi fails after long Docker runs
sudo reboot
# Wait 60s, then verify:
nvidia-smi --query-gpu=name --format=csv,noheader

# If module won't load after reboot
sudo modprobe nvidia
dmesg | tail -20  # check for errors
```

A reboot resolves the issue in virtually all cases. The NVIDIA kernel module reloads on boot and reinitializes the GPU hardware.

### 21.4 Rebuild and Redeploy

After code changes, you need to rebuild the Docker image and distribute it to all nodes. This is the standard development cycle for iterating on the platform.

```bash
source cluster.env

# Build on server
cd ~/fl-build
docker build -t ${FL_IMAGE}:${FL_IMAGE_TAG} .

# Distribute to all clients via VPC internal network
docker save ${FL_IMAGE}:${FL_IMAGE_TAG} | gzip > /tmp/fl-image.tar.gz
for ip in ${CLIENT_IPS}; do
  (scp -i ~/.ssh/deploy.pem /tmp/fl-image.tar.gz ec2-user@${ip}:/tmp/ && \
   ssh -i ~/.ssh/deploy.pem ec2-user@${ip} "sudo docker load < /tmp/fl-image.tar.gz && rm /tmp/fl-image.tar.gz") &
done
wait
rm /tmp/fl-image.tar.gz
```

### 21.5 Cleanup

When you are done with a training run, clean up containers and reclaim disk space:

```bash
source cluster.env

# Stop all FL containers across cluster
ssh -i ${KEY_PATH} ec2-user@${SERVER_PUBLIC_IP} "docker rm -f fl-superlink fl-training fl-orchestrator"
for ip in ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "sudo docker rm -f fl-supernode fl-client"
done

# Prune old images (reclaim disk)
for ip in ${SERVER_PUBLIC_IP} ${CLIENT_IPS}; do
  ssh -i ${KEY_PATH} ec2-user@${ip} "docker image prune -f"
done
```

The `docker image prune -f` command removes all dangling images (images not tagged and not used by any container). This can reclaim several GB per node, especially after multiple image rebuilds.

### 21.6 Verifying a Successful Training Run

After a training run completes, verify that results are valid before considering the run successful:

1. **Check result files exist.** Each task produces a JSON result file in the results directory. Verify that the expected number of files exists.
2. **Check strategy counts.** Each result file should contain entries for all expected strategies. For a task with 11 strategies, the result file should have 11 entries.
3. **Check metric values.** Accuracy, loss, or AUC values should be within reasonable ranges. An accuracy of 50% on a binary classification task suggests the model learned nothing (random guessing). An accuracy above 99% on medical data is suspicious and may indicate data leakage.
4. **Compare across strategies.** IID strategy should produce the highest accuracy. Non-IID strategies should show some degradation. DP strategies should show further degradation proportional to the noise level (lower epsilon = more degradation). If DP results are better than non-DP results, something is wrong.
5. **Run privacy tests.** Execute the privacy attack suite to verify that the model does not excessively leak training data.

```bash
# Quick validation: check result files and strategy counts
ls -la ~/fl-deploy/results/*.json | wc -l   # Expected: one per task
cat ~/fl-deploy/results/fraud_results.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Strategies: {len(d)}')"
```

---

## Appendices

### Appendix A: Privacy-Enhancing Technologies

For comprehensive PET coverage including DP variants, formal guarantees, SecAgg, TEE platforms, and decision matrices, see **[PET_Reference.md](PET_Reference.md)**.

That document covers:

- **PET Taxonomy:** The full classification of privacy-enhancing technologies organized by what they protect -- input privacy (federated learning, synthetic data, k-anonymity), computation privacy (differential privacy, secure aggregation, homomorphic encryption, secure multi-party computation, trusted execution environments), output privacy (DP on outputs, model watermarking, federated LoRA), and verification (privacy attacks, privacy auditing).
- **Differential Privacy Variants:** Detailed descriptions of Central DP, Local DP, Distributed DP, Renyi DP, Zero-Concentrated DP, User-level DP, Record-level DP, Shuffle DP, and Label DP, including when to use each variant.
- **DP Mechanisms:** Laplace, Gaussian, Discrete Gaussian, Exponential, Randomized Response, and Skellam mechanisms, with the mathematical formulations and use cases for each.
- **State-of-the-Art DP for FL:** DP-SGD, DP-FTRL, DP-FedAvg, DP-FedSGD, adaptive clipping, private selection, Poisson subsampling, and PATE, including the epsilon values achieved in practice.
- **Industrial Implementations:** How Google, Apple, Microsoft, Meta, OpenDP, Tumult Labs, and LinkedIn implement differential privacy in production systems.
- **Decision Matrices:** How to choose the right PET combination based on your threat model, regulatory requirements, and performance constraints.

### Appendix B: File Reference

The following table provides a complete inventory of every file in the platform, organized by function. Use this as a quick reference when you need to find the implementation of a specific feature.

| File | Purpose |
|------|---------|
| **Core** | |
| `runners/run_ec2.py` | Server-side experiment runner. `--distributed` enables `start_server()` |
| `runners/run_client.py` | Client-side runner with mTLS, pre-flight data check, reconnect loop |
| `tools/ingest.py` | Client-side data ingestion CLI (validation, manifest, checksums) |
| `Dockerfile` | Unified Docker image (all deps pinned to exact versions) |
| **FL Framework** | |
| `fl_common/strategies.py` | All FL strategy implementations (FedAvg, SCAFFOLD, DP, SecAgg, etc.) |
| `fl_common/federated_adapter.py` | Generic federated LoRA framework for any HuggingFace model |
| `fl_common/data.py` | DataConfig, DataManifest, validation gates, partition utilities |
| `fl_common/dp.py` | Differential privacy primitives + RDP accountant |
| `fl_common/secagg.py` | Secure aggregation (pairwise masking) |
| **Models** | |
| `models/olmo/` | OLMo-1B federated LoRA (uses federated_adapter.py) |
| `models/bilstm/` | BiLSTM (sepsis, ECG). CPU fallback for DP strategies |
| `models/mlp/`, `models/densenet/`, etc. | Task-specific models with NaN sanitization |
| `models/*/server_app.py` | Strategy factory per model type |
| `models/*/client_app.py` | NumPyClient implementation per model type |
| **Tasks** | |
| `tasks/*/data.py` | Data pipeline per task (load, validate, clean, normalize, partition) |
| `tasks/gov_doc/data.py` | Government document data (4 domains: healthcare, finance, urban, research) |
| **Security** | |
| `secure_inference/tenseal_inference.py` | CKKS homomorphic encryption inference (MLP, BiLSTM, DenseNet) |
| `privacy/test_privacy.py` | MIA, DLG, canary extraction attacks |
| **Deploy** | |
| `deploy/distributed/deploy.sh` | Main deploy script (build, distribute, health, run, down) |
| `deploy/cluster.env.template` | Cluster configuration template |
| `deploy/validate_config.sh` | Configuration validation |
| `deploy/health_check.sh` | 38-point cluster health check (--quick, --json) |
| `deploy/gen_mtls_certs.sh` | mTLS certificate generation (CA + server + per-client) |
| `deploy/rotate_certs.sh` | Certificate rotation (check, generate, distribute, verify) |
| `deploy/backup.sh` | Backup results, certs, config, manifests (local + S3) |
| `deploy/rollback.sh` | Image version listing and rollback |
| `run_server_side.sh` | Server-side orchestrator (runs inside `docker:cli` container) |
| `scenarios/*.yaml` | Predefined experiment configurations |

### Appendix C: Adding a New Site (Client)

Adding a new site is one of the most common operational tasks. When a new organization joins the federated learning collaboration, they need a provisioned node, the Docker image, the CA certificate, and their local data ingested. The existing cluster requires minimal changes -- only the `cluster.env` file needs to be updated.

Follow these steps to add their node to the cluster:

1. Launch a new instance in the same VPC (matching instance type). Ensure it is in the client subnet with the client security group.
2. Install GPU driver + Docker + nvidia-container-toolkit (Section 6.3-6.4)
3. Copy CA cert: `scp certs/ca.pem ec2-user@<NEW_IP>:~/fl-deploy/certs/`
4. Ingest local data: `python tools/ingest.py --task <TASK> --input <DATA_PATH> --client-id <SITE_ID>`
5. Pull Docker image from ECR or load from tarball
6. Update `cluster.env`: add IP to `CLIENT_IPS`, increment `NUM_CLIENTS`
7. Update `PARTITION_ID` range (0 to N-1). Existing clients keep their original partition IDs; the new client gets the next ID.

The new client will participate in the next training run. No changes to the server or existing clients are needed, apart from updating the `NUM_CLIENTS` count so that data partitioning accounts for the new participant.

**Impact on existing results:** Adding a new client changes the data partitioning for all clients (because the data is now split N+1 ways instead of N ways). If you are using real site data (not partitioned from a central dataset), this has no impact -- each site simply uses its own data. If you are using synthetic data or partitioning a test dataset, previous results are not directly comparable to results with the new client count. Re-run the full suite for an apples-to-apples comparison.

**Network considerations:** If the new site is in a different VPC or region, you will need VPC peering or a VPN tunnel to enable the gRPC connection to the server. Cross-region communication adds latency (typically 20-200ms per round trip depending on distance), which slows down the FL round time but does not affect model quality.

### Appendix D: Adding a New Task

Adding a new task extends the platform to handle a new type of prediction on a new dataset. This requires implementing both the model (the neural network architecture) and the data pipeline (loading, validating, and preprocessing the data). The platform is designed to make this as modular as possible -- each task is self-contained in its own directory, and the existing infrastructure (FL strategies, distributed deployment, monitoring) works automatically with new tasks.

To add a new machine learning task, follow these steps:

1. **Create model:** `models/<name>/server_app.py` (strategy configuration) + `client_app.py` (NumPyClient implementation). Use an existing model directory as a template.
2. **Create data pipeline:** `tasks/<name>/data.py` (data loading, validation, cleaning, normalization, partitioning). The `data.py` module must implement a standard interface that `runners/run_client.py` calls.
3. **Add to `runners/run_ec2.py`:** new `run_<name>()` function + entry in `task_map`. This registers the task so the server knows how to run it.
4. **Add to `runners/run_client.py`:** new case in `make_client()`. This tells the client which model and data pipeline to use for the new task.
5. **Rebuild Docker image** and distribute (Section 21.4). The new model and data pipeline code must be baked into the Docker image so all nodes have it.
6. **Add scenario YAML** in `scenarios/` (optional, for predefined experiment configurations). Scenarios define which strategies to run, how many rounds, and what hyperparameters to use.
7. **Run single-task validation:** orchestrator target `<name>` to verify the new task works in distributed mode. Start with synthetic data (set `SYNTHETIC=1`) to verify the pipeline before introducing real data.
8. **Test with non-IID data.** After verifying with IID data, test with `Alpha_0.5` and `Alpha_0.1` to ensure the model handles heterogeneous data. Many models that work well on IID data diverge on non-IID data, requiring FedProx or SCAFFOLD strategies.
9. **Run privacy tests.** If the task processes sensitive data, run the MIA and gradient inversion tests to verify that privacy protections are adequate.

### Appendix E: Glossary

This glossary defines technical terms used throughout this document. Terms are listed in the order they are most likely to be encountered.

| Term | Definition |
|------|-----------|
| **Federated Learning (FL)** | A machine learning approach where multiple organizations collaboratively train a model without sharing their raw data. Each site trains on its own data and shares only model updates. |
| **Model** | A mathematical function with adjustable parameters that maps inputs (e.g., patient vital signs) to predictions (e.g., probability of sepsis). |
| **Parameters / Weights** | The adjustable numerical values inside a model that determine its predictions. Training adjusts these values to improve accuracy. |
| **Gradient** | The mathematical derivative of the model's error with respect to its parameters. Gradients indicate how to adjust parameters to reduce prediction errors. In FL, gradients or weight updates are what clients send to the server. |
| **Round** | One cycle of the FL loop: server distributes model, clients train locally, clients send updates, server aggregates. A typical training run has 3-20 rounds. |
| **Aggregation** | The process of combining model updates from multiple clients into a single improved model. FedAvg (Federated Averaging) computes a weighted average of all updates. |
| **Strategy** | A specific FL algorithm that determines how updates are aggregated and what privacy protections are applied (e.g., FedAvg, FedProx, SCAFFOLD, SecAgg, DP). |
| **IID (Independent and Identically Distributed)** | A statistical property meaning all clients' data comes from the same distribution. In practice, this is rarely true -- different hospitals serve different patient populations. |
| **Non-IID** | Data distributions that differ across clients. The "alpha" parameter controls severity: alpha=0.5 is moderate, alpha=0.1 is extreme. |
| **Differential Privacy (DP)** | A mathematical framework that guarantees that the output of a computation does not reveal whether any specific individual's data was included in the input. Achieved by adding calibrated random noise. |
| **Epsilon** | The privacy parameter in differential privacy. Lower epsilon = stronger privacy = more noise = less model accuracy. |
| **Secure Aggregation (SecAgg)** | A cryptographic protocol that allows the server to compute the sum of all client updates without seeing any individual update. Uses pairwise random masks that cancel in the aggregate. |
| **Homomorphic Encryption (HE)** | Encryption that allows mathematical operations (addition, multiplication) to be performed on encrypted data without decrypting it. The result, when decrypted, matches the result of performing the operations on the plaintext. |
| **CKKS** | A specific homomorphic encryption scheme (Cheon-Kim-Kim-Song) designed for approximate arithmetic on real numbers. Well-suited for machine learning inference. |
| **TLS (Transport Layer Security)** | A cryptographic protocol that encrypts network communication, preventing eavesdropping. The successor to SSL. |
| **mTLS (Mutual TLS)** | TLS where both sides (client and server) present certificates, providing mutual authentication. Standard TLS only authenticates the server. |
| **Certificate Authority (CA)** | An entity that issues digital certificates. The CA's certificate is the root of trust -- all certificates signed by the CA are trusted by nodes that have the CA certificate. |
| **gRPC** | A high-performance Remote Procedure Call framework built on HTTP/2, used for communication between FL clients and the server. Supports bidirectional streaming and native TLS. |
| **LoRA (Low-Rank Adaptation)** | A technique for fine-tuning large models by adding small trainable matrices (adapters) to frozen model layers. Reduces trainable parameters by 99%+. |
| **QLoRA** | LoRA combined with 4-bit quantization of the base model, further reducing memory requirements. |
| **MIA (Membership Inference Attack)** | An attack that determines whether a specific data record was used to train a model, by exploiting the model's tendency to produce lower errors on training data. |
| **DLG (Deep Leakage from Gradients)** | An attack that reconstructs training data from observed model gradients by iteratively optimizing a dummy input to match the observed gradients. |
| **Orchestrator** | The automated deployment script that manages the entire FL training lifecycle: starting/stopping containers, running tasks in sequence, handling timeouts. |
| **Partition** | A subset of the training data assigned to a specific client. In IID mode, partitions are random slices. In non-IID mode, partitions have skewed label distributions. |
| **VFL (Vertical Federated Learning)** | FL where each site holds different features for the same entities. Each site trains the part of the model that processes its features. |
| **Split Learning** | A variant of FL where the model is physically split between client and server. The client runs bottom layers on raw data, sends intermediate representations, and the server runs top layers. |
| **Transfer Learning** | Using a model pre-trained on one task as the starting point for training on a different task. In FL, this allows leveraging large pre-trained models without centralizing the fine-tuning data. |

### Appendix F: Troubleshooting Decision Tree

When something goes wrong, use this decision tree to quickly narrow down the root cause:

1. **Is the server running?** Check: `docker ps | grep fl-training` on the server.
   - No: Check server logs with `docker logs fl-training --tail 50`. Go to Section 19.3.
   - Yes: Continue to step 2.

2. **Is the server listening on port 9092?** Check: `ss -tlnp | grep 9092` on the server.
   - No: Another process may hold the port. Check for SuperLink containers. Go to Section 3.5.
   - Yes: Continue to step 3.

3. **Can clients reach the server?** Check: `nc -zv ${SERVER_IP} 9092` from a client.
   - No: Check security groups (Section 15.2) and VPC routing.
   - Yes: Continue to step 4.

4. **Do clients have the correct CA certificate?** Check: `openssl verify -CAfile /certs/ca.pem /certs/ca.pem` inside the client container.
   - No: Redistribute certificates (Section 9.3).
   - Yes: Continue to step 5.

5. **Does the server certificate have the correct SAN?** Check: `openssl x509 -in certs/server.pem -noout -text | grep "Subject Alternative Name" -A1`.
   - Server private IP not listed: Regenerate server certificate (Section 9.3).
   - IP is listed: Check for CUDA/GPU errors in client logs.

### Appendix G: Quick Start Checklist

For experienced operators who want to get the cluster running quickly without reading the full guide, here is the minimum sequence of steps. Each step references the detailed section where it is fully explained. This checklist assumes you already understand federated learning concepts (Section 1) and the privacy trade-offs (Section 2).

1. **Provision infrastructure** (Section 6): Launch 1 server + N client EC2 instances in the same VPC.
2. **Install GPU drivers** (Section 6.3): Run on all GPU instances.
3. **Install Docker + Container Toolkit** (Section 6.4): Run on all instances.
4. **Build Docker image** (Section 7.1): Build on the server.
5. **Distribute Docker image** (Section 7.2 or 7.3): Push to ECR or SCP to clients.
6. **Generate TLS certificates** (Section 9.2-9.3): Run `gen_mtls_certs.sh --full`.
7. **Distribute certificates** (Section 9.3): CA cert to all nodes, server cert/key to server only.
8. **Create cluster.env** (Section 5.1): Fill in all IP addresses and paths.
9. **Set up server-to-client SSH** (Section 10.3): Copy SSH key to server, verify connectivity.
10. **Ingest data on each client** (Section 8.2): Run `tools/ingest.py` on each client node.
11. **Launch orchestrator** (Section 10.1): Start the orchestrator container on the server.
12. **Monitor progress** (Section 14.3): Watch server logs for round completion.
13. **Collect results** (Section 16.2): Back up results to S3 after completion.
14. **Stop instances** (Section 20.2): Stop all EC2 instances to avoid ongoing charges.

**Estimated time for first deployment:** 4-6 hours for infrastructure setup (steps 1-9), plus time for data ingestion. Subsequent training runs take minutes to hours depending on the tasks selected (see Section 20.4 for time estimates).

**Estimated time for repeat deployments:** 15-30 minutes if infrastructure is already provisioned and Docker images are cached on all nodes.

---

### Appendix H: Document Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-05-15 | Initial reference manual. Architecture, deployment, troubleshooting. |
| 2.0 | 2026-05-22 | Added tasks, strategies, security sections. Added monitoring and backup procedures. |
| 2.1 | 2026-05-28 | Updated task matrix with latest test results. Added cost management section. Added incident response runbook. |
| 3.0 | 2026-06-02 | Complete rewrite as explanatory document. Added: How FL Works (Section 1), Privacy Guarantees (Section 2), Federated Adapter Framework (Section 12), Secure Inference (Section 13), strategy selection guide (Section 11.3), privacy attack testing (Section 11.4), glossary (Appendix E), troubleshooting decision tree (Appendix F), quick start checklist (Appendix G). All technical content from previous versions preserved. Comprehensive explanatory context added to all existing sections. |

---

*End of document.*
