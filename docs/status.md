# FL + PET Sandbox — Progress Tracker

**Last updated:** 2026-05-16

---

## Quick Resume

**Where we left off:** All distributed experiments complete. 5 new tasks (anomaly, mortality, drug, readmission, satellite) validated. Generic config-driven pipeline added. Secure inference demos done. Pushed to GitHub (govtech-data-practice/Fl_deployment). EC2 cluster idle — consider stopping to save cost ($222/day).

**Next actions (June sprint — see ROADMAP.md):**
1. **v0.2 (Week 1):** Implement Krum/TrimmedMean robust aggregation + poisoning attacks + audit logger
2. **v0.3 (Week 2):** TEE enclave refactor — move FL server into Nitro Enclave with attestation
3. **v0.4 (Week 3):** Paillier HE aggregation (production), DP-FTRL, SecAgg+
4. **v0.5 (Week 4):** Air-gap deployment, RBAC, new models (U-Net, YOLO, GNN, BERT)
5. Fix anomaly task metric (uses AUC but summary checks accuracy — cosmetic)
6. Fix satellite SCAFFOLD extreme non-IID (0.0 acc — needs investigation)

**Quick start for next session:**
```bash
# Check EC2 status
ssh -i TEE_FL.pem ec2-user@54.151.221.104 'docker ps; nvidia-smi'

# If instances were stopped, start them:
aws ec2 start-instances --instance-ids i-0e2e6a7d9c376fa66 i-0bca5de6c91793de8 i-0d41f4b6ca24092e3 i-00252d69b00cadc41 i-07f9ef697158f6947 i-0175d0516905275a6

# Run experiments
./run_distributed.sh all          # full suite
./run_distributed.sh new          # just new 5 tasks
./run_distributed.sh fraud        # single task

# Sync to GitHub
./github/sync.sh && cd github && git add -A && git commit -m "..." && git push
```

---

## Infrastructure Status

| Resource | Status | Notes |
|----------|--------|-------|
| Local repo | OK | `master` branch, single commit `ed13b8f` |
| Remote `ec2/master` | In sync | Same commit as local |
| AWS credentials | EXPIRED | Key ending `A66H`, needs refresh |
| **FL_deployment** (server) | RUNNING | i-0e2e6a7d9c376fa66, g6.8xlarge, 54.151.221.104, ap-southeast-1c |
| **Client_test_1** | RUNNING | i-0bca5de6c91793de8, g6.4xlarge, 47.130.0.207, ap-southeast-1a |
| **Client_test_2** | RUNNING | i-0d41f4b6ca24092e3, g6.4xlarge, 47.129.54.224, ap-southeast-1a |
| **Client_test_3** | RUNNING | i-00252d69b00cadc41, g6.4xlarge, 52.221.246.101, ap-southeast-1a |
| **Client_test_4** | RUNNING | i-07f9ef697158f6947, g6.4xlarge, 175.41.152.74, ap-southeast-1a |
| **Client_test_5** | RUNNING | i-0175d0516905275a6, g6.4xlarge, 3.0.16.188, ap-southeast-1a |
| All EC2s | 3/3 checks passed | Launched 2026-04-30, AMI `TEE_FL`, key pair `TEE_FL`, SG `launch-wizard-16` |
| Nitro Enclave | Last used 2025-12-10 | 21/21 chest experiments passed |

---

## Experiment Runs Completed

| # | Run | Date | Tasks | Result |
|---|-----|------|-------|--------|
| 1 | Nitro Enclave (chest X-ray hyperparams) | 2025-12-10 | 21 experiments | 21/21 SUCCESS |
| 2 | Large benchmark (chest X-ray strategies) | 2026-01-15 to 01-20 | 38 experiments | 38/38 SUCCESS |
| 3 | PEFT/LoRA (Mistral 7B) | Validated | FL + MIA + canary | MIA 1.0->0.83 w/ DP |

**Local result files:**
- `full_log_20251210_182048.txt` — Nitro Enclave full console log (26K lines)
- `experiment_results.csv` — 38 experiment results (Jan 2026 run)
- `metrics_20251210_182048.csv` — Per-round metrics (1623 rounds)
- `results/` — Empty (results from EC2 runs not synced locally)

---

## Phase Progress

### Phase 1: Integrity + Robustness — NOT STARTED

| Task | Status | Files |
|------|--------|-------|
| Krum aggregation | TODO | `fl_common/strategies.py` |
| Multi-Krum | TODO | `fl_common/strategies.py` |
| Trimmed Mean | TODO | `fl_common/strategies.py` |
| Bulyan | TODO | `fl_common/strategies.py` |
| FLTrust | TODO | `fl_common/strategies.py` |
| Label flip attack test | TODO | `privacy/test_poisoning.py` |
| Gradient scaling attack test | TODO | `privacy/test_poisoning.py` |
| Backdoor attack test | TODO | `privacy/test_poisoning.py` |
| Add to runners/run_ec2.py | TODO | `runners/run_ec2.py` |
| Poisoning scenario YAMLs | TODO | `scenarios/` |

### Phase 2: Audit + Provenance — NOT STARTED

| Task | Status | Files |
|------|--------|-------|
| AuditLogger class | TODO | `fl_common/audit.py` |
| Integrate into MetricCapture | TODO | `runners/run_ec2.py` |
| Model provenance JSON | TODO | `results/provenance_*.json` |
| Privacy budget ledger | TODO | `results/privacy_ledger.json` |
| HMAC signing for audit entries | TODO | `fl_common/audit.py` |

### Phase 3: TEE as PET — NOT STARTED

| Task | Status | Files |
|------|--------|-------|
| Refactor SuperLink into enclave | TODO | `deploy/` |
| Attestation verification lib | TODO | `fl_common/attestation.py` |
| Client-side PCR validation | TODO | Client code |
| CI: build EIF + record PCRs | TODO | CI config |

### Phase 4: Additional PETs — NOT STARTED

| Task | Status | Files |
|------|--------|-------|
| Federated Analytics module | TODO | `fl_analytics/` |
| DP-CTGAN synthetic data | TODO | TBD |
| Paillier HE FedAvg | TODO | `fl_common/he.py` |
| SMPC (Shamir secret sharing) | TODO | `fl_common/smpc.py` |

### Phase 5: Deployment Hardening — NOT STARTED

| Task | Status | Files |
|------|--------|-------|
| Air-gap bundle script | TODO | `deploy/offline/` |
| RBAC role definitions | TODO | TBD |
| Data sovereignty policy engine | TODO | TBD |
| Multi-classification support | TODO | TBD |

### Phase 6: Testing + Validation — NOT STARTED

| Task | Status | Files |
|------|--------|-------|
| Robust strategy test harness | TODO | `tests/` |
| Poisoning attack scenarios | TODO | `scenarios/` |
| TEE attestation e2e test | TODO | `tests/` |
| Audit log integrity test | TODO | `tests/` |
| Air-gap deployment test | TODO | `tests/` |

---

## Code Changes (Uncommitted)

| File | Change | Category |
|------|--------|----------|
| `privacy/test_privacy.py` | Modified (staged) | Privacy tests |
| `runners/run_ec2.py` | Modified (unstaged) | EC2 runner |
| `tasks/ecg/data.py` | Modified (unstaged) | ECG data loader |

**Untracked (not in git):**
- `.idea/` — IDE config (gitignore)
- `ca.crt`, `client.key` — TLS certs (DO NOT COMMIT)
- `eicu/`, `eicu_unzipped/` — Patient data (DO NOT COMMIT)
- `medicaldata/` — Medical datasets (DO NOT COMMIT)
- `experiment_results.csv`, `metrics_*.csv`, `full_log_*.txt` — Run outputs

---

## Decisions Made

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-15 | Created govt readiness plan (6 phases) | Gap analysis: audit, byzantine robustness, TEE-as-PET are critical |
| 2026-05-15 | Phase 1+2 prioritised | Integrity and audit are non-negotiable; lowest complexity |
| 2026-05-15 | DP not recommended for large models | Validated: DenseNet AUC=0.50 with DP. Use SecAgg instead |
| 2026-05-15 | FedAdam/FedYogi dropped from recommendations | Diverge on pretrained models. FedAvg+SCAFFOLD preferred |

---

## Session Log

| Date | Session | What happened |
|------|---------|--------------|
| 2026-05-15 | Initial review | Reviewed full repo, experiment logs, deployment guide. Identified 6 gaps for govt. Created `plan.md` and `status.md`. Saved offline memory. |
| 2026-05-15 | Expanded model/data coverage | Updated `plan.md` with 30 target models, 14 data types, 12 govt domains. Prioritised into Phase A (swap), B (new data types), C (new FL paradigms). |
| 2026-05-15 | EC2 status confirmed | All 6 instances running (1 server g6.8xlarge + 5 clients g6.4xlarge), 3/3 checks passed, launched 2026-04-30. Clients had no Elastic IPs. |
| 2026-05-15 | Distributed infra setup | Installed NVIDIA driver 595.71.05 + nvidia-container-toolkit on all 5 clients. GPU (L4) now working in Docker containers. |
| 2026-05-15 | Created distributed runner | `run_distributed.sh` orchestrates server (start_server) + clients (runners/run_client.py with reconnect loop). Fixed partition-0 KeyError in MLP client. |
| 2026-05-15 | Fraud smoke test PASSED | 11/11 strategies passed in distributed mode (5 clients, L4 GPU, TLS). |
| 2026-05-15 | Full distributed run launched | All 8 tasks running: fraud, sepsis, ecg, chest, vfl, split, transfer, privacy. PID 79871, log at `distributed_run.log`. |
| 2026-05-15 | Deployment guide created | `docs/Distributed_Deployment_Guide.md` — covers infra, GPU setup, Docker, TLS, orchestration, monitoring, troubleshooting, cost management, security checklist. |
| 2026-05-15 | PET reference created | `docs/PET_Reference.md` — DP variants + industrial standards, SecAgg (pairwise/Paillier/SMPC), TEE platforms, decision matrices, formal guarantees, post-quantum status, deployment checklists. |
| 2026-05-15 | Distributed run progress | Fraud (11/11), Sepsis (done), ECG (done). First pass timing issue, second pass working correctly. |
| 2026-05-15 | New tasks + models added | 5 new tasks (anomaly, mortality, drug, satellite, readmission) + 5 new models (autoencoder, logreg, cnn1d, tabnet, resnet-small) + secure inference demos. Fixed input_dim mismatches. |
| 2026-05-15 | Generic config layer | `models/generic/` + `tasks/generic/` — config-driven FL via env vars, no code changes for new datasets. GENERIC_DATA_MODULE for custom data loaders. |
| 2026-05-15 | New tasks distributed results | anomaly: 11 run (AUC 0.93, exit 1 metric mismatch), mortality: 11/11 PASS (0.876), drug: done, readmission: 11/11 PASS, satellite: 6/7 PASS (1 SCAFFOLD extreme=0.0) |
| 2026-05-15 | All distributed runs complete | Cluster idle. 15 result files on server. Roadmap compressed to June 2026 completion. |
| 2026-05-15 | GitHub pushed | govtech-data-practice/Fl_deployment — 130 files, ROADMAP.md, no secrets. 4 commits. |
| 2026-05-16 | Production technical reference | Created `docs/FL_Production_Technical_Reference.md` — brutally honest assessment of architecture, security, compliance, failure modes, monitoring gaps, and tested-vs-claimed status for every feature. |
