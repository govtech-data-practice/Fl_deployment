#!/usr/bin/env python3
"""
Test 3 new data domains: ECG, Financial Fraud, Government LLM
Each uses the existing FL pipeline with minimal adaptation.
"""
import sys, os, time, logging, random, math, json
import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s")
logger = logging.getLogger("domains")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from collections import OrderedDict

# ======================================================================
# DOMAIN 1: ECG (PTB-XL style — synthetic for now, same BiLSTM arch)
# ======================================================================

def test_ecg():
    """Test FL on ECG time series using BiLSTM (same arch as sepsis)."""
    logger.info("=" * 60)
    logger.info("DOMAIN 1: ECG Time Series (BiLSTM)")
    logger.info("=" * 60)

    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.simulation import run_simulation
    from flwr.clientapp import ClientApp
    from flwr.common import Context
    from torch.utils.data import DataLoader, Dataset

    # Synthetic ECG: 12 leads, 250 timesteps (1 sec at 250Hz), 5 classes
    class SyntheticECG(Dataset):
        def __init__(self, n, n_classes=5, seed=42):
            rng = np.random.RandomState(seed)
            self.X = torch.randn(n, 250, 12)  # (samples, timesteps, leads)
            self.y = torch.from_numpy(rng.randint(0, n_classes, size=n)).long()

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i].float()

    class BiLSTMECG(nn.Module):
        def __init__(self, input_dim=12, hidden=64, n_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden * 2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)

    # FL client
    import flwr as fl

    _ecg_cache = {}

    class ECGClient(fl.client.NumPyClient):
        def __init__(self, pid, n_clients):
            self.pid = pid
            self.model = BiLSTMECG().to("cpu")
            # Each client gets different data
            n_per = 200
            self.train_dl = DataLoader(SyntheticECG(n_per, seed=42 + pid), batch_size=32, shuffle=True)
            self.val_dl = DataLoader(SyntheticECG(50, seed=9999), batch_size=32)

        def get_parameters(self, config):
            return [v.cpu().numpy() for v in self.model.state_dict().values()]

        def set_parameters(self, params):
            sd = dict(zip(self.model.state_dict().keys(), params))
            self.model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.train()
            opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
            crit = nn.BCELoss()
            for X, y in self.train_dl:
                opt.zero_grad()
                crit(self.model(X), y).backward()
                opt.step()
            return self.get_parameters({}), len(self.train_dl.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            self.model.eval()
            crit = nn.BCELoss()
            loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for X, y in self.val_dl:
                    pred = self.model(X)
                    loss += crit(pred, y).item()
                    correct += ((pred > 0.5).float() == y).sum().item()
                    total += y.size(0)
            acc = correct / total if total else 0
            return loss / max(len(self.val_dl), 1), total, {"accuracy": acc}

    def ecg_client_fn(context: Context):
        pid = int(context.node_config.get("partition-id", 0))
        return ECGClient(pid, 5).to_client()

    # Build strategy
    from fl_common import build_strategy

    results = {}
    for strat_name in ["IID", "FedProx_Mu0.1_Alpha_0.5", "SCAFFOLD_Alpha_0.5", "SecAgg_Alpha_0.5"]:
        label = strat_name.split("_Alpha")[0].split("_Mu")[0]
        strat = build_strategy(strat_name, 5, model_init_fn=lambda: BiLSTMECG(),
                               metric_name="accuracy", lr=0.001, patience=10)

        class Cap:
            def __init__(s, st):
                s.strategy = st; s.acc = 0
            def __getattr__(s, n):
                return getattr(s.strategy, n)
            def configure_fit(s, *a, **kw):
                return s.strategy.configure_fit(*a, **kw)
            def configure_evaluate(s, *a, **kw):
                return s.strategy.configure_evaluate(*a, **kw)
            def aggregate_fit(s, *a, **kw):
                return s.strategy.aggregate_fit(*a, **kw)
            def aggregate_evaluate(s, *a, **kw):
                l, m = s.strategy.aggregate_evaluate(*a, **kw)
                if l is not None:
                    s.acc = float(m.get("accuracy", 0))
                return l, m

        c = Cap(strat)

        def sf(ctx):
            return ServerAppComponents(strategy=c, config=ServerConfig(num_rounds=5))

        t0 = time.time()
        try:
            run_simulation(server_app=ServerApp(server_fn=sf),
                           client_app=ClientApp(client_fn=ecg_client_fn),
                           num_supernodes=5,
                           backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
            results[label] = c.acc
            logger.info("  %s: acc=%.4f (%ds)" % (label, c.acc, time.time() - t0))
        except Exception as e:
            results[label] = 0
            logger.info("  %s: ERROR %s" % (label, e))

    logger.info("  ECG results: %s" % results)
    return results


# ======================================================================
# DOMAIN 2: Financial Fraud (synthetic tabular, same pipeline)
# ======================================================================

def test_fraud():
    """Test FL on financial fraud detection using MLP."""
    logger.info("\n" + "=" * 60)
    logger.info("DOMAIN 2: Financial Fraud Detection (MLP)")
    logger.info("=" * 60)

    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.simulation import run_simulation
    from flwr.clientapp import ClientApp
    from flwr.common import Context
    from torch.utils.data import DataLoader, Dataset

    # Synthetic fraud data: 30 features (amount, time, v1-v28), binary label
    class SyntheticFraud(Dataset):
        def __init__(self, n, fraud_rate=0.02, seed=42):
            rng = np.random.RandomState(seed)
            self.X = torch.from_numpy(rng.randn(n, 30).astype(np.float32))
            self.y = torch.from_numpy((rng.rand(n) < fraud_rate).astype(np.float32))

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    class FraudMLP(nn.Module):
        def __init__(self, input_dim=30, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(hidden, 1), nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    import flwr as fl

    class FraudClient(fl.client.NumPyClient):
        def __init__(self, pid):
            self.model = FraudMLP()
            self.train_dl = DataLoader(SyntheticFraud(1000, seed=42 + pid), batch_size=64, shuffle=True)
            self.val_dl = DataLoader(SyntheticFraud(200, seed=9999), batch_size=64)

        def get_parameters(self, config):
            return [v.cpu().numpy() for v in self.model.state_dict().values()]

        def set_parameters(self, params):
            sd = dict(zip(self.model.state_dict().keys(), params))
            self.model.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            self.model.train()
            opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
            crit = nn.BCELoss()
            for X, y in self.train_dl:
                opt.zero_grad()
                crit(self.model(X), y).backward()
                opt.step()
            return self.get_parameters({}), len(self.train_dl.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            self.model.eval()
            crit = nn.BCELoss()
            loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for X, y in self.val_dl:
                    pred = self.model(X)
                    loss += crit(pred, y).item()
                    correct += ((pred > 0.5).float() == y).sum().item()
                    total += y.size(0)
            return loss / max(len(self.val_dl), 1), total, {"accuracy": correct / total if total else 0}

    def fraud_cfn(context: Context):
        pid = int(context.node_config.get("partition-id", 0))
        return FraudClient(pid).to_client()

    from fl_common import build_strategy

    results = {}
    for strat_name in ["IID", "FedProx_Mu0.1_Alpha_0.5", "SCAFFOLD_Alpha_0.5",
                        "SecAgg_Alpha_0.5", "DP_Central_Eps50.0_Alpha_0.5"]:
        label = strat_name.split("_Alpha")[0].split("_Mu")[0]
        strat = build_strategy(strat_name, 5, model_init_fn=lambda: FraudMLP(),
                               metric_name="accuracy", lr=0.001, patience=10)

        class Cap:
            def __init__(s, st):
                s.strategy = st; s.acc = 0
            def __getattr__(s, n):
                return getattr(s.strategy, n)
            def configure_fit(s, *a, **kw):
                return s.strategy.configure_fit(*a, **kw)
            def configure_evaluate(s, *a, **kw):
                return s.strategy.configure_evaluate(*a, **kw)
            def aggregate_fit(s, *a, **kw):
                return s.strategy.aggregate_fit(*a, **kw)
            def aggregate_evaluate(s, *a, **kw):
                l, m = s.strategy.aggregate_evaluate(*a, **kw)
                if l is not None:
                    s.acc = float(m.get("accuracy", 0))
                return l, m

        c = Cap(strat)

        def sf(ctx):
            return ServerAppComponents(strategy=c, config=ServerConfig(num_rounds=5))

        t0 = time.time()
        try:
            run_simulation(server_app=ServerApp(server_fn=sf),
                           client_app=ClientApp(client_fn=fraud_cfn),
                           num_supernodes=5,
                           backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
            results[label] = c.acc
            logger.info("  %s: acc=%.4f (%ds)" % (label, c.acc, time.time() - t0))
        except Exception as e:
            results[label] = 0
            logger.info("  %s: ERROR %s" % (label, e))

    logger.info("  Fraud results: %s" % results)
    return results


# ======================================================================
# DOMAIN 3: Government LLM (federated LoRA, domain-adapted templates)
# ======================================================================

def test_gov_llm():
    """Test federated LoRA on government-domain synthetic text."""
    logger.info("\n" + "=" * 60)
    logger.info("DOMAIN 3: Government LLM (Federated LoRA on Mistral 7B)")
    logger.info("=" * 60)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    MAX_LEN = 192
    LORA_R = 16

    # Government-domain synthetic data (3 agencies)
    AGENCY_DATA = {
        0: {  # Tax/Revenue
            "name": "Tax Authority",
            "notes": [
                "%d taxpayer filed %s return showing income $%dK, deductions $%dK. %s." % (
                    random.randint(1000, 9999), random.choice(["individual", "corporate", "partnership"]),
                    random.randint(30, 500), random.randint(5, 100),
                    random.choice(["Flagged for audit", "Accepted", "Under review", "Penalty assessed",
                                   "Refund approved", "Schedule C discrepancy noted"])
                ) for _ in range(200)
            ],
        },
        1: {  # Immigration
            "name": "Immigration Agency",
            "notes": [
                "%s national, age %d, %s visa application. %s. %s." % (
                    random.choice(["Chinese", "Indian", "Filipino", "Vietnamese", "Indonesian",
                                   "Malaysian", "Thai", "Korean", "Japanese", "Australian"]),
                    random.randint(18, 65),
                    random.choice(["work", "student", "tourist", "dependent", "skilled worker"]),
                    random.choice(["Documents verified", "Interview required", "Background check pending",
                                   "Medical exam complete", "Sponsor confirmed"]),
                    random.choice(["Approved", "Denied", "Further review", "Conditional approval",
                                   "Referred to security", "Expedited processing"])
                ) for _ in range(200)
            ],
        },
        2: {  # Public Health
            "name": "Public Health Agency",
            "notes": [
                "Region %s reported %d new %s cases. %s. Population %dK. %s." % (
                    random.choice(["North", "South", "East", "West", "Central"]),
                    random.randint(5, 500),
                    random.choice(["influenza", "dengue", "tuberculosis", "hepatitis", "COVID",
                                   "measles", "food poisoning", "respiratory illness"]),
                    random.choice(["Cluster detected", "Endemic level", "Outbreak declared",
                                   "Under surveillance", "Contact tracing initiated"]),
                    random.randint(50, 5000),
                    random.choice(["Vaccination campaign launched", "Travel advisory issued",
                                   "Hospital surge capacity activated", "Quarantine measures in place"])
                ) for _ in range(200)
            ],
        },
    }

    NONMEMBER = [
        "Routine %s inspection of facility %d completed. No violations found." % (
            random.choice(["safety", "health", "fire", "building", "food"]),
            random.randint(100, 999)
        ) for _ in range(100)
    ]

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    logger.info("  Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                  device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"], bias="none")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  LoRA params: %d (%.2f%%)" % (trainable, trainable / sum(p.numel() for p in model.parameters()) * 100))

    def get_ls(m):
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in m.named_parameters()
                           if v.requires_grad and "lora" in k)

    def set_ls(m, s):
        with torch.no_grad():
            for k, v in m.named_parameters():
                if k in s:
                    v.copy_(s[k].to(v.device))

    def fedavg_ls(ss, ws=None):
        if ws is None:
            ws = [1.0 / len(ss)] * len(ss)
        r = OrderedDict()
        for k in ss[0]:
            r[k] = sum(w * s[k].float() for w, s in zip(ws, ss))
        return r

    def train_local(m, tok, notes):
        texts = ["Document: " + n for n in random.sample(notes, min(100, len(notes)))]
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        m.train()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
        tl, ns = 0, 0
        idx = list(range(len(texts)))
        random.shuffle(idx)
        for i in range(0, len(idx), 4):
            b = idx[i:i + 4]
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = m(input_ids=enc["input_ids"][b].to(DEVICE),
                      attention_mask=enc["attention_mask"][b].to(DEVICE),
                      labels=enc["input_ids"][b].to(DEVICE))
            o.loss.backward()
            opt.step()
            tl += o.loss.item()
            ns += 1
        return tl / max(ns, 1)

    def eval_mia(m, tok, members, nonmembers):
        m.eval()
        def losses(notes):
            ls = []
            for n in notes[:40]:
                e = tok("Document: " + n, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    o = m(input_ids=e["input_ids"].to(DEVICE), labels=e["input_ids"].to(DEVICE))
                ls.append(o.loss.item())
            return np.array(ls)
        ml = losses(random.sample(members, min(40, len(members))))
        nl = losses(random.sample(nonmembers, min(40, len(nonmembers))))
        al = np.concatenate([ml, nl])
        ay = np.concatenate([np.ones(len(ml)), np.zeros(len(nl))])
        best = max(((al < np.percentile(al, p)).astype(float) == ay).mean() for p in range(10, 91, 5))
        return (best - 0.5) * 2

    # Federated LoRA: 3 agencies, 5 rounds
    N_ROUNDS = 5
    logger.info("  FL training: 3 agencies, %d rounds" % N_ROUNDS)
    gs = get_ls(model)
    all_members = [n for a in AGENCY_DATA.values() for n in a["notes"]]

    for rnd in range(1, N_ROUNDS + 1):
        ss = []
        for aid in range(3):
            set_ls(model, gs)
            loss = train_local(model, tok, AGENCY_DATA[aid]["notes"])
            ss.append(get_ls(model))
        gs = fedavg_ls(ss)
        set_ls(model, gs)

        if rnd == N_ROUNDS:
            adv = eval_mia(model, tok, all_members, NONMEMBER)
            logger.info("    Round %d: loss=%.3f, MIA_adv=%.3f" % (rnd, loss, adv))

    # Test QA
    model.eval()
    qa_tests = [
        ("What are common tax audit triggers?", ["income", "deduction", "audit"]),
        ("What documents are needed for a work visa?", ["visa", "work", "document"]),
        ("How is a disease outbreak declared?", ["outbreak", "cases", "surveillance"]),
    ]
    qa_score = 0
    for q, kws in qa_tests:
        inp = tok("<s>[INST] " + q + " [/INST]", return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model.generate(**inp, max_new_tokens=60, temperature=0.3, do_sample=True)
        r = tok.decode(o[0], skip_special_tokens=True)
        if "[/INST]" in r:
            r = r.split("[/INST]")[-1]
        hits = sum(1 for k in kws if k.lower() in r.lower())
        qa_score += hits / len(kws)
        logger.info("    Q: %s" % q)
        logger.info("    A: %s..." % r[:120])

    qa_score /= len(qa_tests)
    logger.info("  Gov LLM: QA=%.3f, MIA=%.3f" % (qa_score, adv))

    return {"qa": qa_score, "mia_advantage": adv}


# ======================================================================
# MAIN
# ======================================================================

def main():
    logger.info("#" * 60)
    logger.info("  TESTING 3 NEW DOMAINS")
    logger.info("#" * 60)
    t0 = time.time()

    ecg = test_ecg()
    fraud = test_fraud()
    gov = test_gov_llm()

    SEP = "=" * 60
    logger.info("\n" + SEP)
    logger.info("RESULTS")
    logger.info(SEP)
    logger.info("\n  ECG (BiLSTM, 5 rounds):")
    for k, v in ecg.items():
        logger.info("    %-15s acc=%.4f %s" % (k, v, "PASS" if v > 0.01 else "FAIL"))
    logger.info("\n  Financial Fraud (MLP, 5 rounds):")
    for k, v in fraud.items():
        logger.info("    %-15s acc=%.4f %s" % (k, v, "PASS" if v > 0.01 else "FAIL"))
    logger.info("\n  Government LLM (Mistral 7B LoRA, 5 rounds):")
    logger.info("    QA Score:      %.3f" % gov["qa"])
    logger.info("    MIA Advantage: %.3f" % gov["mia_advantage"])

    logger.info("\n  Total: %ds (%.0fmin)" % (time.time() - t0, (time.time() - t0) / 60))
    logger.info(SEP)


if __name__ == "__main__":
    main()
