#!/usr/bin/env python3
"""
PEFT Benchmark: Federated LoRA on Mistral 7B
Tests centralized vs federated vs federated+DP, with MIA and QA evaluation.
"""
import sys, os, time, json, logging, random, math
import numpy as np
from collections import OrderedDict
import torch, torch.nn as nn

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s | %(message)s")
logger = logging.getLogger("peft")

DEVICE = "cuda"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
N_HOSPITALS = 3
N_ROUNDS = 10
LOCAL_NOTES = 200
MAX_LEN = 192
LORA_R = 16

random.seed(42); torch.manual_seed(42)

def gen_notes(dx_list, plan_list, n=600):
    notes = []
    for i in range(n):
        age = random.randint(28, 92)
        sex = random.choice(["M", "F"])
        dx = dx_list[i % len(dx_list)]
        plan = plan_list[i % len(plan_list)]
        hr = random.randint(55, 145)
        bp = "%d/%d" % (random.randint(75, 195), random.randint(40, 105))
        notes.append("%d%s with %s. HR %d, BP %s. %s." % (age, sex, dx, hr, bp, plan))
    return notes

HOSPITAL_NOTES = {
    0: gen_notes(["STEMI","NSTEMI","CHF","AFib","aortic stenosis","VT storm","cardiac tamponade","acute MI"],
                 ["PCI with DES","IV furosemide","rate control metoprolol","TAVR consult","amiodarone load","cardioversion","pericardiocentesis","CABG consult"]),
    1: gen_notes(["COPD exacerbation","bilateral PE","pneumothorax","status asthmaticus","lung mass","ILD","pleural effusion","pulmonary HTN"],
                 ["BiPAP+methylpred","heparin drip","chest tube placed","continuous albuterol+mag","bronchoscopy scheduled","nintedanib started","thoracentesis 1.5L","inhaled treprostinil"]),
    2: gen_notes(["septic shock","urosepsis","necrotizing fasciitis","bacteremia","cholangitis","C diff colitis","meningitis","neutropenic fever"],
                 ["30mL/kg bolus+pip-tazo+norepi","meropenem+vasopressin","emergent debridement","line removal+micafungin","ERCP decompression","PO vancomycin QID","ceftriaxone+vanco+dex","cefepime+GCSF"]),
}
NONMEMBER = gen_notes(["DVT","pancreatitis","GI bleed","DKA","ischemic stroke","hip fracture","cellulitis","SBO"],
                       ["heparin","NPO+IVF","transfusion","insulin drip","alteplase","surgery consult","IV abx","NGT decompression"], n=200)

QA = [("What are the symptoms and initial management of STEMI?", ["chest pain","st elevation","troponin","pci","aspirin"]),
      ("How do you manage septic shock per Sepsis-3?", ["fluid","crystalloid","antibiotic","vasopressor","lactate"]),
      ("Treatment for acute COPD exacerbation with respiratory failure?", ["steroid","bronchodilator","bipap","oxygen"])]


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token; tok.padding_side = "right"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16)
    m = prepare_model_for_kbit_training(m)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
    m = get_peft_model(m, cfg)
    t = sum(p.numel() for p in m.parameters() if p.requires_grad)
    a = sum(p.numel() for p in m.parameters())
    logger.info("  LoRA: %d / %d (%.2f%%)" % (t, a, t/a*100))
    return m, tok

def get_ls(m): return OrderedDict((k,v.detach().cpu().clone()) for k,v in m.named_parameters() if v.requires_grad and "lora" in k)
def set_ls(m,s):
    with torch.no_grad():
        for k,v in m.named_parameters():
            if k in s: v.copy_(s[k].to(v.device))
def favg(ss,ws=None):
    if ws is None: ws=[1.0/len(ss)]*len(ss)
    r=OrderedDict()
    for k in ss[0]: r[k]=sum(w*s[k].float() for w,s in zip(ws,ss))
    return r

def train_local(m, tok, notes, use_dp=False, dp_clip=5.0, dp_noise=0.02):
    sample = random.sample(notes, min(LOCAL_NOTES, len(notes)))
    texts = ["Clinical note: " + n for n in sample]
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    tl, ns = 0.0, 0
    idx = list(range(len(texts))); random.shuffle(idx)
    for i in range(0, len(idx), 4):
        b = idx[i:i+4]
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = m(input_ids=enc["input_ids"][b].to(DEVICE), attention_mask=enc["attention_mask"][b].to(DEVICE), labels=enc["input_ids"][b].to(DEVICE))
        o.loss.backward()
        if use_dp:
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], dp_clip)
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad += torch.randn_like(p.grad) * dp_noise * dp_clip
        opt.step()
        tl += o.loss.item(); ns += 1
    return tl / max(ns, 1)

def eval_qa(m, tok):
    m.eval(); sc = 0
    for q, kws in QA:
        inp = tok("<s>[INST] " + q + " [/INST]", return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = m.generate(**inp, max_new_tokens=80, temperature=0.3, do_sample=True, top_p=0.9)
        r = tok.decode(o[0], skip_special_tokens=True)
        if "[/INST]" in r: r = r.split("[/INST]")[-1]
        sc += sum(1 for k in kws if k.lower() in r.lower()) / len(kws)
    return sc / len(QA)

def eval_mia(m, tok, mem, non):
    m.eval()
    def losses(ns):
        ls = []
        for n in ns[:60]:
            e = tok("Clinical note: " + n, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = m(input_ids=e["input_ids"].to(DEVICE), labels=e["input_ids"].to(DEVICE))
            ls.append(o.loss.item())
        return np.array(ls)
    ml = losses(random.sample(mem, min(60, len(mem))))
    nl = losses(random.sample(non, min(60, len(non))))
    al = np.concatenate([ml, nl])
    ay = np.concatenate([np.ones(len(ml)), np.zeros(len(nl))])
    best = max(((al < np.percentile(al, p)).astype(float) == ay).mean() for p in range(10, 91, 5))
    return (best - 0.5) * 2, ml.mean() - nl.mean()

def reset_lora(m):
    for n, p in m.named_parameters():
        if "lora" in n and p.requires_grad:
            if "lora_B" in n: nn.init.zeros_(p)
            else: nn.init.kaiming_uniform_(p, a=math.sqrt(5))

def run_fl(m, tok, use_dp, label):
    logger.info("  %s: %d hospitals, %d rounds" % (label, N_HOSPITALS, N_ROUNDS))
    gs = get_ls(m)
    kb = sum(v.numel()*4 for v in gs.values()) / 1024
    all_mem = [n for h in HOSPITAL_NOTES.values() for n in h]
    for rnd in range(1, N_ROUNDS+1):
        ss, sz = [], []
        for hid in range(N_HOSPITALS):
            set_ls(m, gs)
            train_local(m, tok, HOSPITAL_NOTES[hid], use_dp=use_dp)
            ss.append(get_ls(m)); sz.append(LOCAL_NOTES)
        tn = sum(sz); ws = [n/tn for n in sz]
        gs = favg(ss, ws); set_ls(m, gs)
        if rnd % 3 == 0 or rnd == N_ROUNDS:
            qa = eval_qa(m, tok)
            adv, gap = eval_mia(m, tok, all_mem, NONMEMBER)
            logger.info("    Round %d: QA=%.3f, MIA_adv=%.3f, gap=%.3f, payload=%dKB" % (rnd, qa, adv, gap, kb))
    return gs

def main():
    SEP = "=" * 70
    logger.info(SEP)
    logger.info("PEFT BENCHMARK: Federated LoRA on Mistral 7B")
    logger.info("  GPU: %s" % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))
    logger.info("  Hospitals: %d, Rounds: %d, Notes/round: %d" % (N_HOSPITALS, N_ROUNDS, LOCAL_NOTES))
    logger.info(SEP)
    t0 = time.time()

    logger.info("\n--- Loading model ---")
    m, tok = load_model()
    base_qa = eval_qa(m, tok)
    logger.info("  Base QA: %.3f" % base_qa)

    logger.info("\n--- 1. Centralized ---")
    all_notes = [n for h in HOSPITAL_NOTES.values() for n in h]
    for ep in range(1, N_ROUNDS+1):
        train_local(m, tok, all_notes)
        if ep % 3 == 0 or ep == N_ROUNDS:
            qa = eval_qa(m, tok)
            adv, gap = eval_mia(m, tok, all_notes, NONMEMBER)
            logger.info("    Epoch %d: QA=%.3f, MIA=%.3f" % (ep, qa, adv))
    c_qa = eval_qa(m, tok)
    c_adv, _ = eval_mia(m, tok, all_notes, NONMEMBER)

    logger.info("\n--- 2. Federated LoRA ---")
    reset_lora(m)
    run_fl(m, tok, use_dp=False, label="FL")
    f_qa = eval_qa(m, tok)
    f_adv, _ = eval_mia(m, tok, all_notes, NONMEMBER)

    logger.info("\n--- 3. Federated LoRA + DP ---")
    reset_lora(m)
    run_fl(m, tok, use_dp=True, label="FL+DP")
    d_qa = eval_qa(m, tok)
    d_adv, _ = eval_mia(m, tok, all_notes, NONMEMBER)

    dt = time.time() - t0
    logger.info("\n" + SEP)
    logger.info("RESULTS")
    logger.info(SEP)
    logger.info("  %-25s %10s %15s" % ("Method", "QA Score", "MIA Advantage"))
    logger.info("  " + "-" * 52)
    logger.info("  %-25s %10.3f %15s" % ("Base (no fine-tune)", base_qa, "N/A"))
    logger.info("  %-25s %10.3f %15.4f" % ("Centralized", c_qa, c_adv))
    logger.info("  %-25s %10.3f %15.4f" % ("Federated LoRA", f_qa, f_adv))
    logger.info("  %-25s %10.3f %15.4f" % ("Federated LoRA + DP", d_qa, d_adv))
    logger.info("\n  Adapter payload: %d KB (vs ~14GB full model)" % (sum(v.numel()*4 for v in get_ls(m).values()) / 1024))
    logger.info("  Total: %ds (%.0fmin)" % (dt, dt/60))
    logger.info(SEP)

if __name__ == "__main__":
    main()
