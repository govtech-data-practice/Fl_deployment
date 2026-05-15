#!/usr/bin/env python3
"""
Federated LoRA Fine-tuning of Mistral 7B + Privacy Leakage Test
================================================================
3 hospitals × 2000 clinical notes each, QLoRA, 10 FL rounds.
Compares privacy leakage with and without DP.
"""

import sys
import os
import json
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger("fed_mistral")

DEVICE = "cuda"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
NUM_HOSPITALS = 3
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 4
LR = 2e-4
MAX_LEN = 192
LORA_R = 16
EVAL_EVERY = 2  # evaluate MIA every N rounds
DATA_DIR = "clinical_data"


# ======================================================================
# Data loading
# ======================================================================

def load_hospital_data(hid):
    path = os.path.join(DATA_DIR, f"hospital_{hid}.json")
    with open(path) as f:
        return json.load(f)["notes"]


def load_nonmember_data():
    with open(os.path.join(DATA_DIR, "nonmember.json")) as f:
        return json.load(f)["notes"]


def tokenize_notes(notes, tokenizer, max_len=MAX_LEN):
    texts = [f"<s>[INST] Summarize this clinical note [/INST] {n}</s>" for n in notes]
    enc = tokenizer(
        texts, truncation=True, max_length=max_len,
        padding="max_length", return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    return enc


# ======================================================================
# Model setup
# ======================================================================

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  LoRA: {trainable:,} / {total:,} trainable ({trainable/total*100:.2f}%)")

    return model, tokenizer


def get_lora_state(model):
    return OrderedDict(
        (k, v.detach().cpu().clone())
        for k, v in model.named_parameters() if v.requires_grad and "lora" in k
    )


def set_lora_state(model, state):
    with torch.no_grad():
        for k, v in model.named_parameters():
            if k in state:
                v.copy_(state[k].to(v.device))


def fedavg_lora(states, weights):
    avg = OrderedDict()
    for key in states[0]:
        avg[key] = sum(w * s[key].float() for w, s in zip(weights, states))
    return avg


def dp_noise_state(state, prev_state, clip_norm, noise_mult, seed):
    rng = np.random.RandomState(seed)
    delta = OrderedDict((k, state[k] - prev_state[k]) for k in state)
    flat = torch.cat([d.flatten() for d in delta.values()])
    scale = min(1.0, clip_norm / (flat.norm().item() + 1e-10))
    noisy = OrderedDict()
    for k, d in delta.items():
        d_clipped = d * scale
        noise = torch.from_numpy(rng.normal(0, noise_mult * clip_norm, size=d.shape).astype(np.float32))
        noisy[k] = prev_state[k] + d_clipped + noise
    return noisy


# ======================================================================
# Local training
# ======================================================================

def local_train(model, tokenizer, notes, epochs=LOCAL_EPOCHS):
    """Train on a sample of hospital notes (use subset per round for speed)."""
    sample = random.sample(notes, min(200, len(notes)))  # 200 notes per round
    enc = tokenize_notes(sample, tokenizer)

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    total_loss, steps = 0.0, 0
    n = len(sample)
    for _ in range(epochs):
        indices = list(range(n))
        random.shuffle(indices)
        for i in range(0, n, BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            ids = enc["input_ids"][batch_idx].to(DEVICE)
            mask = enc["attention_mask"][batch_idx].to(DEVICE)
            labels = enc["labels"][batch_idx].to(DEVICE)

            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            total_loss += out.loss.item()
            steps += 1

    return total_loss / max(steps, 1), len(sample)


# ======================================================================
# Privacy: Membership Inference Attack
# ======================================================================

def compute_losses(model, tokenizer, notes, batch_size=8):
    """Per-sample loss for MIA."""
    model.eval()
    enc = tokenize_notes(notes, tokenizer)
    losses = []
    with torch.no_grad():
        for i in range(0, len(notes), batch_size):
            ids = enc["input_ids"][i:i+batch_size].to(DEVICE)
            mask = enc["attention_mask"][i:i+batch_size].to(DEVICE)
            labels = enc["labels"][i:i+batch_size].to(DEVICE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=mask, labels=labels)

            # Per-sample loss via manual computation
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            per_token = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_labels.size())

            # Mean per sample (ignore padding)
            pad_mask = (shift_labels != tokenizer.pad_token_id).float()
            per_sample = (per_token * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)
            losses.extend(per_sample.cpu().tolist())

    return np.array(losses)


def mia_attack(model, tokenizer, member_notes, nonmember_notes, label=""):
    """Threshold-based MIA: members have lower loss."""
    logger.info(f"\n  MIA on {label} model:")

    # Use subset for speed
    mem_sample = random.sample(member_notes, min(300, len(member_notes)))
    non_sample = random.sample(nonmember_notes, min(300, len(nonmember_notes)))

    mem_losses = compute_losses(model, tokenizer, mem_sample)
    non_losses = compute_losses(model, tokenizer, non_sample)

    all_l = np.concatenate([mem_losses, non_losses])
    all_y = np.concatenate([np.ones(len(mem_losses)), np.zeros(len(non_losses))])

    # Find best threshold
    best_acc = 0.5
    for pct in range(10, 91, 5):
        t = np.percentile(all_l, pct)
        preds = (all_l < t).astype(float)
        acc = (preds == all_y).mean()
        best_acc = max(best_acc, acc)

    advantage = (best_acc - 0.5) * 2
    gap = non_losses.mean() - mem_losses.mean()

    logger.info(f"    Members:     loss={mem_losses.mean():.4f} ± {mem_losses.std():.4f}")
    logger.info(f"    Non-members: loss={non_losses.mean():.4f} ± {non_losses.std():.4f}")
    logger.info(f"    Loss gap:    {gap:.4f}")
    logger.info(f"    Best MIA acc: {best_acc:.4f}")
    logger.info(f"    Advantage:   {advantage:.4f} {'— VULNERABLE' if advantage > 0.1 else '— PROTECTED'}")

    return best_acc, advantage, gap


# ======================================================================
# Main FL loop
# ======================================================================

def run_fl(use_dp=False, clip_norm=1.0, noise_mult=0.5):
    label = "DP-FL" if use_dp else "FL"
    logger.info(f"\n{'#'*60}")
    logger.info(f"  {label}: Mistral 7B QLoRA, {NUM_HOSPITALS} hospitals × 2000 notes")
    logger.info(f"  {NUM_ROUNDS} rounds, {LOCAL_EPOCHS} local epoch, 200 notes/round/hospital")
    if use_dp:
        logger.info(f"  DP: clip={clip_norm}, σ={noise_mult}")
    logger.info(f"{'#'*60}")

    model, tokenizer = load_model()
    global_state = get_lora_state(model)

    payload_kb = sum(v.numel() * 4 for v in global_state.values()) / 1024
    logger.info(f"  LoRA payload: {payload_kb:.0f} KB per round")

    all_member_notes = []
    hospital_notes = {}
    for hid in range(NUM_HOSPITALS):
        notes = load_hospital_data(hid)
        hospital_notes[hid] = notes
        all_member_notes.extend(notes)
    nonmember_notes = load_nonmember_data()

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()
        client_states = []
        client_sizes = []

        for hid in range(NUM_HOSPITALS):
            set_lora_state(model, global_state)
            prev = get_lora_state(model)

            loss, n_samples = local_train(model, tokenizer, hospital_notes[hid])
            new_state = get_lora_state(model)

            if use_dp:
                new_state = dp_noise_state(
                    new_state, prev, clip_norm, noise_mult,
                    seed=rnd * 100 + hid,
                )

            client_states.append(new_state)
            client_sizes.append(n_samples)

        total_n = sum(client_sizes)
        weights = [n / total_n for n in client_sizes]
        global_state = fedavg_lora(client_states, weights)

        # Quick eval
        set_lora_state(model, global_state)
        model.eval()
        eval_notes = random.sample(all_member_notes, 50)
        eval_losses = compute_losses(model, tokenizer, eval_notes, batch_size=8)
        dt = time.time() - t0
        logger.info(f"  Round {rnd}/{NUM_ROUNDS}: eval_loss={eval_losses.mean():.4f} ({dt:.1f}s)")

        # Periodic MIA check
        if rnd % EVAL_EVERY == 0 or rnd == NUM_ROUNDS:
            mia_attack(model, tokenizer, all_member_notes, nonmember_notes, f"{label} round {rnd}")

    return model, tokenizer, global_state


def main():
    logger.info("=" * 60)
    logger.info("FEDERATED MISTRAL 7B QLORA — PRIVACY LEAKAGE TEST")
    logger.info(f"Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    # Generate data if needed
    if not os.path.exists(DATA_DIR):
        logger.info("Generating clinical data...")
        import gen_clinical_data
        gen_clinical_data.main()

    t_start = time.time()

    # --- FL without DP ---
    model_nodp, tok, _ = run_fl(use_dp=False)
    t_nodp = time.time()

    # Clear GPU for next run
    del model_nodp
    torch.cuda.empty_cache()

    # --- FL with DP ---
    model_dp, tok, _ = run_fl(use_dp=True, clip_norm=1.0, noise_mult=1.0)
    t_dp = time.time()

    # --- Final Report ---
    logger.info(f"\n{'='*60}")
    logger.info("FINAL REPORT")
    logger.info(f"{'='*60}")
    logger.info(f"  Model: {MODEL_ID} (4-bit QLoRA)")
    logger.info(f"  Data: {NUM_HOSPITALS} hospitals × 2000 notes")
    logger.info(f"  FL rounds: {NUM_ROUNDS}")
    logger.info(f"  FL (no DP): {t_nodp - t_start:.0f}s")
    logger.info(f"  FL (DP):    {t_dp - t_nodp:.0f}s")
    logger.info(f"  Total:      {t_dp - t_start:.0f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
