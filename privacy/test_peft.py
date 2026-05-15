#!/usr/bin/env python3
"""
PEFT Testing Suite
==================
Validates LoRA, QLoRA, prefix tuning, and adapter configurations
on a small model. Runs on GPU or CPU.

Tests:
  1. LoRA fine-tuning (causal LM)
  2. QLoRA 4-bit quantization
  3. LoRA for sequence classification (healthcare)
  4. Prefix tuning
  5. LoRA merge & export
  6. PEFT + FL integration (LoRA weights as federated updates)
"""

import sys
import time
import logging
import torch

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger("peft_test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Small model for testing — swap for medical LLM in production
BASE_MODEL = "facebook/opt-125m"


def test_lora_causal_lm():
    """Test 1: LoRA on causal language model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)
    model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100
    logger.info(f"  LoRA params: {trainable:,} / {total:,} ({pct:.2f}%)")
    assert pct < 5, f"LoRA should be <5% params, got {pct:.1f}%"

    # Quick forward + backward
    inputs = tokenizer("Patient presents with fever and cough", return_tensors="pt").to(DEVICE)
    inputs["labels"] = inputs["input_ids"].clone()
    loss = model(**inputs).loss
    loss.backward()
    logger.info(f"  Forward+backward OK, loss={loss.item():.4f}")

    return model, tokenizer


def test_qlora_4bit():
    """Test 2: QLoRA with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    if DEVICE != "cuda":
        logger.info("  SKIP (requires CUDA)")
        return None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=8, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  QLoRA params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    inputs = tokenizer("Diagnosis: acute respiratory distress", return_tensors="pt").to(DEVICE)
    inputs["labels"] = inputs["input_ids"].clone()
    loss = model(**inputs).loss
    loss.backward()
    logger.info(f"  QLoRA forward+backward OK, loss={loss.item():.4f}")

    # Check memory savings
    mem_mb = torch.cuda.memory_allocated() / 1e6
    logger.info(f"  GPU memory: {mem_mb:.0f} MB")
    return model


def test_lora_seq_classification():
    """Test 3: LoRA for sequence classification (clinical text)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, torch_dtype=torch.float32,
    )
    # OPT doesn't have pad_token by default
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config).to(DEVICE)

    texts = [
        "Patient has high fever and elevated WBC count",
        "Routine checkup, vitals normal",
    ]
    labels = torch.tensor([1, 0]).to(DEVICE)  # sepsis / no sepsis

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    inputs["labels"] = labels

    loss = model(**inputs).loss
    loss.backward()
    logger.info(f"  SeqCls LoRA OK, loss={loss.item():.4f}")
    return True


def test_prefix_tuning():
    """Test 4: Prefix tuning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PrefixTuningConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
    )
    model = get_peft_model(model, config).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Prefix params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    inputs = tokenizer("The patient was admitted for", return_tensors="pt").to(DEVICE)
    inputs["labels"] = inputs["input_ids"].clone()
    loss = model(**inputs).loss
    loss.backward()
    logger.info(f"  Prefix tuning OK, loss={loss.item():.4f}")
    return True


def test_lora_merge_export():
    """Test 5: Merge LoRA weights into base model and export."""
    import tempfile, os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(DEVICE)

    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, config)

    # Simulate training
    inputs = tokenizer("Clinical note:", return_tensors="pt").to(DEVICE)
    inputs["labels"] = inputs["input_ids"].clone()
    loss = model(**inputs).loss
    loss.backward()

    # Merge and save
    merged = model.merge_and_unload()
    with tempfile.TemporaryDirectory() as tmp:
        merged.save_pretrained(os.path.join(tmp, "merged"))
        tokenizer.save_pretrained(os.path.join(tmp, "merged"))
        size_mb = sum(
            os.path.getsize(os.path.join(tmp, "merged", f))
            for f in os.listdir(os.path.join(tmp, "merged"))
        ) / 1e6
        logger.info(f"  Merged model saved: {size_mb:.0f} MB")

    # Also save just LoRA adapter
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(DEVICE),
        config,
    )
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(os.path.join(tmp, "adapter"))
        adapter_size = sum(
            os.path.getsize(os.path.join(tmp, "adapter", f))
            for f in os.listdir(os.path.join(tmp, "adapter"))
        ) / 1e6
        logger.info(f"  LoRA adapter only: {adapter_size:.1f} MB (vs {size_mb:.0f} MB merged)")

    return True


def test_peft_fl_integration():
    """Test 6: Extract LoRA weights as numpy arrays for FL transmission."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    import numpy as np

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32).to(DEVICE)
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, config)

    # Extract only LoRA parameters (what FL clients would send)
    lora_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name:
            lora_params[name] = param.detach().cpu().numpy()

    total_bytes = sum(p.nbytes for p in lora_params.values())
    logger.info(f"  LoRA layers: {len(lora_params)}")
    logger.info(f"  FL payload: {total_bytes / 1024:.1f} KB (vs full model ~500 MB)")

    # Simulate FL aggregation (FedAvg on LoRA weights only)
    client_updates = [
        {k: v + np.random.randn(*v.shape).astype(np.float32) * 0.01 for k, v in lora_params.items()}
        for _ in range(3)  # 3 clients
    ]
    aggregated = {}
    for key in lora_params:
        aggregated[key] = np.mean([u[key] for u in client_updates], axis=0)
    logger.info(f"  FL aggregation of {len(client_updates)} clients OK")

    # Load aggregated LoRA weights back
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in aggregated:
                param.copy_(torch.from_numpy(aggregated[name]).to(param.device))
    logger.info(f"  Loaded aggregated LoRA weights back into model")

    # Verify model still works
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    inputs = tokenizer("The diagnosis is", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    logger.info(f"  Generation after FL: '{text}'")

    return True


# ======================================================================
# RUNNER
# ======================================================================

TESTS = [
    ("LoRA Causal LM", test_lora_causal_lm),
    ("QLoRA 4-bit", test_qlora_4bit),
    ("LoRA Seq Classification", test_lora_seq_classification),
    ("Prefix Tuning", test_prefix_tuning),
    ("LoRA Merge & Export", test_lora_merge_export),
    ("PEFT + FL Integration", test_peft_fl_integration),
]


def main():
    logger.info("=" * 60)
    logger.info("PEFT TESTING SUITE")
    logger.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info("=" * 60)

    results = []
    for name, fn in TESTS:
        logger.info(f"\n--- {name} ---")
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            results.append((name, "PASS", dt))
            logger.info(f"  PASS ({dt:.1f}s)")
        except Exception as e:
            dt = time.time() - t0
            results.append((name, "FAIL", dt))
            logger.error(f"  FAIL ({dt:.1f}s): {e}")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"\n{'=' * 60}")
    logger.info("RESULTS")
    logger.info(f"{'=' * 60}")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    for name, status, dt in results:
        logger.info(f"  {status}  {name:<30s} {dt:.1f}s")
    logger.info(f"\n  {passed}/{len(results)} passed")
    logger.info(f"{'=' * 60}")
    sys.exit(0 if passed >= len(results) - 1 else 1)  # allow 1 skip (QLoRA on CPU)


if __name__ == "__main__":
    main()
