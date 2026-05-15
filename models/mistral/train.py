#!/usr/bin/env python3
"""
Mistral 7B QLoRA Fine-tuning on Medical Data
=============================================
Uses 4-bit quantization + LoRA to fit on a single L4 (24GB).
Trains on a synthetic medical instruction dataset for testing.
"""

import os
import sys
import time
import logging
import torch

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(message)s")
logger = logging.getLogger("mistral")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./mistral-medical-lora"
NUM_TRAIN_STEPS = 50
BATCH_SIZE = 2
MAX_SEQ_LEN = 256


# ======================================================================
# Synthetic medical instruction data (for testing — swap for real data)
# ======================================================================

MEDICAL_SAMPLES = [
    {"instruction": "What are the symptoms of sepsis?",
     "response": "Common symptoms of sepsis include fever or hypothermia, tachycardia (heart rate >90 bpm), tachypnea (respiratory rate >20), altered mental status, and signs of organ dysfunction such as decreased urine output or elevated lactate levels."},
    {"instruction": "Describe the treatment protocol for community-acquired pneumonia.",
     "response": "Treatment includes empirical antibiotics (typically a beta-lactam plus a macrolide, or a respiratory fluoroquinolone), supportive care with oxygen therapy if SpO2 <94%, fluid resuscitation, and monitoring for complications such as empyema or respiratory failure."},
    {"instruction": "What is the differential diagnosis for chest pain?",
     "response": "Differential diagnosis includes acute coronary syndrome, pulmonary embolism, aortic dissection, pneumothorax, pericarditis, costochondritis, GERD, and anxiety-related chest pain. ECG, troponin, and chest imaging help differentiate these conditions."},
    {"instruction": "Explain the pathophysiology of diabetic ketoacidosis.",
     "response": "DKA results from absolute or relative insulin deficiency leading to unrestrained lipolysis, producing excess free fatty acids that are converted to ketone bodies in the liver. This causes metabolic acidosis, osmotic diuresis from hyperglycemia, and electrolyte derangements."},
    {"instruction": "What are the indications for mechanical ventilation?",
     "response": "Indications include acute respiratory failure (PaO2 <60 mmHg or PaCO2 >50 mmHg), inability to protect the airway, respiratory muscle fatigue, severe hemodynamic instability, and failure of non-invasive ventilation."},
    {"instruction": "Describe the management of acute myocardial infarction.",
     "response": "STEMI management follows the MONA protocol: Morphine, Oxygen, Nitroglycerin, Aspirin, plus urgent percutaneous coronary intervention (PCI) within 90 minutes or fibrinolytic therapy within 12 hours. Antiplatelet agents, anticoagulation, and beta-blockers are standard adjuncts."},
    {"instruction": "What laboratory findings suggest acute kidney injury?",
     "response": "AKI is indicated by rising serum creatinine (>0.3 mg/dL increase within 48h or >1.5x baseline within 7 days), decreased urine output (<0.5 mL/kg/h for >6 hours), elevated BUN, hyperkalemia, metabolic acidosis, and fluid overload."},
    {"instruction": "How is atrial fibrillation managed?",
     "response": "Management involves rate control (beta-blockers, calcium channel blockers, or digoxin), rhythm control (amiodarone, flecainide, or cardioversion), and stroke prevention with anticoagulation (DOACs or warfarin) based on CHA2DS2-VASc score."},
    {"instruction": "What are the warning signs of stroke?",
     "response": "Warning signs follow the BE-FAST mnemonic: Balance loss, Eyes (vision changes), Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Additional signs include sudden severe headache, confusion, and numbness."},
    {"instruction": "Describe the classification of heart failure.",
     "response": "Heart failure is classified by ejection fraction: HFrEF (EF ≤40%), HFmrEF (EF 41-49%), and HFpEF (EF ≥50%). NYHA functional classification grades severity I-IV based on symptoms during physical activity. ACC/AHA stages A-D describe disease progression."},
    {"instruction": "What is the Glasgow Coma Scale?",
     "response": "GCS assesses consciousness via three components: Eye opening (1-4), Verbal response (1-5), Motor response (1-6). Total score ranges 3-15. Score ≤8 indicates severe brain injury requiring intubation, 9-12 is moderate, 13-15 is mild."},
    {"instruction": "Explain the Sepsis-3 diagnostic criteria.",
     "response": "Sepsis-3 defines sepsis as life-threatening organ dysfunction caused by a dysregulated host response to infection, identified by a SOFA score increase ≥2 points. Septic shock is defined as sepsis with persistent hypotension requiring vasopressors and serum lactate >2 mmol/L despite adequate fluid resuscitation."},
]


def format_prompt(sample):
    return (
        f"<s>[INST] {sample['instruction']} [/INST] "
        f"{sample['response']}</s>"
    )


def main():
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    logger.info("=" * 60)
    logger.info("MISTRAL 7B QLORA FINE-TUNING")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 60)

    # --- Quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- Tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Model ---
    logger.info("Loading model (4-bit quantized)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    load_time = time.time() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    mem_gb = torch.cuda.memory_allocated() / 1e9
    logger.info(f"GPU memory after load: {mem_gb:.1f} GB")

    # --- LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA: {trainable:,} / {total:,} trainable ({trainable/total*100:.2f}%)")

    # --- Dataset ---
    logger.info("Preparing dataset...")
    texts = [format_prompt(s) for s in MEDICAL_SAMPLES]

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=MAX_SEQ_LEN,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.map(lambda x: {"labels": x["input_ids"]})
    dataset.set_format("torch")

    logger.info(f"Training samples: {len(dataset)}")

    # --- Training ---
    logger.info("Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
    )

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    mem_gb = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Training complete in {train_time:.1f}s")
    logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    # --- Save adapter ---
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    adapter_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    ) / 1e6
    logger.info(f"Adapter saved: {adapter_size:.1f} MB at {OUTPUT_DIR}")

    # --- Inference test ---
    logger.info("\n--- Inference Test ---")
    prompts = [
        "What are the early signs of sepsis?",
        "How do you treat acute heart failure?",
        "What is the SOFA score used for?",
    ]

    model.eval()
    for prompt in prompts:
        inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt").to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                **inputs, max_new_tokens=100, temperature=0.7,
                do_sample=True, top_p=0.9,
            )
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract just the response part
        if "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()
        logger.info(f"\nQ: {prompt}")
        logger.info(f"A: {response[:200]}...")

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info(f"  Model: {MODEL_ID}")
    logger.info(f"  Adapter: {adapter_size:.1f} MB")
    logger.info(f"  Training: {train_time:.1f}s")
    logger.info(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
