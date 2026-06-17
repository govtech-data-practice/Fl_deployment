# Tutorial 11: LLM Federated Fine-tuning

**Time:** 30 minutes | **Level:** Advanced | **Prerequisites:** [Tutorial 4](../intermediate/04-differential-privacy.md), GPU recommended

## What You'll Learn

- Why federate LLM fine-tuning (LoRA/QLoRA)
- Run federated LoRA on Mistral 7B
- Understand adapter aggregation
- Measure privacy leakage in LLMs

## Concept: Why Federated LoRA?

Fine-tuning a 7B-parameter LLM on sensitive data creates privacy risks. Federated LoRA addresses this:

1. **Freeze the base model** — the pretrained weights don't change
2. **Train only LoRA adapters** — small matrices (~160MB vs 14GB)
3. **Federate the adapters** — each site trains locally, server aggregates adapter weights
4. **Apply DP to adapters** — noise only added to the small adapter, not the full model

```
                  Frozen Mistral 7B (14GB) — never shared
                         |
         +───────────────+───────────────+
         |               |               |
    Site A            Site B            Site C
    [LoRA adapter     [LoRA adapter     [LoRA adapter
     160MB, trained    160MB, trained    160MB, trained
     on local docs]    on local docs]    on local docs]
         |               |               |
         +──── Server aggregates adapters only ────+
```

## Step 1: Install LLM Dependencies

```bash
pip install -e ".[pets]"
```

This installs `transformers`, `peft`, `accelerate`, and `bitsandbytes`.

## Step 2: Understanding the Models

Two LLM models are available in `models/llm/`:

| Model | Parameters | Adapter | Use Case |
|-------|-----------|---------|----------|
| Mistral 7B | 7B (QLoRA 4-bit) | 160MB | Clinical NLP, document QA |
| OLMo | 1-7B | LoRA | Open-source alternative |

## Step 3: Run Federated LoRA (OLMo)

```bash
# OLMo federated LoRA fine-tuning
python runners/run_ec2.py gov_doc --synthetic
```

This:
1. Loads the base OLMo model (frozen)
2. Adds LoRA adapter layers
3. Each simulated client fine-tunes the adapter on local documents
4. Server aggregates adapter weights using FedAvg
5. Adapter is merged back for inference

## Step 4: Privacy in LLM Fine-tuning

LLMs are particularly vulnerable to memorisation. Key findings from our testing:

| Attack | Without DP | With DP |
|--------|-----------|---------|
| Membership inference (MIA) | AUC ~1.0 | AUC ~0.83 |
| Canary extraction | 41.7% | 16.7% |
| Verbatim memorisation | High | Reduced |

The privacy attack suite (`privacy/attack_suite.py`) tests these attacks specifically for LLMs.

## Step 5: LoRA Configuration

Key LoRA parameters (in `models/llm/`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank (adapter size) |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout in adapter layers |
| `target_modules` | `["q_proj", "v_proj"]` | Which layers get adapters |

Higher `lora_r` = more capacity but larger adapter = more communication.

## Step 6: Adapter Aggregation

Unlike standard FL where full model weights are aggregated, federated LoRA only aggregates the adapter matrices:

```python
# Standard FL: aggregate all parameters (millions)
# Federated LoRA: aggregate only adapter params (~160MB)

# This means:
# - Much less communication per round
# - DP noise applied to smaller parameter space
# - Base model knowledge is preserved (no catastrophic forgetting)
```

## What You Learned

- Federated LoRA fine-tunes only adapter weights, not the full LLM
- Communication cost is dramatically reduced (160MB vs 14GB)
- LLMs are vulnerable to memorisation — DP is especially important
- LoRA rank controls the capacity/communication trade-off

## Next Steps

- [Tutorial 12: Operations & Production](12-operations.md) — run FL in production
