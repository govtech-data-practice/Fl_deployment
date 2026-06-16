"""
Federated Adapter Framework
============================
Generic framework for federated fine-tuning of any HuggingFace model using LoRA adapters.

Architecture (inspired by FlexOLMo, AI2):
    1. Each client (agency) holds private data and trains a LoRA adapter locally
    2. Only adapter weights (~0.1-1% of model params) are transmitted
    3. Server aggregates adapter weights across clients
    4. Base model is frozen — never transmitted, identical on all nodes

This pattern works with ANY model:
    - LLMs: OLMo, Llama, Mistral, Phi, Qwen, GPT-NeoX
    - Vision: ViT, DINOv2, CLIP
    - Speech: Whisper
    - Multimodal: LLaVA, Florence
    - Custom: any nn.Module via manual LoRA injection

Usage:
    # Config-driven — change model by editing config, no code changes
    config = AdapterConfig(
        model_id="allenai/OLMo-1B-hf",
        task_type="causal_lm",
        lora_rank=8,
        quantize_bits=4,
    )
    client = FederatedAdapterClient(config, partition_id=0, num_clients=5)

    # Works with Flower
    fl.client.start_client(server_address="...", client=client)
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import torch

logger = logging.getLogger("fl.adapter")


# ── Configuration ───────────────────────────────────────────────────

@dataclass
class AdapterConfig:
    """Configuration for federated adapter training.

    Change model_id to switch between any HuggingFace model.
    All other params have sensible defaults.
    """
    # Model
    model_id: str = "allenai/OLMo-1B-hf"
    task_type: str = "causal_lm"        # causal_lm, seq2seq, token_cls, seq_cls, image_cls
    trust_remote_code: bool = True

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # None = auto-detect

    # Quantization
    quantize_bits: int = 4              # 0 = no quantization, 4 = QLoRA, 8 = 8-bit
    compute_dtype: str = "float16"

    # Training
    max_seq_len: int = 256
    learning_rate: float = 2e-4
    batch_size: int = 4
    local_epochs: int = 1
    max_grad_norm: float = 1.0

    # Data
    max_samples: int = 500

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "AdapterConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "AdapterConfig":
        """Load config from environment variables (for Docker containers)."""
        return cls(
            model_id=os.environ.get("ADAPTER_MODEL_ID", "allenai/OLMo-1B-hf"),
            task_type=os.environ.get("ADAPTER_TASK_TYPE", "causal_lm"),
            lora_rank=int(os.environ.get("ADAPTER_LORA_RANK", "8")),
            lora_alpha=int(os.environ.get("ADAPTER_LORA_ALPHA", "16")),
            quantize_bits=int(os.environ.get("ADAPTER_QUANTIZE_BITS", "4")),
            max_seq_len=int(os.environ.get("ADAPTER_MAX_SEQ_LEN", "256")),
            learning_rate=float(os.environ.get("ADAPTER_LR", "2e-4")),
            batch_size=int(os.environ.get("ADAPTER_BATCH_SIZE", "4")),
            max_samples=int(os.environ.get("MAX_SAMPLES", "500")),
        )


# ── Presets for common models ───────────────────────────────────────

PRESETS = {
    # LLMs
    "olmo-1b": AdapterConfig(model_id="allenai/OLMo-1B-hf", task_type="causal_lm", lora_rank=8, quantize_bits=4),
    "olmo-7b": AdapterConfig(model_id="allenai/OLMo-7B-hf", task_type="causal_lm", lora_rank=8, quantize_bits=4),
    "llama-3-8b": AdapterConfig(model_id="meta-llama/Meta-Llama-3-8B", task_type="causal_lm", lora_rank=8, quantize_bits=4),
    "mistral-7b": AdapterConfig(model_id="mistralai/Mistral-7B-v0.3", task_type="causal_lm", lora_rank=8, quantize_bits=4),
    "phi-3-mini": AdapterConfig(model_id="microsoft/Phi-3-mini-4k-instruct", task_type="causal_lm", lora_rank=8, quantize_bits=4),
    "qwen2-1.5b": AdapterConfig(model_id="Qwen/Qwen2-1.5B", task_type="causal_lm", lora_rank=8, quantize_bits=0),

    # Vision
    "vit-base": AdapterConfig(model_id="google/vit-base-patch16-224", task_type="image_cls", lora_rank=4, quantize_bits=0,
                              lora_target_modules=["query", "value"]),
    "dinov2-base": AdapterConfig(model_id="facebook/dinov2-base", task_type="image_cls", lora_rank=4, quantize_bits=0,
                                 lora_target_modules=["query", "value"]),

    # Speech
    "whisper-small": AdapterConfig(model_id="openai/whisper-small", task_type="seq2seq", lora_rank=4, quantize_bits=0,
                                   lora_target_modules=["q_proj", "v_proj"]),

    # Text classification
    "bert-base": AdapterConfig(model_id="bert-base-uncased", task_type="seq_cls", lora_rank=4, quantize_bits=0,
                               lora_target_modules=["query", "value"]),
    "biobert": AdapterConfig(model_id="dmis-lab/biobert-v1.1", task_type="token_cls", lora_rank=4, quantize_bits=0,
                             lora_target_modules=["query", "value"]),
}


# ── Model Loading ───────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {}


def load_model(config: AdapterConfig):
    """Load a HuggingFace model with LoRA adapter.

    Returns (model, tokenizer/processor).
    Cached — safe to call multiple times.
    """
    cache_key = f"{config.model_id}_{config.lora_rank}_{config.quantize_bits}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        raise ImportError(
            "Missing dependencies. Install: pip install transformers peft accelerate bitsandbytes"
        )

    logger.info("Loading %s (LoRA r=%d, %d-bit)...", config.model_id, config.lora_rank, config.quantize_bits)

    # Quantization
    bnb_config = None
    if config.quantize_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
    elif config.quantize_bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model based on task type
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "device_map": "auto",
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    if config.task_type in ("causal_lm",):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    elif config.task_type in ("seq2seq",):
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_id, **model_kwargs)
    elif config.task_type in ("seq_cls",):
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(config.model_id, **model_kwargs)
    elif config.task_type in ("token_cls",):
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(config.model_id, **model_kwargs)
    elif config.task_type in ("image_cls",):
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(config.model_id, **model_kwargs)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(config.model_id, **model_kwargs)

    # LoRA task type mapping
    peft_task_map = {
        "causal_lm": TaskType.CAUSAL_LM,
        "seq2seq": TaskType.SEQ_2_SEQ_LM,
        "seq_cls": TaskType.SEQ_CLS,
        "token_cls": TaskType.TOKEN_CLS,
    }

    # Auto-detect target modules if not specified
    target_modules = config.lora_target_modules
    if target_modules is None:
        target_modules = _detect_target_modules(model)

    lora_config = LoraConfig(
        task_type=peft_task_map.get(config.task_type),
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("  Trainable: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # Tokenizer / processor
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _model_cache[cache_key] = (model, tokenizer)
    return model, tokenizer


def _detect_target_modules(model) -> List[str]:
    """Auto-detect LoRA target modules from model architecture."""
    module_names = set()
    for name, _ in model.named_modules():
        # Common attention projection names across architectures
        for target in ["q_proj", "k_proj", "v_proj", "o_proj",  # LLaMA/OLMo/Mistral
                        "query", "key", "value",                   # BERT/ViT
                        "qkv",                                      # some ViT variants
                        "q_proj", "v_proj"]:                       # Whisper
            if name.endswith(target):
                module_names.add(target)
    if not module_names:
        # Fallback: target all linear layers
        module_names = {"q_proj", "v_proj"}
        logger.warning("Could not auto-detect LoRA targets, using default: %s", module_names)
    logger.info("  LoRA targets: %s", sorted(module_names))
    return sorted(module_names)


# ── Federated Adapter Client ───────────────────────────────────────

class FederatedAdapterClient:
    """Generic Flower client for federated adapter training.

    Works with any HuggingFace model. Just change the config.

    The pattern:
        1. get_parameters() → returns only trainable LoRA weights (tiny)
        2. set_parameters() → loads aggregated LoRA weights from server
        3. fit() → fine-tunes LoRA on local data
        4. evaluate() → evaluates on local validation data

    Base model weights are FROZEN and never transmitted.
    """

    def __init__(self, config: AdapterConfig, partition_id: int, num_clients: int):
        self.config = config
        self.pid = partition_id
        self.n_clients = num_clients
        self._train_data = None
        self._val_data = None

    def load_data(self, train_texts: list, val_texts: list):
        """Set training and validation data. Called by the task-specific wrapper."""
        self._train_data = train_texts
        self._val_data = val_texts

    def get_parameters(self) -> list:
        """Return only LoRA adapter weights (not the frozen base model)."""
        model, _ = load_model(self.config)
        return [p.detach().cpu().numpy() for p in model.parameters() if p.requires_grad]

    def set_parameters(self, params: list):
        """Set LoRA adapter weights. Sanitizes NaN/Inf from DP noise."""
        model, _ = load_model(self.config)
        trainable = [p for p in model.parameters() if p.requires_grad]
        for p, new_val in zip(trainable, params):
            t = torch.tensor(new_val, device=p.device, dtype=p.dtype)
            if not torch.isfinite(t).all():
                t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
            p.data.copy_(t)

    def fit(self, parameters: list, config: dict) -> tuple:
        """Fine-tune LoRA on local data."""
        if self._train_data is None:
            raise ValueError("No training data. Call load_data() first.")

        model, tokenizer = load_model(self.config)
        self.set_parameters(parameters)

        lr = float(config.get("learning_rate", self.config.learning_rate))
        epochs = int(config.get("local_epochs", self.config.local_epochs))
        batch_size = int(config.get("batch_size", self.config.batch_size))

        # Tokenize
        encodings = tokenizer(
            self._train_data,
            truncation=True,
            max_length=self.config.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        if self.config.task_type == "causal_lm":
            encodings["labels"] = encodings["input_ids"].clone()

        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            encodings.get("labels", encodings["input_ids"]),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
        model.train()
        total_loss, n_batches = 0.0, 0

        for epoch in range(epochs):
            for batch in loader:
                input_ids, attention_mask, labels = [b.to(model.device) for b in batch]
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                total_loss += outputs.loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return self.get_parameters(), len(self._train_data), {"loss": avg_loss}

    def evaluate(self, parameters: list, config: dict) -> tuple:
        """Evaluate on local validation data."""
        if self._val_data is None:
            return 0.0, 0, {"perplexity": 0.0}

        model, tokenizer = load_model(self.config)
        self.set_parameters(parameters)

        encodings = tokenizer(
            self._val_data,
            truncation=True,
            max_length=self.config.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        if self.config.task_type == "causal_lm":
            encodings["labels"] = encodings["input_ids"].clone()

        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            encodings.get("labels", encodings["input_ids"]),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size)

        model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, labels = [b.to(model.device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        perplexity = min(np.exp(avg_loss), 10000.0)
        return avg_loss, len(self._val_data), {"perplexity": perplexity, "loss": avg_loss}


# ── Adapter Size Calculator ─────────────────────────────────────────

def estimate_adapter_size(config: AdapterConfig) -> dict:
    """Estimate adapter size without loading the model.

    Useful for capacity planning — how much data is transmitted per round.
    """
    # Rough estimates based on model architecture
    model_sizes = {
        "1b": 1_000_000_000,
        "1.5b": 1_500_000_000,
        "3b": 3_000_000_000,
        "7b": 7_000_000_000,
        "8b": 8_000_000_000,
        "13b": 13_000_000_000,
    }

    # Extract model size from ID
    total_params = 0
    model_id_lower = config.model_id.lower()
    for size_str, params in model_sizes.items():
        if size_str in model_id_lower:
            total_params = params
            break
    if total_params == 0:
        total_params = 500_000_000  # default 500M

    # LoRA trainable params ≈ 2 * rank * hidden_dim * num_target_layers
    # Roughly 0.1-0.5% of total params for rank 8
    trainable_ratio = config.lora_rank / 1000  # rough approximation
    trainable_params = int(total_params * trainable_ratio)

    # Size in bytes (float32 = 4 bytes per param)
    adapter_bytes = trainable_params * 4
    base_bytes = total_params * (config.quantize_bits / 8 if config.quantize_bits > 0 else 4)

    return {
        "model_id": config.model_id,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100 * trainable_params / total_params,
        "adapter_size_mb": adapter_bytes / 1e6,
        "base_model_size_gb": base_bytes / 1e9,
        "per_round_transfer_mb": adapter_bytes * 2 / 1e6,  # upload + download
        "lora_rank": config.lora_rank,
        "quantize_bits": config.quantize_bits,
    }


# ── CLI ─────────────────────────────────────────────────────────────

def print_presets():
    """Print all available model presets with size estimates."""
    print("Available Model Presets")
    print("=" * 80)
    print(f"{'Preset':<20} {'Model ID':<40} {'Adapter MB':<12} {'Base GB':<10}")
    print("-" * 80)
    for name, config in sorted(PRESETS.items()):
        est = estimate_adapter_size(config)
        print(f"{name:<20} {config.model_id:<40} {est['adapter_size_mb']:.1f} MB     {est['base_model_size_gb']:.1f} GB")


if __name__ == "__main__":
    print_presets()
