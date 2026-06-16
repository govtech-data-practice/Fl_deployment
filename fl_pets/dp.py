"""Differential Privacy for Federated Learning.

Library: Opacus 1.6+ (Meta)
    https://opacus.ai/
    pip install opacus

Opacus provides:
    - PrivacyEngine: wraps a PyTorch model+optimizer for DP-SGD training
    - RDPAccountant: tracks cumulative privacy loss via Renyi DP
    - Per-sample gradient clipping and calibrated Gaussian noise
    - Automatic batch memory management for large models

FL-specific presets:
    DP_STRONG:   sigma=1.5, max_norm=1.0  (strongest privacy)
    DP_MODERATE: sigma=0.8, max_norm=1.0  (balanced)
    DP_RELAXED:  sigma=0.5, max_norm=1.0  (weaker privacy, better utility)

    Tasks without a preset fail closed to DP_STRONG.
"""

from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.validators import ModuleValidator

# ── Named presets for FL deployments ────────────────────────────────

DP_PRESETS = {
    "DP_STRONG":   {"noise_multiplier": 1.5, "max_grad_norm": 1.0},
    "DP_MODERATE": {"noise_multiplier": 0.8, "max_grad_norm": 1.0},
    "DP_RELAXED":  {"noise_multiplier": 0.5, "max_grad_norm": 1.0},
}

DEFAULT_PRESET = "DP_STRONG"


def get_preset(name=None):
    """Return DP config for a named preset. Fail-closed to DP_STRONG."""
    if name is None or name not in DP_PRESETS:
        name = DEFAULT_PRESET
    cfg = DP_PRESETS[name].copy()
    cfg["preset"] = name
    return cfg


def make_private(model, optimizer, data_loader, preset="DP_STRONG",
                 noise_multiplier=None, max_grad_norm=None, epochs=1,
                 delta=1e-5):
    """Wrap a PyTorch model+optimizer with Opacus PrivacyEngine.

    Args:
        model: PyTorch nn.Module.
        optimizer: PyTorch optimizer.
        data_loader: Training DataLoader.
        preset: Named DP preset (DP_STRONG, DP_MODERATE, DP_RELAXED).
        noise_multiplier: Override preset sigma.
        max_grad_norm: Override preset clipping norm.
        epochs: Number of training epochs (for budget calculation).
        delta: Target delta for (epsilon, delta)-DP.

    Returns:
        (model, optimizer, data_loader, privacy_engine) — all wrapped for DP-SGD.
    """
    cfg = get_preset(preset)
    sigma = noise_multiplier or cfg["noise_multiplier"]
    clip = max_grad_norm or cfg["max_grad_norm"]

    # Validate model compatibility with Opacus
    model = ModuleValidator.fix(model)

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=sigma,
        max_grad_norm=clip,
    )

    return model, optimizer, data_loader, privacy_engine


def compute_epsilon(noise_multiplier, sample_rate, steps, delta=1e-5,
                    alphas=None):
    """Compute epsilon using Opacus RDP accountant.

    Args:
        noise_multiplier: Gaussian noise sigma.
        sample_rate: Batch size / dataset size.
        steps: Number of training steps (batches, not epochs).
        delta: Target delta.
        alphas: RDP orders (default: Opacus defaults).

    Returns:
        epsilon: The (epsilon, delta)-DP guarantee.
    """
    accountant = RDPAccountant()
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    return accountant.get_epsilon(delta=delta)


__all__ = [
    "PrivacyEngine",
    "RDPAccountant",
    "ModuleValidator",
    "DP_PRESETS",
    "DEFAULT_PRESET",
    "get_preset",
    "make_private",
    "compute_epsilon",
]
