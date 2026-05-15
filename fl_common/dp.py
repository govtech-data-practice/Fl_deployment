"""
Differential Privacy for Federated Learning
=============================================
Two modes:
  - Client-side (local DP): each client clips update + adds noise before sending
  - Server-side (central DP): server clips each client update, aggregates, adds noise

Privacy accounting via Rényi DP (RDP) → (ε, δ)-DP conversion.
"""

import math
import logging
import numpy as np
from typing import List

logger = logging.getLogger("fl.dp")


# ======================================================================
# Core DP primitives
# ======================================================================

def clip_update(update: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    """Clip model update (Δw) to L2 norm ≤ max_norm."""
    flat = np.concatenate([np.atleast_1d(u).ravel() for u in update])
    norm = float(np.linalg.norm(flat))
    scale = min(1.0, max_norm / (norm + 1e-10))
    if scale < 1.0:
        return [u * scale for u in update]
    return [u.copy() for u in update]


def add_gaussian_noise(
    update: List[np.ndarray],
    noise_multiplier: float,
    max_norm: float,
    seed: int = None,
) -> List[np.ndarray]:
    """Add calibrated Gaussian noise: N(0, (noise_multiplier * max_norm)²)."""
    sigma = noise_multiplier * max_norm
    rng = np.random.RandomState(seed)
    noised = []
    for u in update:
        u = np.asarray(u)
        if u.ndim == 0:
            noised.append(u + np.float32(rng.normal(0, sigma)))
        else:
            noised.append(u + rng.normal(0, sigma, size=u.shape).astype(u.dtype))
    return noised


def clip_and_noise(
    params_before: List[np.ndarray],
    params_after: List[np.ndarray],
    max_norm: float,
    noise_multiplier: float,
    seed: int = None,
) -> List[np.ndarray]:
    """Full client-side DP: compute update, clip, add noise, reconstruct params.

    1. Δ = params_after - params_before
    2. Δ_clipped = clip(Δ, max_norm)
    3. Δ_noisy = Δ_clipped + N(0, σ²I) where σ = noise_multiplier * max_norm
    4. return params_before + Δ_noisy
    """
    delta = [a - b for a, b in zip(params_after, params_before)]
    delta = clip_update(delta, max_norm)
    delta = add_gaussian_noise(delta, noise_multiplier, max_norm, seed)
    return [b + d for b, d in zip(params_before, delta)]


# ======================================================================
# Privacy Accountant (RDP-based)
# ======================================================================

class PrivacyAccountant:
    """Track cumulative privacy loss using Rényi DP.

    Uses the analytical Gaussian mechanism RDP bound and composes
    over multiple steps, then converts to (ε, δ)-DP.
    """

    def __init__(self, noise_multiplier: float, sample_rate: float = 1.0,
                 delta: float = 1e-5):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0
        # Only use integer alpha ≥ 2 for comb() correctness
        self.alphas = list(range(2, 128))

    def step(self, num_steps: int = 1):
        self.steps += num_steps

    def _rdp_gaussian(self, alpha: int) -> float:
        """RDP of the (subsampled) Gaussian mechanism for one step."""
        if self.noise_multiplier <= 0:
            return float('inf')
        sigma2 = self.noise_multiplier ** 2

        if self.sample_rate >= 1.0:
            # Full-batch: RDP = α / (2σ²)
            return alpha / (2 * sigma2)

        # Subsampled Gaussian (Poisson sampling)
        q = self.sample_rate
        # Upper bound from Mironov et al. (2019), Proposition 3
        return math.log1p(
            q ** 2 * math.comb(alpha, 2) / sigma2
            + sum(
                q ** j * math.comb(alpha, j) * (j - 1) / (2 * sigma2) ** (j - 1)
                for j in range(3, min(alpha + 1, 20))
            )
        ) / (alpha - 1)

    def get_epsilon(self) -> float:
        """Current ε given accumulated steps."""
        if self.steps == 0:
            return 0.0
        best_eps = float('inf')
        for alpha in self.alphas:
            rdp = self._rdp_gaussian(alpha) * self.steps
            # RDP → (ε, δ)-DP conversion
            eps = rdp - math.log(self.delta) / (alpha - 1)
            if eps < best_eps:
                best_eps = eps
        return max(best_eps, 0.0)

    def __repr__(self):
        return (
            f"PrivacyAccountant(σ={self.noise_multiplier}, q={self.sample_rate}, "
            f"steps={self.steps}, ε={self.get_epsilon():.2f}, δ={self.delta})"
        )
