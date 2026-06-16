"""Secure Aggregation for Federated Learning.

Library: Flower SecAgg+ — https://flower.ai/
    pip install flwr

Flower provides native SecAgg+ as part of its framework:
    - Pairwise masking with automatic key agreement
    - Dropout tolerance (clients can fail mid-round)
    - Integrated with Flower's ServerApp/ClientApp lifecycle

For standalone use outside Flower, this module also exposes
a lightweight SecAgg implementation for testing and demonstration.

Usage with Flower (production):
    from flwr.server.strategy import FedAvg
    # Flower handles SecAgg internally when configured

Usage standalone (testing):
    from fl_pets.secagg import mask_parameters, verify_cancellation
"""

import hashlib
import numpy as np
from typing import List

try:
    from flwr.common.secure_aggregation import (
        SecureAggregation,
    )
    FLOWER_SECAGG_AVAILABLE = True
except (ImportError, AttributeError):
    FLOWER_SECAGG_AVAILABLE = False


def _pair_seed(round_seed: int, i: int, j: int, layer: int = 0) -> int:
    """Deterministic seed for a (round, client pair, layer) tuple."""
    lo, hi = min(i, j), max(i, j)
    raw = f"{round_seed}:{lo}:{hi}:{layer}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def mask_parameters(params: List[np.ndarray], client_id: int,
                    num_clients: int, round_seed: int,
                    scale: float = 0.01) -> List[np.ndarray]:
    """Apply SecAgg+ pairwise masking to model parameters.

    For each pair (i, j): a shared RNG produces mask M.
    The client with the smaller ID adds +M, the larger adds -M.
    When the server sums all masked updates, masks cancel exactly.

    Args:
        params: Model parameters as numpy arrays.
        client_id: This client's index (0-based).
        num_clients: Total participating clients.
        round_seed: Shared seed for this round.
        scale: Mask magnitude.

    Returns:
        Masked parameters. Sum across all clients = sum of originals.
    """
    masked = []
    for li, p in enumerate(params):
        p = np.asarray(p, dtype=np.float32)
        total = np.zeros_like(p)
        for oid in range(num_clients):
            if oid == client_id:
                continue
            seed = _pair_seed(round_seed, client_id, oid, li)
            rng = np.random.RandomState(seed)
            if p.ndim == 0:
                m = np.float32(rng.randn() * scale)
            else:
                m = rng.randn(*p.shape).astype(np.float32) * scale
            total += m if client_id < oid else -m
        masked.append(p + total)
    return masked


def verify_cancellation(params: List[np.ndarray], num_clients: int,
                        round_seed: int, scale: float = 0.01) -> dict:
    """Verify that SecAgg masks cancel correctly across all clients.

    Returns:
        Dict with max_error and per-client masked values.
    """
    all_masked = []
    for cid in range(num_clients):
        m = mask_parameters(params, cid, num_clients, round_seed, scale)
        all_masked.append(m)

    # Sum across clients
    agg = [sum(all_masked[c][l] for c in range(num_clients))
           for l in range(len(params))]
    expected = [p * num_clients for p in params]

    max_err = max(np.max(np.abs(a - e)) for a, e in zip(agg, expected))

    return {
        "max_error": float(max_err),
        "aggregate": agg,
        "expected": expected,
        "num_clients": num_clients,
        "flower_secagg_available": FLOWER_SECAGG_AVAILABLE,
    }


__all__ = [
    "mask_parameters",
    "verify_cancellation",
    "FLOWER_SECAGG_AVAILABLE",
]
