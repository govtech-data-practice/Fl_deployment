"""SecAgg+: pairwise deterministic masks that cancel across all clients."""

import hashlib
import numpy as np


def _pair_seed(round_seed: int, i: int, j: int, layer: int = 0) -> int:
    """Deterministic seed for (round, pair, layer). Always in [0, 2^32)."""
    lo, hi = min(i, j), max(i, j)
    raw = f"{round_seed}:{lo}:{hi}:{layer}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def secagg_mask_parameters(params, client_id, num_clients, round_seed, scale=0.01):
    """Add pairwise masks that cancel when summed across all N clients.

    For pair (i, j): shared RNG produces mask M.
    Smaller ID adds +M, larger adds -M. Sum across all clients = 0.
    """
    masked = []
    for li, p in enumerate(params):
        p = np.asarray(p, dtype=np.float32)  # ensure ndarray (handles scalar params)
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
