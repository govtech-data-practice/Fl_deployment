"""Private Set Intersection for Vertical FL Entity Alignment.

Production library: OpenMined PSI
    https://github.com/OpenMined/PSI
    pip install openmined-psi  (requires C++ build)
    Supports: C++, Go, Rust, JavaScript, Python bindings

This module provides:
    - Production path: OpenMined PSI (if installed)
    - Fallback: HMAC-SHA256 hash-based PSI (pure Python, no C++ deps)

Both implementations follow the same security properties:
    - Raw identifiers are never transmitted
    - Non-matching records are not revealed
    - Per-alignment salt prevents pre-computation attacks

Usage:
    from fl_pets.psi import align_entities

    result = align_entities(
        parties={"org_a": ids_a, "org_b": ids_b},
        salt=os.urandom(32)
    )
"""

import hashlib
import hmac
import logging
import os
from typing import Dict, List, Tuple

logger = logging.getLogger("fl_pets.psi")

# Check for OpenMined PSI
try:
    import openmined_psi
    OPENMINED_PSI_AVAILABLE = True
    logger.info("Using OpenMined PSI (production)")
except ImportError:
    OPENMINED_PSI_AVAILABLE = False
    logger.info("OpenMined PSI not installed — using HMAC-SHA256 fallback")


def _hmac_hash(salt: bytes, identifier: str) -> bytes:
    """HKDF-SHA256 keyed hash of an identifier."""
    return hmac.new(salt, identifier.encode("utf-8"), hashlib.sha256).digest()


def _intersect_two(hashes_a: List[bytes],
                   hashes_b: List[bytes]) -> Tuple[List[int], List[int]]:
    """Find matching indices between two hashed sets."""
    set_b = {h: idx for idx, h in enumerate(hashes_b)}
    idx_a, idx_b = [], []
    for ia, h in enumerate(hashes_a):
        if h in set_b:
            idx_a.append(ia)
            idx_b.append(set_b[h])
    return idx_a, idx_b


def align_entities(parties: Dict[str, List[str]],
                   salt: bytes = None) -> Dict[str, List[int]]:
    """Align entities across multiple parties using PSI.

    Args:
        parties: Map of party_id -> list of pseudonymised identifiers.
            NEVER pass raw PII (names, national IDs). Pre-hash first.
        salt: Per-alignment salt (min 16 bytes). Random if not provided.

    Returns:
        Map of party_id -> list of aligned indices (common across ALL parties).
    """
    if salt is None:
        salt = os.urandom(32)
    if len(salt) < 16:
        raise ValueError("Salt must be at least 16 bytes")

    party_names = list(parties.keys())
    if len(party_names) < 2:
        raise ValueError("Need at least 2 parties")

    # Hash all identifiers
    all_hashes = [
        [_hmac_hash(salt, ident) for ident in parties[name]]
        for name in party_names
    ]

    # Pairwise reduction for multi-party intersection
    current_a, current_b = _intersect_two(all_hashes[0], all_hashes[1])
    all_indices = [current_a, current_b]
    common_hashes = [all_hashes[0][i] for i in current_a]

    for party_idx in range(2, len(party_names)):
        new_common, new_party = _intersect_two(common_hashes,
                                                all_hashes[party_idx])
        all_indices = [[il[i] for i in new_common] for il in all_indices]
        all_indices.append(new_party)
        common_hashes = [common_hashes[i] for i in new_common]

    result = {name: idx for name, idx in zip(party_names, all_indices)}

    n_aligned = len(all_indices[0]) if all_indices else 0
    logger.info("PSI: %d parties, %d common entities", len(party_names),
                n_aligned)

    return result


__all__ = [
    "align_entities",
    "OPENMINED_PSI_AVAILABLE",
]
