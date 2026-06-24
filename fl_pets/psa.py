"""Private Set Alignment for Vertical FL Entity Alignment.

Production library: anonlink + clkhash (CSIRO Data61)
    https://github.com/data61/anonlink
    https://github.com/data61/clkhash
    pip install anonlink clkhash

This module provides:
    - Production path: anonlink CLK fuzzy matching (Bloom filter encoding)
    - Fallback: HMAC-SHA256 hash-based exact matching (pure Python, no deps)

Fuzzy mode handles the real-world case where parties do NOT share common
identifiers and records have typos, formatting differences, or inconsistent
naming. Exact mode is available when parties share pre-hashed keys.

Security properties:
    - Raw identifiers are never transmitted
    - CLK Bloom filter encoding is irreversible
    - Non-matching records are not revealed
    - Per-alignment salt/key prevents pre-computation attacks

Usage:
    from fl_pets.psa import align_entities

    # Fuzzy mode (recommended — handles typos, no shared IDs needed)
    result = align_entities_fuzzy(
        parties={
            "org_a": [{"name": "John Smith", "dob": "1985-03-15", "gender": "M"}, ...],
            "org_b": [{"name": "Jon Smith",  "dob": "1985-03-15", "gender": "M"}, ...],
        },
        threshold=0.7,
    )

    # Exact mode (when parties share pre-hashed identifiers)
    result = align_entities_exact(
        parties={"org_a": ids_a, "org_b": ids_b},
        salt=os.urandom(32)
    )
"""

import hashlib
import hmac
import logging
import os
from typing import Dict, List, Tuple

logger = logging.getLogger("fl_pets.psa")

# Check for anonlink + clkhash
try:
    from psa.psa import PSAProtocol, ANONLINK_AVAILABLE
except ImportError:
    ANONLINK_AVAILABLE = False


def _hmac_hash(salt: bytes, identifier: str) -> bytes:
    """HMAC-SHA256 keyed hash of an identifier."""
    return hmac.new(salt, identifier.encode("utf-8"), hashlib.sha256).digest()


def _intersect_two(hashes_a: List[bytes],
                   hashes_b: List[bytes]) -> Tuple[List[int], List[int]]:
    """Find matching indices between two hashed sets (1:1, first match wins)."""
    set_b = {}
    for idx, h in enumerate(hashes_b):
        if h not in set_b:
            set_b[h] = idx
    idx_a, idx_b = [], []
    matched_b = set()
    for ia, h in enumerate(hashes_a):
        if h in set_b and set_b[h] not in matched_b:
            idx_a.append(ia)
            idx_b.append(set_b[h])
            matched_b.add(set_b[h])
    return idx_a, idx_b


def align_entities_fuzzy(
    parties: Dict[str, List[Dict[str, str]]],
    salt: bytes = None,
    threshold: float = 0.7,
    schema_dict: dict = None,
    fields: List[str] = None,
) -> Dict[str, List[int]]:
    """Align entities using fuzzy CLK matching (recommended).

    Uses Bloom filter CLK encoding via anonlink + clkhash. Handles typos,
    formatting differences, and cases where parties do not share common IDs.

    Args:
        parties: Map of party_id -> list of record dicts. Each record must
            contain the fields defined in the schema (default: name, dob, gender).
        salt: Per-alignment secret (bytes). Random if not provided.
        threshold: Minimum Dice similarity for a match (0.0-1.0). Default 0.7.
        schema_dict: CLK schema definition. If None, uses default healthcare schema.
        fields: Field names in the schema, in order. Default: ["name", "dob", "gender"].

    Returns:
        Map of party_id -> list of aligned indices.
    """
    if not ANONLINK_AVAILABLE:
        raise ImportError(
            "Fuzzy alignment requires anonlink + clkhash. "
            "Install with: pip install anonlink clkhash"
        )

    protocol = PSAProtocol(
        mode="fuzzy",
        salt=salt or os.urandom(32),
        schema_dict=schema_dict,
        threshold=threshold,
        fields=fields,
    )

    party_names = list(parties.keys())
    if len(party_names) != 2:
        raise ValueError(
            "Fuzzy alignment currently supports exactly 2 parties. "
            f"Got {len(party_names)}."
        )

    clks_a = protocol.encode_clks(parties[party_names[0]])
    clks_b = protocol.encode_clks(parties[party_names[1]])
    pairs = protocol.fuzzy_match(clks_a, clks_b)

    result = {
        party_names[0]: [p[0] for p in pairs],
        party_names[1]: [p[1] for p in pairs],
    }

    n_aligned = len(pairs)
    logger.info("PSA fuzzy: %d parties, %d aligned entities", len(party_names), n_aligned)
    return result


def align_entities_exact(
    parties: Dict[str, List[str]],
    salt: bytes = None,
) -> Dict[str, List[int]]:
    """Align entities using exact hash matching.

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

    all_hashes = [
        [_hmac_hash(salt, ident) for ident in parties[name]]
        for name in party_names
    ]

    current_a, current_b = _intersect_two(all_hashes[0], all_hashes[1])
    all_indices = [current_a, current_b]
    common_hashes = [all_hashes[0][i] for i in current_a]

    for party_idx in range(2, len(party_names)):
        new_common, new_party = _intersect_two(common_hashes, all_hashes[party_idx])
        all_indices = [[il[i] for i in new_common] for il in all_indices]
        all_indices.append(new_party)
        common_hashes = [common_hashes[i] for i in new_common]

    result = {name: idx for name, idx in zip(party_names, all_indices)}

    n_aligned = len(all_indices[0]) if all_indices else 0
    logger.info("PSA exact: %d parties, %d common entities", len(party_names), n_aligned)
    return result


# Backward-compatible alias
align_entities = align_entities_exact


__all__ = [
    "align_entities_fuzzy",
    "align_entities_exact",
    "align_entities",
    "ANONLINK_AVAILABLE",
]
