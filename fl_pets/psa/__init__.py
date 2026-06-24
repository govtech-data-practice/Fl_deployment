"""Private Set Alignment (PSA) for Vertical FL Entity Alignment.

Production library: anonlink + clkhash (CSIRO Data61)
Fallback: HMAC-SHA256 hash-based exact matching (pure Python)

Usage:
    from fl_pets.psa import align_entities_fuzzy, align_entities_exact
    from fl_pets.psa import PSAProtocol, EntityAligner
"""

import hashlib
import hmac
import logging
import os
from typing import Dict, List, Tuple

from fl_pets.psa.protocol import PSAProtocol, ANONLINK_AVAILABLE
from fl_pets.psa.alignment import EntityAligner

logger = logging.getLogger("fl_pets.psa")


def _hmac_hash(salt: bytes, identifier: str) -> bytes:
    return hmac.new(salt, identifier.encode("utf-8"), hashlib.sha256).digest()


def _intersect_two(hashes_a: List[bytes],
                   hashes_b: List[bytes]) -> Tuple[List[int], List[int]]:
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
    """Align entities using fuzzy CLK matching (recommended)."""
    if not ANONLINK_AVAILABLE:
        raise ImportError("Fuzzy alignment requires anonlink + clkhash. "
                          "Install with: pip install anonlink clkhash")

    protocol = PSAProtocol(
        mode="fuzzy", salt=salt or os.urandom(32),
        schema_dict=schema_dict, threshold=threshold, fields=fields,
    )
    party_names = list(parties.keys())
    if len(party_names) != 2:
        raise ValueError(f"Fuzzy alignment currently supports exactly 2 parties. Got {len(party_names)}.")

    clks_a = protocol.encode_clks(parties[party_names[0]])
    clks_b = protocol.encode_clks(parties[party_names[1]])
    pairs = protocol.fuzzy_match(clks_a, clks_b)

    result = {party_names[0]: [p[0] for p in pairs], party_names[1]: [p[1] for p in pairs]}
    logger.info("PSA fuzzy: %d parties, %d aligned entities", len(party_names), len(pairs))
    return result


def align_entities_exact(
    parties: Dict[str, List[str]],
    salt: bytes = None,
) -> Dict[str, List[int]]:
    """Align entities using exact hash matching."""
    if salt is None:
        salt = os.urandom(32)
    if len(salt) < 16:
        raise ValueError("Salt must be at least 16 bytes")

    party_names = list(parties.keys())
    if len(party_names) < 2:
        raise ValueError("Need at least 2 parties")

    all_hashes = [[_hmac_hash(salt, ident) for ident in parties[name]] for name in party_names]
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


align_entities = align_entities_exact

__all__ = [
    "PSAProtocol", "EntityAligner", "ANONLINK_AVAILABLE",
    "align_entities_fuzzy", "align_entities_exact", "align_entities",
]
