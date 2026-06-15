"""Entity alignment wrapper for Vertical FL.

Provides a high-level interface for aligning entities across VFL parties
before split-model training begins.

Usage:
    aligner = EntityAligner(salt=os.urandom(32))
    aligned = aligner.align(
        parties={"bank_a": id_list_a, "bank_b": id_list_b, "bank_c": id_list_c}
    )
    # aligned["bank_a"] = [indices into bank_a's data]
    # aligned["bank_b"] = [indices into bank_b's data]
"""

import os
import logging
from typing import Dict, List

from psi.psi import PSIProtocol

logger = logging.getLogger("psi.alignment")


class EntityAligner:
    """Secure entity alignment for VFL using PSI.

    Args:
        salt: Per-alignment salt (bytes). If None, a random 32-byte salt is
              generated. In production, the salt should be agreed upon by all
              parties before alignment.
    """

    def __init__(self, salt: bytes = None):
        self.salt = salt or os.urandom(32)
        self.protocol = PSIProtocol(self.salt)

    def align(self, parties: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Align entities across multiple VFL parties.

        Args:
            parties: Map of party_id -> list of pseudonymised identifiers.
                     Identifiers should already be pre-processed (e.g., hashed
                     national IDs). Never pass raw direct identifiers.

        Returns:
            Map of party_id -> list of aligned indices into that party's data.
            Only records present in ALL parties are included.
        """
        party_names = list(parties.keys())
        if len(party_names) < 2:
            raise ValueError("Need at least 2 parties for alignment")

        logger.info(
            "Starting entity alignment: %d parties, record counts: %s",
            len(party_names),
            {name: len(ids) for name, ids in parties.items()}
        )

        # Hash all identifiers
        all_hashes = [self.protocol.hash_identifiers(parties[name]) for name in party_names]

        # Run PSI
        if len(party_names) == 2:
            idx_a, idx_b = PSIProtocol.intersect(all_hashes[0], all_hashes[1])
            all_indices = [idx_a, idx_b]
        else:
            all_indices = PSIProtocol.multi_party_intersect(all_hashes)

        result = {name: indices for name, indices in zip(party_names, all_indices)}

        n_aligned = len(all_indices[0]) if all_indices else 0
        logger.info(
            "Alignment complete: %d common entities across %d parties",
            n_aligned, len(party_names)
        )

        return result
