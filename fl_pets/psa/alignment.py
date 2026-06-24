"""Entity alignment wrapper for Vertical FL.

Provides a high-level interface for aligning entities across VFL parties
before split-model training begins. Supports both fuzzy matching (CLK
Bloom filters via anonlink) and exact matching (HMAC hash intersection).

Usage (fuzzy — recommended for real-world data):
    aligner = EntityAligner(mode="fuzzy")
    aligned = aligner.align_fuzzy(
        parties={
            "hospital_a": [{"name": "John Smith", "dob": "1985-03-15", "gender": "M"}, ...],
            "hospital_b": [{"name": "Jon Smith",  "dob": "1985-03-15", "gender": "M"}, ...],
        }
    )

Usage (exact — when parties share common identifiers):
    aligner = EntityAligner(mode="exact", salt=os.urandom(32))
    aligned = aligner.align_exact(
        parties={"bank_a": id_list_a, "bank_b": id_list_b}
    )
"""

import os
import logging
from typing import Dict, List

from fl_pets.psa.protocol import PSAProtocol

logger = logging.getLogger("psa.alignment")


class EntityAligner:
    """Secure entity alignment for VFL using PSA.

    Args:
        mode: "fuzzy" (default) for CLK-based alignment, "exact" for hash-based.
        salt: Per-alignment salt (bytes). If None, a random 32-byte salt is generated.
        schema_dict: CLK schema definition (fuzzy mode only). If None, uses the
                     default healthcare schema (name, dob, gender).
        threshold: Minimum Dice similarity for a fuzzy match (0.0-1.0). Default 0.7.
        fields: List of field names for CLK encoding. Default: ["name", "dob", "gender"].
    """

    def __init__(
        self,
        mode: str = "fuzzy",
        salt: bytes = None,
        schema_dict: dict = None,
        threshold: float = 0.7,
        fields: List[str] = None,
    ):
        self.salt = salt or os.urandom(32)
        self.protocol = PSAProtocol(
            mode=mode,
            salt=self.salt,
            schema_dict=schema_dict,
            threshold=threshold,
            fields=fields,
        )

    def align_fuzzy(
        self, parties: Dict[str, List[Dict[str, str]]]
    ) -> Dict[str, List[int]]:
        """Align entities using fuzzy CLK matching (recommended).

        Args:
            parties: Map of party_id -> list of record dicts. Each record
                     must contain the fields defined in the schema (default:
                     name, dob, gender).

        Returns:
            Map of party_id -> list of aligned indices into that party's data.
        """
        party_names = list(parties.keys())
        if len(party_names) != 2:
            raise ValueError(
                "Fuzzy alignment currently supports exactly 2 parties. "
                f"Got {len(party_names)}."
            )

        logger.info(
            "Starting fuzzy entity alignment: %d parties, record counts: %s",
            len(party_names),
            {name: len(recs) for name, recs in parties.items()}
        )

        # Encode CLKs for each party
        clks_a = self.protocol.encode_clks(parties[party_names[0]])
        clks_b = self.protocol.encode_clks(parties[party_names[1]])

        # Run fuzzy matching
        pairs = self.protocol.fuzzy_match(clks_a, clks_b)

        idx_a = [p[0] for p in pairs]
        idx_b = [p[1] for p in pairs]
        result = {party_names[0]: idx_a, party_names[1]: idx_b}

        logger.info(
            "Fuzzy alignment complete: %d matched entities across %d parties",
            len(pairs), len(party_names)
        )
        return result

    def align_exact(self, parties: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """Align entities using exact hash matching.

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
            "Starting exact entity alignment: %d parties, record counts: %s",
            len(party_names),
            {name: len(ids) for name, ids in parties.items()}
        )

        all_hashes = [self.protocol.hash_identifiers(parties[name]) for name in party_names]

        if len(party_names) == 2:
            idx_a, idx_b = PSAProtocol.intersect(all_hashes[0], all_hashes[1])
            all_indices = [idx_a, idx_b]
        else:
            all_indices = PSAProtocol.multi_party_intersect(all_hashes)

        result = {name: indices for name, indices in zip(party_names, all_indices)}

        n_aligned = len(all_indices[0]) if all_indices else 0
        logger.info(
            "Exact alignment complete: %d common entities across %d parties",
            n_aligned, len(party_names)
        )
        return result
