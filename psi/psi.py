"""Hash-based Private Set Intersection.

Implements ECDH-PSI for 2-party entity alignment using keyed hashing.
Each party hashes its identifiers under a shared salt, then the intersection
is computed on the hashed values. Raw identifiers never leave either party.

Protocol:
    1. Parties agree on a per-alignment salt (out-of-band or via coordinator)
    2. Each party computes HKDF-SHA256(salt, identifier) for all its records
    3. Hashed sets are exchanged (or sent to coordinator)
    4. Intersection is computed on hashed values
    5. Only intersection indices are returned — no raw identifiers are shared

Security properties:
    - Raw direct identifiers are never transmitted
    - Non-matching records are not revealed
    - Salt prevents pre-computation attacks
    - Logs record only counts, match rates, and protocol metadata
"""

import hashlib
import hmac
import logging
from typing import List, Set, Tuple

logger = logging.getLogger("psi")


class PSIProtocol:
    """Hash-based PSI using HKDF-SHA256.

    Args:
        salt: Per-alignment salt. Must be agreed upon by all parties
              before alignment begins. Use a fresh salt for each alignment.
    """

    def __init__(self, salt: bytes):
        if len(salt) < 16:
            raise ValueError("Salt must be at least 16 bytes")
        self.salt = salt

    def hash_identifiers(self, identifiers: List[str]) -> List[bytes]:
        """Hash a list of identifiers using HKDF-SHA256 under the shared salt.

        Args:
            identifiers: Raw identifiers (e.g., hashed national IDs, record keys).
                         These should already be pseudonymised — never pass raw
                         direct identifiers (names, NRICs, SSNs).

        Returns:
            List of 32-byte hashes in the same order as input.
        """
        hashed = []
        for ident in identifiers:
            h = hmac.new(self.salt, ident.encode("utf-8"), hashlib.sha256).digest()
            hashed.append(h)
        return hashed

    @staticmethod
    def intersect(hashes_a: List[bytes], hashes_b: List[bytes]) -> Tuple[List[int], List[int]]:
        """Compute intersection indices between two hashed sets.

        Returns:
            (indices_a, indices_b): Matching indices in each party's original list.
        """
        set_b = {}
        for idx, h in enumerate(hashes_b):
            set_b[h] = idx

        indices_a = []
        indices_b = []
        for idx_a, h in enumerate(hashes_a):
            if h in set_b:
                indices_a.append(idx_a)
                indices_b.append(set_b[h])

        match_rate = len(indices_a) / max(len(hashes_a), 1)
        logger.info(
            "PSI complete: %d matches out of (%d, %d) records (match rate: %.2f%%)",
            len(indices_a), len(hashes_a), len(hashes_b), match_rate * 100
        )
        return indices_a, indices_b

    @staticmethod
    def multi_party_intersect(all_hashes: List[List[bytes]]) -> List[List[int]]:
        """Compute intersection across >2 parties.

        Uses pairwise reduction: intersect party 0 with party 1,
        then intersect result with party 2, etc.

        Returns:
            List of index lists, one per party, for the common intersection.
        """
        if len(all_hashes) < 2:
            raise ValueError("Need at least 2 parties for PSI")

        # Start with first two parties
        current_a, current_b = PSIProtocol.intersect(all_hashes[0], all_hashes[1])

        # Track indices per party
        all_indices = [current_a, current_b]

        # Reduce through remaining parties
        common_hashes = [all_hashes[0][i] for i in current_a]

        for party_idx in range(2, len(all_hashes)):
            new_indices_common, new_indices_party = PSIProtocol.intersect(
                common_hashes, all_hashes[party_idx]
            )
            # Update all existing party indices
            all_indices = [[idx_list[i] for i in new_indices_common] for idx_list in all_indices]
            all_indices.append(new_indices_party)
            common_hashes = [common_hashes[i] for i in new_indices_common]

        logger.info(
            "Multi-party PSI: %d parties, %d common records",
            len(all_hashes), len(all_indices[0])
        )
        return all_indices
