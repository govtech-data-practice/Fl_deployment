"""Private Set Alignment using CLK Bloom filters (anonlink + clkhash).

Supports two alignment modes:
    - **Fuzzy (CLK):** Bloom filter encoding of quasi-identifiers (name, DOB, etc.)
      with similarity-based matching. Handles typos, formatting differences, and
      missing shared IDs. Requires anonlink + clkhash.
    - **Exact (HMAC):** Hash-based exact matching on pre-hashed identifiers.
      Faster but requires shared identifier keys across parties.

Protocol (fuzzy mode):
    1. Parties agree on a linkage schema (which fields, n-gram size, Bloom filter length)
    2. Each party encodes its records into CLK hashes (Cryptographic Longterm Keys)
    3. CLK hashes are compared using Dice coefficient similarity
    4. Greedy 1:1 matching above a threshold produces aligned indices
    5. Raw quasi-identifiers never leave either party

Protocol (exact mode):
    1. Parties agree on a per-alignment salt
    2. Each party computes HMAC-SHA256(salt, identifier) for all records
    3. Intersection is computed on hashed values
    4. Only intersection indices are returned

Security properties:
    - Raw direct identifiers are never transmitted
    - Non-matching records are not revealed (fuzzy mode uses similarity threshold)
    - CLK encoding is irreversible (one-way Bloom filter hashing)
    - Per-alignment salt/key prevents pre-computation attacks
"""

import csv
import hashlib
import hmac
import logging
from io import StringIO
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("psa")

# Check for anonlink + clkhash
try:
    from clkhash import clk
    from clkhash.schema import from_json_dict as _schema_from_json_dict
    import anonlink
    ANONLINK_AVAILABLE = True
    logger.info("anonlink + clkhash available (CLK fuzzy matching enabled)")
except ImportError:
    ANONLINK_AVAILABLE = False
    logger.info("anonlink/clkhash not installed — fuzzy matching unavailable, exact mode only")


# ----- Default CLK schema for healthcare quasi-identifiers -----

def _make_default_schema(salt_b64: str = None, info_b64: str = None) -> dict:
    """Build a default CLK schema with per-session KDF parameters.

    Args:
        salt_b64: Base64-encoded KDF salt. If None, generates a random 16-byte salt.
        info_b64: Base64-encoded KDF info. If None, generates a random 8-byte info.
    """
    import base64 as _b64
    if salt_b64 is None:
        salt_b64 = _b64.b64encode(__import__("os").urandom(16)).decode()
    if info_b64 is None:
        info_b64 = _b64.b64encode(__import__("os").urandom(8)).decode()
    return {
        "version": 3,
        "clkConfig": {
            "l": 1024,
            "xor_folds": 1,  # XOR folding: basic countermeasure against frequency attacks
            "kdf": {
                "type": "HKDF",
                "hash": "SHA256",
                "info": info_b64,
                "salt": salt_b64,
                "keySize": 64,
            },
        },
        "features": [
            {
                "identifier": "name",
                "format": {"type": "string", "encoding": "utf-8"},
                "hashing": {
                    "comparison": {"type": "ngram", "n": 2},
                    "strategy": {"bitsPerFeature": 200},
                },
            },
            {
                "identifier": "dob",
                "format": {"type": "string", "encoding": "utf-8"},
                "hashing": {
                    "comparison": {"type": "ngram", "n": 1},
                    "strategy": {"bitsPerFeature": 100},
                },
            },
            {
                "identifier": "gender",
                "format": {"type": "enum", "values": ["M", "F", "O", "U"]},
                "hashing": {
                    "comparison": {"type": "exact"},
                    "strategy": {"bitsPerFeature": 50},
                },
            },
        ],
    }


# Backward-compatible constant (WARNING: uses fixed KDF params — for tests only)
DEFAULT_CLK_SCHEMA = _make_default_schema(
    salt_b64="c2VjdXJlLXNhbHQ=",  # base64("secure-salt")
    info_b64="cHNhLWhlYWx0aGNhcmU=",  # base64("psa-healthcare")
)


class PSAProtocol:
    """Private Set Alignment protocol.

    Supports fuzzy matching (CLK Bloom filters via anonlink) and
    exact matching (HMAC-SHA256 hash intersection) modes.

    Args:
        mode: "fuzzy" (default) for CLK-based alignment, "exact" for hash-based.
        salt: Per-alignment salt (bytes). Required for exact mode. For fuzzy mode,
              the secret key is derived from this salt.
        schema_dict: CLK schema definition (fuzzy mode only). If None, uses the
                     default healthcare schema (name, dob, gender fields).
        threshold: Minimum Dice similarity for a fuzzy match (0.0-1.0).
                   Default 0.7. Lower = more matches but more false positives.
        fields: List of field names in the CLK schema, in order. Used to extract
                values from record dicts. Default: ["name", "dob", "gender"].
    """

    def __init__(
        self,
        mode: str = "fuzzy",
        salt: bytes = None,
        schema_dict: dict = None,
        threshold: float = 0.7,
        fields: List[str] = None,
    ):
        if mode not in ("fuzzy", "exact"):
            raise ValueError(f"mode must be 'fuzzy' or 'exact', got '{mode}'")

        if mode == "fuzzy" and not ANONLINK_AVAILABLE:
            raise ImportError(
                "Fuzzy mode requires anonlink + clkhash. "
                "Install with: pip install anonlink clkhash"
            )

        self.mode = mode
        self.threshold = threshold
        self.fields = fields or ["name", "dob", "gender"]

        if mode == "exact":
            self.salt = salt or __import__("os").urandom(32)
            if len(self.salt) < 16:
                raise ValueError("Salt must be at least 16 bytes")
        else:
            self.salt = salt or __import__("os").urandom(32)
            self._schema = _schema_from_json_dict(schema_dict or DEFAULT_CLK_SCHEMA)
            self._secret_key = self.salt.hex()

    # ---------- Fuzzy (CLK) mode ----------

    def encode_clks(self, records: List[Dict[str, str]]) -> list:
        """Encode records into CLK Bloom filter hashes.

        Args:
            records: List of dicts with keys matching self.fields.

        Returns:
            List of CLK bitarray hashes.
        """
        if self.mode != "fuzzy":
            raise RuntimeError("encode_clks() is only available in fuzzy mode")
        if not records:
            return []

        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(self.fields)
        for r in records:
            writer.writerow([r.get(f, "") for f in self.fields])
        buf.seek(0)
        return clk.generate_clk_from_csv(buf, self._secret_key, self._schema)

    def fuzzy_match(
        self, clks_a: list, clks_b: list
    ) -> List[Tuple[int, int]]:
        """Match two sets of CLKs using Dice similarity.

        Returns:
            List of (index_a, index_b) matched pairs.
        """
        if self.mode != "fuzzy":
            raise RuntimeError("fuzzy_match() is only available in fuzzy mode")

        results = anonlink.candidate_generation.find_candidate_pairs(
            [clks_a, clks_b],
            anonlink.similarities.dice_coefficient_accelerated,
            self.threshold,
        )
        solution = anonlink.solving.greedy_solve(results)

        pairs = []
        for group in solution:
            records_by_dataset = {}
            for dataset_idx, record_idx in group:
                records_by_dataset[dataset_idx] = record_idx
            if 0 in records_by_dataset and 1 in records_by_dataset:
                pairs.append((records_by_dataset[0], records_by_dataset[1]))

        logger.info(
            "PSA fuzzy match: %d pairs (threshold=%.2f)",
            len(pairs), self.threshold
        )
        return pairs

    # ---------- Exact (HMAC) mode ----------

    def hash_identifiers(self, identifiers: List[str]) -> List[bytes]:
        """Hash identifiers using HMAC-SHA256 under the shared salt (exact mode).

        Args:
            identifiers: Pre-hashed or pseudonymised identifiers.

        Returns:
            List of 32-byte HMAC hashes.
        """
        if self.mode != "exact":
            raise RuntimeError("hash_identifiers() is only available in exact mode")
        return [
            hmac.new(self.salt, ident.encode("utf-8"), hashlib.sha256).digest()
            for ident in identifiers
        ]

    @staticmethod
    def intersect(hashes_a: List[bytes], hashes_b: List[bytes]) -> Tuple[List[int], List[int]]:
        """Compute intersection indices between two hashed sets (exact mode).

        1:1 matching: each hash in B is matched at most once (first occurrence
        in A wins). Duplicate hashes within a party are logged as warnings.

        Returns:
            (indices_a, indices_b): Matching indices in each party's original list.
        """
        # Build index for B; warn on duplicates
        set_b = {}
        for idx, h in enumerate(hashes_b):
            if h in set_b:
                logger.warning("Duplicate hash in party B at indices %d and %d", set_b[h], idx)
            else:
                set_b[h] = idx

        indices_a, indices_b = [], []
        matched_b = set()
        for idx_a, h in enumerate(hashes_a):
            if h in set_b and set_b[h] not in matched_b:
                indices_a.append(idx_a)
                indices_b.append(set_b[h])
                matched_b.add(set_b[h])

        match_rate = len(indices_a) / max(len(hashes_a), 1)
        logger.info(
            "PSA exact match: %d matches out of (%d, %d) records (%.1f%%)",
            len(indices_a), len(hashes_a), len(hashes_b), match_rate * 100
        )
        return indices_a, indices_b

    @staticmethod
    def multi_party_intersect(all_hashes: List[List[bytes]]) -> List[List[int]]:
        """Compute intersection across >2 parties (exact mode).

        Uses pairwise reduction: intersect party 0 with party 1,
        then intersect result with party 2, etc.

        Returns:
            List of index lists, one per party, for the common intersection.
        """
        if len(all_hashes) < 2:
            raise ValueError("Need at least 2 parties for alignment")

        current_a, current_b = PSAProtocol.intersect(all_hashes[0], all_hashes[1])
        all_indices = [current_a, current_b]
        common_hashes = [all_hashes[0][i] for i in current_a]

        for party_idx in range(2, len(all_hashes)):
            new_common, new_party = PSAProtocol.intersect(
                common_hashes, all_hashes[party_idx]
            )
            all_indices = [[idx_list[i] for i in new_common] for idx_list in all_indices]
            all_indices.append(new_party)
            common_hashes = [common_hashes[i] for i in new_common]

        logger.info(
            "Multi-party PSA: %d parties, %d common records",
            len(all_hashes), len(all_indices[0])
        )
        return all_indices
