"""Private Set Alignment (PSA) for secure entity alignment.

Used as a pre-processing step for Vertical FL (VFL) to align entities
across parties without revealing non-matching records.

PSA extends traditional PSI (exact matching) with fuzzy matching via
Bloom filter CLK encoding (anonlink + clkhash), enabling alignment
when parties do not share common identifiers.

Usage:
    from psa import PSAProtocol, EntityAligner
"""

from psa.psa import PSAProtocol
from psa.entity_alignment import EntityAligner

__all__ = ["PSAProtocol", "EntityAligner"]
