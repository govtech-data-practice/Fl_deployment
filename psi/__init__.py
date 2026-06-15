"""Private Set Intersection (PSI) for secure entity alignment.

Used as a pre-processing step for Vertical FL (VFL) to align entities
across parties without revealing non-matching records.

Usage:
    from psi import PSIProtocol, EntityAligner
"""

from psi.psi import PSIProtocol
from psi.entity_alignment import EntityAligner

__all__ = ["PSIProtocol", "EntityAligner"]
