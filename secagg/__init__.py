"""Secure Aggregation configuration and utilities.

Re-exports the core SecAgg implementation from fl_common.secagg.
This directory exists as the guide-referenced entry point for SecAgg
configuration and tooling.

Usage:
    from secagg import secagg_mask_parameters
"""

from fl_common.secagg import secagg_mask_parameters, _pair_seed

__all__ = ["secagg_mask_parameters", "_pair_seed"]
