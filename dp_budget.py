#!/usr/bin/env python3
"""Differential Privacy budget calculator.

Computes (epsilon, delta)-DP guarantees for the Gaussian mechanism
using Renyi DP accounting.

Usage:
    python dp_budget.py --preset DP_STRONG --rounds 100
    python dp_budget.py --sigma 1.5 --rounds 100 --delta 1e-5
    python dp_budget.py --all --rounds 100
"""

import argparse
import sys

from fl_common.dp import PrivacyAccountant, DP_PRESETS, get_dp_config


def compute_budget(sigma: float, rounds: int, delta: float = 1e-5,
                   sample_rate: float = 1.0) -> float:
    accountant = PrivacyAccountant(
        noise_multiplier=sigma, sample_rate=sample_rate, delta=delta
    )
    accountant.step(rounds)
    return accountant.get_epsilon()


def main():
    parser = argparse.ArgumentParser(
        description="Compute DP privacy budget (epsilon) for FL training."
    )
    parser.add_argument("--preset", choices=list(DP_PRESETS.keys()),
                        help="Named DP preset")
    parser.add_argument("--sigma", type=float, help="Noise multiplier (overrides preset)")
    parser.add_argument("--rounds", type=int, default=100, help="Number of training rounds")
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta parameter")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Poisson subsampling rate (1.0 = full batch)")
    parser.add_argument("--all", action="store_true",
                        help="Show budget for all named presets")
    args = parser.parse_args()

    if args.all:
        print(f"DP Budget Summary — {args.rounds} rounds, delta={args.delta}")
        print(f"{'Preset':<15} {'sigma':>8} {'C':>6} {'epsilon':>10}")
        print("-" * 42)
        for name, cfg in DP_PRESETS.items():
            eps = compute_budget(cfg["sigma"], args.rounds, args.delta, args.sample_rate)
            print(f"{name:<15} {cfg['sigma']:>8.1f} {cfg['C']:>6.1f} {eps:>10.2f}")
        return

    if args.sigma is not None:
        sigma = args.sigma
        preset_name = "custom"
    elif args.preset:
        cfg = get_dp_config(args.preset)
        sigma = cfg["sigma"]
        preset_name = cfg["preset"]
    else:
        cfg = get_dp_config(None)  # fail-closed to DP_STRONG
        sigma = cfg["sigma"]
        preset_name = cfg["preset"]

    eps = compute_budget(sigma, args.rounds, args.delta, args.sample_rate)

    print(f"Preset:       {preset_name}")
    print(f"Sigma:        {sigma}")
    print(f"Rounds:       {args.rounds}")
    print(f"Delta:        {args.delta}")
    print(f"Sample rate:  {args.sample_rate}")
    print(f"Epsilon:      {eps:.4f}")

    if eps > 25:
        print("\nWARNING: epsilon > 25 — weak privacy guarantee.")
    elif eps > 10:
        print(f"\nNote: moderate privacy (epsilon={eps:.1f}).")
    else:
        print(f"\nStrong privacy guarantee (epsilon={eps:.1f}).")


if __name__ == "__main__":
    main()
