#!/usr/bin/env python3
"""
Additive Secret Sharing — Secure Inference Demo
=================================================
Demonstrates 2-party secure linear inference using additive secret sharing.

Protocol:
    Input x is split into shares:   x = x1 + x2   (mod PRIME)
    Model w is known to both servers (common in MPC inference).

    Server 1 holds x1, computes partial: y1 = w . x1
    Server 2 holds x2, computes partial: y2 = w . x2

    Reconstruct:  y = y1 + y2 = w . (x1 + x2) = w . x

Neither server alone learns x, because each share is uniformly random
(information-theoretically secure with an honest-but-curious adversary).

For the bias term, Server 1 adds it; Server 2 contributes 0.

All arithmetic is in a prime field Z_p to avoid overflow issues with
arbitrary-precision integers, then converted back to real numbers.
"""

import random
import math
import time

# A large prime for modular arithmetic (fits in 64-bit for speed).
# Using a Mersenne-prime-like value.
PRIME = (1 << 61) - 1  # 2^61 - 1 = 2305843009213693951 (Mersenne prime)
HALF_PRIME = PRIME // 2

# Fixed-point scaling: map floats to integers in Z_p.
SCALE = 10**6


# ======================================================================
# Secret sharing primitives
# ======================================================================

def share(value_int: int, prime: int = PRIME) -> tuple:
    """Split an integer into two additive shares mod prime.
    Returns (s1, s2) such that (s1 + s2) % prime == value_int % prime."""
    v = value_int % prime
    s1 = random.randrange(0, prime)
    s2 = (v - s1) % prime
    return s1, s2


def reconstruct(s1: int, s2: int, prime: int = PRIME) -> int:
    """Reconstruct value from two additive shares."""
    return (s1 + s2) % prime


def to_signed(value: int, prime: int = PRIME) -> int:
    """Convert from Z_p to signed integer (values > p/2 are negative)."""
    if value > prime // 2:
        return value - prime
    return value


def float_to_field(f: float, scale: int = SCALE, prime: int = PRIME) -> int:
    """Map a float to a field element using fixed-point encoding."""
    return round(f * scale) % prime


def field_to_float(v: int, scale: int = SCALE, prime: int = PRIME) -> float:
    """Map a field element back to a float."""
    return to_signed(v, prime) / scale


# ======================================================================
# Simple model
# ======================================================================

def _make_model(n_features=6, seed=99):
    """Generate random model weights and bias."""
    rng = random.Random(seed)
    w = [rng.gauss(0, 1) for _ in range(n_features)]
    b = rng.gauss(0, 0.3)
    return w, b


def _plaintext_inference(w, b, x):
    """Standard linear inference (plaintext)."""
    return sum(wi * xi for wi, xi in zip(w, x)) + b


# ======================================================================
# Two-party MPC inference
# ======================================================================

class Server:
    """One of two MPC servers.  Holds model weights (in the clear) and
    receives a share of the input."""

    def __init__(self, name, w_field, b_field, include_bias=False):
        self.name = name
        self.w = w_field  # model weights as field elements
        self.b = b_field if include_bias else 0

    def compute_partial(self, x_share: list) -> int:
        """Compute w . x_share (mod PRIME), optionally adding bias."""
        result = 0
        for wi, xi in zip(self.w, x_share):
            # wi * xi in the field.  Since both are scaled, the product
            # is scaled^2, so we divide by SCALE.
            product = (wi * xi) % PRIME
            result = (result + product) % PRIME
        # Add bias (already at scale^2 since we multiply by 1*SCALE below)
        result = (result + self.b) % PRIME
        return result


def run_mpc_inference(w, b, x):
    """Full 2-party secret-sharing inference pipeline.

    We use a simplified approach:
      - Weights are in the clear on both servers (common for model-public scenarios).
      - Input x is secret-shared.
      - Each server computes w . (its share of x).
      - Shares are reconstructed to get the result.

    Fixed-point handling:
      - x is at scale S, w is at scale S.
      - w * x_share is at scale S^2.
      - Final result after reconstruction is at scale S^2.
      - We divide by S once at the end to get back to scale S, then to float.
    """
    n = len(w)

    # Encode weights and bias as field elements (at scale S)
    w_field = [float_to_field(wi) for wi in w]
    # Bias needs to be at scale S^2 to match the product terms
    b_field = float_to_field(b * SCALE)  # b * S * S / S = b * S

    # Secret-share input features
    x_shares_1 = []
    x_shares_2 = []
    for xi in x:
        xi_field = float_to_field(xi)
        s1, s2 = share(xi_field)
        x_shares_1.append(s1)
        x_shares_2.append(s2)

    # Create servers — bias goes to Server 1 only
    server1 = Server("Server-1", w_field, b_field, include_bias=True)
    server2 = Server("Server-2", w_field, 0, include_bias=False)

    # Each server computes its partial result
    y1 = server1.compute_partial(x_shares_1)
    y2 = server2.compute_partial(x_shares_2)

    # Reconstruct
    y_field = reconstruct(y1, y2)

    # Convert back: result is at scale S^2, so divide by S^2
    result = to_signed(y_field, PRIME) / (SCALE * SCALE)

    return result, y1, y2


# ======================================================================
# Main demo
# ======================================================================

def main():
    N_FEATURES = 6
    N_SAMPLES = 8

    print("=" * 65)
    print("  Additive Secret Sharing — Secure Inference Demo")
    print("=" * 65)

    # --- Setup ---
    print("\n[1] Model setup")
    w, b = _make_model(N_FEATURES)
    print(f"    Features : {N_FEATURES}")
    print(f"    Weights  : {[round(wi, 4) for wi in w]}")
    print(f"    Bias     : {round(b, 4)}")
    print(f"    Field    : Z_{PRIME} (Mersenne prime 2^61 - 1)")
    print(f"    Scale    : {SCALE}")

    # --- Inference ---
    print(f"\n[2] Running {N_SAMPLES} inferences (plaintext vs secret-shared)\n")
    print(f"{'#':>3} | {'Plaintext':>12} | {'MPC Result':>12} | {'Diff':>10} | {'Share1':>20} | {'Share2':>20}")
    print("-" * 95)

    rng = random.Random(42)
    max_diff = 0.0

    for i in range(N_SAMPLES):
        x = [rng.gauss(0, 1) for _ in range(N_FEATURES)]

        # Plaintext
        y_plain = _plaintext_inference(w, b, x)

        # MPC
        y_mpc, y1, y2 = run_mpc_inference(w, b, x)

        diff = abs(y_plain - y_mpc)
        max_diff = max(max_diff, diff)

        print(f" {i:>2} | {y_plain:>12.6f} | {y_mpc:>12.6f} | {diff:>10.7f} | {y1:>20d} | {y2:>20d}")

    print("-" * 95)

    # --- Security demonstration ---
    print(f"\n[3] Security property demonstration")
    x_demo = [1.5, -0.7, 2.3, 0.1, -1.8, 0.9]
    print(f"    Original input: {x_demo}")

    x_field = [float_to_field(xi) for xi in x_demo]
    shares = [share(xi_f) for xi_f in x_field]

    print(f"\n    Server 1 sees shares: {[s[0] for s in shares]}")
    print(f"    Server 2 sees shares: {[s[1] for s in shares]}")
    print(f"\n    Each share is a random number in [0, {PRIME})")
    print(f"    Individually, a share reveals NOTHING about the original value.")
    print(f"    Only when combined: share1 + share2 mod p = original")

    # Verify reconstruction
    for j, (xi, (s1, s2)) in enumerate(zip(x_demo, shares)):
        recovered = field_to_float(reconstruct(s1, s2))
        print(f"    Feature {j}: {s1} + {s2} mod p -> {recovered:.4f}  (original: {xi})")

    # --- Summary ---
    print(f"\n[4] Summary")
    print(f"    Max |plaintext - MPC| = {max_diff:.7f}")
    print(f"      (difference is due to fixed-point rounding, scale = {SCALE})")
    print()
    print("    Protocol properties:")
    print("    - Information-theoretic security (no crypto assumptions)")
    print("    - Honest-but-curious adversary model (passive corruption)")
    print("    - Linear operations are free (just local computation)")
    print("    - Non-linear operations (ReLU, sigmoid) need interaction")
    print("      -> garbled circuits, beaver triples, or function secret sharing")
    print()
    print("    Real frameworks: MP-SPDZ, ABY3, CrypTen, SecretFlow")
    print()


if __name__ == "__main__":
    main()
