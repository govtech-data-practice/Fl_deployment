#!/usr/bin/env python3
"""
Functional Encryption for Inner Products — Educational Demo
=============================================================
Demonstrates the *concept* of inner-product functional encryption (IPFE).

Idea:
    An authority issues a *function key* sk_w for a weight vector w.
    A client encrypts input x to get ct.
    Anyone with sk_w can compute <w, x> from ct, but learns NOTHING else
    about x.

This is powerful for ML inference: the server holds sk_w (derived from
model weights), the client encrypts features, and the server obtains
the prediction without seeing raw features.

Implementation:
    We use a simplified DDH-based IPFE scheme (Abdalla et al., 2015) in a
    small group for educational purposes.  This is NOT cryptographically
    secure (the group is too small and we use a naive discrete log).

    The real construction works in elliptic curve groups where discrete
    log is hard, and with proper parameter selection.

Reference:
    Abdalla, Bourse, De Caro, Pointcheval. "Simple Functional Encryption
    Schemes for Inner Products." PKC 2015.
"""

import random
import math
import time


# ======================================================================
# Small prime-order group for demonstration
# ======================================================================

# We use Z_p^* with a prime p and generator g of a subgroup of order q.
# For education: small primes so discrete log is feasible.
# In production: use elliptic curves (e.g., Curve25519) where DLog is hard.

# p = 2 * q + 1 where both p and q are prime (safe prime)
Q = 1000000007   # prime, order of our subgroup
P = 2 * Q + 1    # = 2000000015 ... let's find a real safe prime

def _find_safe_prime(start=10000):
    """Find a safe prime p = 2q + 1 where q is prime, for demo purposes."""
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    q = start
    while True:
        if is_prime(q):
            p = 2 * q + 1
            if is_prime(p):
                return p, q
        q += 1


# Use a manageable safe prime
P, Q = _find_safe_prime(50000)

def _find_generator(p, q):
    """Find a generator of the subgroup of order q in Z_p^*."""
    for g_candidate in range(2, p):
        # g = g_candidate^2 mod p gives an element of order q (if it's not 1)
        g = pow(g_candidate, 2, p)
        if g != 1 and pow(g, q, p) == 1:
            return g
    raise RuntimeError("No generator found")


G = _find_generator(P, Q)


def _discrete_log_brute(base, target, p, q):
    """Brute-force discrete log: find x such that base^x = target mod p.
    Only feasible for small groups.  In real IPFE, the result is small
    enough that baby-step-giant-step or Pollard's rho works, or the
    scheme uses a pairing-based approach that avoids DLog entirely.

    Returns a signed integer in [-q/2, q/2]."""
    # For our demo, the inner product values are small, so we search
    # a limited range around 0.
    MAX_SEARCH = 200000

    # Build a lookup table for baby-step values: base^j for j in [0, MAX_SEARCH)
    # Then use giant-step to find match.  But for simplicity, just linear scan
    # in both directions.

    # Search positive: x = 0, 1, 2, ...
    val = 1
    for x in range(MAX_SEARCH):
        if val == target:
            # x might actually represent a negative number if x > q/2
            if x > q // 2:
                return x - q
            return x
        val = (val * base) % p

    # Search from q downward (negative values): base^(q-1) = base^{-1}
    inv_base = pow(base, p - 2, p)  # modular inverse of base mod p
    val = inv_base  # base^{-1} = base^{q-1}
    for x in range(1, MAX_SEARCH):
        if val == target:
            return -x
        val = (val * inv_base) % p

    raise ValueError(f"Discrete log not found within +/-{MAX_SEARCH}")


# ======================================================================
# Simplified IPFE scheme
# ======================================================================

class IPFEMasterKey:
    """Authority's master secret key."""
    def __init__(self, s: list, n: int):
        self.s = s  # list of n secret key components in Z_q
        self.n = n


class IPFEPublicKey:
    """Public parameters."""
    def __init__(self, h: list, n: int):
        self.h = h  # h_i = g^{s_i} mod p
        self.n = n


class IPFEFunctionKey:
    """Function key for a specific weight vector w.
    sk_w = sum(s_i * w_i) mod q."""
    def __init__(self, sk_w: int, w: list):
        self.sk_w = sk_w
        self.w = w


class IPFECiphertext:
    """Ciphertext encrypting a vector x.
    ct = (c0, c1, ..., cn) where c0 = g^r, c_i = h_i^r * g^{x_i}."""
    def __init__(self, c0: int, c: list):
        self.c0 = c0
        self.c = c


def ipfe_setup(n: int) -> tuple:
    """Generate master key and public key for n-dimensional vectors.

    msk.s = (s_1, ..., s_n)  random in Z_q
    pk.h  = (g^{s_1}, ..., g^{s_n})  in Z_p
    """
    s = [random.randrange(1, Q) for _ in range(n)]
    h = [pow(G, si, P) for si in s]
    msk = IPFEMasterKey(s, n)
    pk = IPFEPublicKey(h, n)
    return msk, pk


def ipfe_keygen(msk: IPFEMasterKey, w: list) -> IPFEFunctionKey:
    """Derive a function key for weight vector w.

    sk_w = sum(s_i * w_i) mod q

    Anyone with sk_w can compute <w, x> from Enc(x), but nothing more.
    """
    sk_w = sum(si * wi for si, wi in zip(msk.s, w)) % Q
    return IPFEFunctionKey(sk_w, w)


def ipfe_encrypt(pk: IPFEPublicKey, x: list) -> IPFECiphertext:
    """Encrypt input vector x.

    Pick random r in Z_q.
    c0 = g^r
    c_i = h_i^r * g^{x_i}   for i = 1..n
    """
    r = random.randrange(1, Q)
    c0 = pow(G, r, P)
    c = []
    for i in range(pk.n):
        xi_mod = x[i] % Q
        ci = (pow(pk.h[i], r, P) * pow(G, xi_mod, P)) % P
        c.append(ci)
    return IPFECiphertext(c0, c)


def ipfe_decrypt(fk: IPFEFunctionKey, ct: IPFECiphertext) -> int:
    """Decrypt to obtain <w, x> (the inner product).

    Compute:
        numerator   = product(c_i^{w_i})  for i = 1..n
        denominator = c0^{sk_w}
        g^{<w,x>}   = numerator / denominator  mod p

    Then recover <w, x> via discrete log (feasible when result is small).
    """
    # numerator = product of c_i^{w_i} mod p
    numerator = 1
    for ci, wi in zip(ct.c, fk.w):
        wi_mod = wi % Q
        numerator = (numerator * pow(ci, wi_mod, P)) % P

    # denominator = c0^{sk_w} mod p
    denominator = pow(ct.c0, fk.sk_w, P)

    # g^{<w,x>} = numerator * denominator^{-1} mod p
    g_ip = (numerator * pow(denominator, -1, P)) % P

    # Recover <w, x> via discrete log
    ip = _discrete_log_brute(G, g_ip, P, Q)
    return ip


# ======================================================================
# Main demo
# ======================================================================

def main():
    N_DIM = 4  # keep small for discrete log feasibility

    print("=" * 65)
    print("  Inner-Product Functional Encryption — Educational Demo")
    print("=" * 65)

    print(f"\n  Group parameters (EDUCATIONAL ONLY — not secure):")
    print(f"    Safe prime p = {P}")
    print(f"    Subgroup order q = {Q}")
    print(f"    Generator g = {G}")
    print(f"    Vector dimension = {N_DIM}")

    # --- Setup ---
    print(f"\n[1] Authority: generating master key and public key ...")
    msk, pk = ipfe_setup(N_DIM)
    print(f"    Master secret s = {msk.s}")
    print(f"    Public key    h = {pk.h}")

    # --- Model as weight vector ---
    # Use small integer weights so DLog is feasible
    rng = random.Random(42)
    w = [rng.randint(-5, 5) for _ in range(N_DIM)]
    print(f"\n[2] Model weights (issued as function key): w = {w}")

    fk = ipfe_keygen(msk, w)
    print(f"    Function key sk_w = {fk.sk_w}")
    print(f"    (Server receives sk_w; it can compute <w, x> for any Enc(x))")

    # --- Encrypt and decrypt ---
    N_TESTS = 5
    print(f"\n[3] Running {N_TESTS} encrypted inference requests\n")
    print(f"{'#':>3} | {'Input x':>25} | {'<w,x> plain':>12} | {'<w,x> FE':>10} | {'Match':>6}")
    print("-" * 70)

    for t in range(N_TESTS):
        x = [rng.randint(-10, 10) for _ in range(N_DIM)]

        # Plaintext inner product
        ip_plain = sum(wi * xi for wi, xi in zip(w, x))

        # Functional encryption
        ct = ipfe_encrypt(pk, x)
        ip_fe = ipfe_decrypt(fk, ct)

        match = "OK" if ip_plain == ip_fe else "FAIL"
        print(f" {t:>2} | {str(x):>25} | {ip_plain:>12} | {ip_fe:>10} | {match:>6}")

    print("-" * 70)

    # --- What the server learns ---
    print(f"\n[4] Security analysis")
    x_secret = [3, -7, 2, 5]
    ip = sum(wi * xi for wi, xi in zip(w, x_secret))
    print(f"    Client's secret input: x = {x_secret}")
    print(f"    Weight vector:         w = {w}")
    print(f"    Inner product:         <w, x> = {ip}")
    print()
    print(f"    The server learns ONLY that <w, x> = {ip}")
    print(f"    It does NOT learn x = {x_secret}")
    print(f"    Many different x values could produce <w, x> = {ip}")
    print()

    # Show alternative inputs with same inner product
    print(f"    Example: other inputs with <w, x> = {ip}:")
    count = 0
    for _ in range(100000):
        x_alt = [rng.randint(-10, 10) for _ in range(N_DIM)]
        if sum(wi * xi for wi, xi in zip(w, x_alt)) == ip and x_alt != x_secret:
            print(f"      x' = {x_alt}  also gives <w, x'> = {ip}")
            count += 1
            if count >= 3:
                break

    # --- Summary ---
    print(f"\n[5] Summary")
    print(f"    Functional encryption lets a server compute a SPECIFIC function")
    print(f"    of encrypted data without learning the data itself.")
    print()
    print(f"    For ML inference:")
    print(f"    - Authority (model owner) issues function keys for model weights")
    print(f"    - Client encrypts features")
    print(f"    - Server computes prediction = <weights, features> from ciphertext")
    print(f"    - Server learns only the prediction, not the features")
    print()
    print(f"    Limitations of this demo:")
    print(f"    - Tiny group (DLog is easy) — real schemes use elliptic curves")
    print(f"    - Only supports linear (inner product) functions")
    print(f"    - No support for non-linear activations (need multi-input FE or")
    print(f"      quadratic FE for that)")
    print()
    print(f"    Production libraries:")
    print(f"    - CiFEr (C library for functional encryption)")
    print(f"    - GoFE  (Go library)")
    print(f"    - FENTEC project (EU research, multiple FE schemes)")
    print(f"    - Decentralized MCFE (multi-client functional encryption)")
    print()


if __name__ == "__main__":
    main()
