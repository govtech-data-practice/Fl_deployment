#!/usr/bin/env python3
"""
Paillier Homomorphic Encryption — Secure Inference Demo
========================================================
Demonstrates linear model inference on encrypted data using the Paillier
cryptosystem.  The entire Paillier scheme (keygen, encrypt, decrypt,
homomorphic addition, scalar multiplication) is implemented from scratch
using only Python builtins.

Key idea:
    Paillier is an *additively* homomorphic encryption scheme:
        Enc(a) * Enc(b) mod n^2  =  Enc(a + b)
        Enc(a)^k       mod n^2  =  Enc(k * a)

    This lets us compute a linear function  y = w . x  on encrypted x:
        Enc(y) = product_i( Enc(x_i)^{w_i} )  mod n^2

WARNING: 512-bit keys are used here for speed.  Production systems need
>= 2048 bits.  This code is for education only — do NOT use in production.
"""

import random
import math
import time
import sys


# ======================================================================
# Minimal Paillier implementation
# ======================================================================

def _is_prime(n, k=20):
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    # write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _gen_prime(bits):
    """Generate a random prime of the given bit length."""
    while True:
        p = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if _is_prime(p):
            return p


def _L(u, n):
    """Paillier L-function: L(u) = (u - 1) / n  (integer division)."""
    return (u - 1) // n


def _modinv(a, m):
    """Modular multiplicative inverse using Python's built-in pow."""
    return pow(a, -1, m)


class PaillierPublicKey:
    """Paillier public key (n, g)."""

    def __init__(self, n, g):
        self.n = n
        self.g = g
        self.n_sq = n * n

    def encrypt(self, plaintext):
        """Encrypt an integer plaintext.  Handles negative values via modular
        representation: negative m is stored as n + m."""
        n, g, n_sq = self.n, self.g, self.n_sq
        # map negative plaintext into Z_n
        m = plaintext % n
        # pick random r in Z*_n
        while True:
            r = random.randrange(1, n)
            if math.gcd(r, n) == 1:
                break
        # c = g^m * r^n mod n^2
        c = (pow(g, m, n_sq) * pow(r, n, n_sq)) % n_sq
        return c


class PaillierPrivateKey:
    """Paillier private key (lambda, mu) with associated public key."""

    def __init__(self, pub, lam, mu):
        self.pub = pub
        self.lam = lam
        self.mu = mu

    def decrypt(self, ciphertext):
        """Decrypt a ciphertext back to an integer.  Values > n/2 are
        interpreted as negative (two's-complement style)."""
        n, n_sq = self.pub.n, self.pub.n_sq
        x = _L(pow(ciphertext, self.lam, n_sq), n)
        m = (x * self.mu) % n
        # map back to signed integer
        if m > n // 2:
            m -= n
        return m


def paillier_keygen(bits=512):
    """Generate a Paillier key pair.

    Parameters
    ----------
    bits : int
        Bit length of each prime factor (key modulus n will be ~2*bits).

    Returns
    -------
    pub : PaillierPublicKey
    priv : PaillierPrivateKey
    """
    half = bits // 2
    p = _gen_prime(half)
    q = _gen_prime(half)
    while p == q:
        q = _gen_prime(half)

    n = p * q
    n_sq = n * n
    g = n + 1  # standard simplification: g = n + 1

    # lambda = lcm(p-1, q-1)
    lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)

    # mu = L(g^lambda mod n^2)^{-1} mod n
    x = _L(pow(g, lam, n_sq), n)
    mu = _modinv(x, n)

    pub = PaillierPublicKey(n, g)
    priv = PaillierPrivateKey(pub, lam, mu)
    return pub, priv


# ======================================================================
# Homomorphic operations (work on ciphertexts)
# ======================================================================

def enc_add(pub, c1, c2):
    """Homomorphic addition: Enc(a) * Enc(b) = Enc(a + b)."""
    return (c1 * c2) % pub.n_sq


def enc_scalar_mul(pub, c, scalar):
    """Homomorphic scalar multiplication: Enc(a)^k = Enc(k * a).
    Handles negative scalars by inverting the ciphertext first."""
    if scalar < 0:
        # Enc(a)^{-1} = Enc(-a)
        c = _modinv(c, pub.n_sq)
        scalar = -scalar
    return pow(c, scalar, pub.n_sq)


# ======================================================================
# Simple logistic regression on synthetic data
# ======================================================================

def _sigmoid(z):
    """Numerically stable sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _generate_data(n_samples=200, n_features=4, seed=42):
    """Generate linearly separable synthetic data."""
    rng = random.Random(seed)
    true_w = [rng.gauss(0, 1) for _ in range(n_features)]
    true_b = rng.gauss(0, 0.5)

    X, y = [], []
    for _ in range(n_samples):
        x = [rng.gauss(0, 1) for _ in range(n_features)]
        logit = sum(wi * xi for wi, xi in zip(true_w, x)) + true_b
        label = 1 if rng.random() < _sigmoid(logit) else 0
        X.append(x)
        y.append(label)
    return X, y, n_features


def _train_logistic(X, y, n_features, lr=0.1, epochs=100):
    """Train logistic regression with SGD (pure Python)."""
    rng = random.Random(123)
    w = [rng.gauss(0, 0.01) for _ in range(n_features)]
    b = 0.0

    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = sum(wj * xj for wj, xj in zip(w, xi)) + b
            p = _sigmoid(z)
            err = p - yi
            for j in range(n_features):
                w[j] -= lr * err * xi[j]
            b -= lr * err
    return w, b


# ======================================================================
# Inference: plaintext vs encrypted
# ======================================================================

def _plaintext_inference(w, b, x):
    """Standard linear inference: z = w . x + b."""
    return sum(wj * xj for wj, xj in zip(w, x)) + b


def _encrypted_inference(pub, priv, w_int, b_int, x, scale):
    """Inference on Paillier-encrypted features.

    We use fixed-point: each float is multiplied by `scale` and rounded
    to an integer before encryption.  Weights are kept in the clear
    (the server knows the model).

    Steps:
        1. Client encrypts each feature: Enc(x_i * scale)
        2. Server computes:  Enc(sum_i w_i * x_i * scale)
           using homomorphic scalar-mul and addition.
           Then adds Enc(b * scale) (encrypted bias).
        3. Client decrypts, divides by scale to recover the logit.
    """
    # Step 1 — client side: encrypt scaled features
    enc_x = [pub.encrypt(round(xj * scale)) for xj in x]

    # Step 2 — server side: homomorphic linear combination
    # Enc(w_0 * x_0 * scale)
    result = enc_scalar_mul(pub, enc_x[0], w_int[0])
    for j in range(1, len(w_int)):
        term = enc_scalar_mul(pub, enc_x[j], w_int[j])
        result = enc_add(pub, result, term)
    # add encrypted bias
    enc_bias = pub.encrypt(b_int)
    result = enc_add(pub, result, enc_bias)

    # Step 3 — client side: decrypt and rescale
    # The result is at scale^2 because both weights and features were scaled.
    raw = priv.decrypt(result)
    return raw / (scale * scale)


# ======================================================================
# Main demo
# ======================================================================

def main():
    SCALE = 1000  # fixed-point scaling factor
    N_FEATURES = 4
    N_TEST = 5  # number of test samples to run encrypted inference on

    print("=" * 65)
    print("  Paillier Homomorphic Encryption — Secure Inference Demo")
    print("=" * 65)

    # --- Key generation ---
    print("\n[1] Generating Paillier key pair (512-bit primes) ...")
    t0 = time.time()
    pub, priv = paillier_keygen(bits=512)
    t_keygen = time.time() - t0
    print(f"    Key generated in {t_keygen:.2f}s")
    print(f"    n has {pub.n.bit_length()} bits")

    # --- Train model ---
    print("\n[2] Training logistic regression on synthetic data ...")
    X, y, d = _generate_data(n_samples=300, n_features=N_FEATURES)
    w, b = _train_logistic(X, y, d)
    print(f"    Weights: {[round(wj, 4) for wj in w]}")
    print(f"    Bias:    {round(b, 4)}")

    # Integer weights for homomorphic ops (scale and round).
    # Weights are at scale S, features will be encrypted at scale S,
    # so products are at scale S^2.  Bias must also be at S^2.
    w_int = [round(wj * SCALE) for wj in w]
    b_int = round(b * SCALE * SCALE)

    # --- Inference comparison ---
    print(f"\n[3] Running inference on {N_TEST} samples  (plaintext vs encrypted)")
    print("-" * 65)
    print(f"{'Sample':>7} | {'Plaintext':>12} | {'Encrypted':>12} | {'Diff':>10} | {'Enc Time':>10}")
    print("-" * 65)

    total_plain, total_enc = 0.0, 0.0
    max_diff = 0.0

    for i in range(N_TEST):
        x = X[i]

        # Plaintext
        t0 = time.time()
        z_plain = _plaintext_inference(w, b, x)
        t_plain = time.time() - t0
        total_plain += t_plain

        # Encrypted
        t0 = time.time()
        z_enc = _encrypted_inference(pub, priv, w_int, b_int, x, SCALE)
        t_enc = time.time() - t0
        total_enc += t_enc

        diff = abs(z_plain - z_enc)
        max_diff = max(max_diff, diff)
        print(f"  {i:>5} | {z_plain:>12.6f} | {z_enc:>12.6f} | {diff:>10.6f} | {t_enc:>8.3f}s")

    print("-" * 65)

    # --- Summary ---
    print("\n[4] Summary")
    print(f"    Max |plaintext - encrypted| = {max_diff:.6f}")
    print(f"      (difference is due to fixed-point rounding with scale={SCALE})")
    print(f"    Total plaintext time : {total_plain * 1000:.3f} ms")
    print(f"    Total encrypted time : {total_enc * 1000:.1f} ms")
    slowdown = total_enc / max(total_plain, 1e-9)
    print(f"    Slowdown factor      : ~{slowdown:.0f}x")
    print()
    print("    Takeaway: Paillier enables exact linear computation on ciphertexts,")
    print("    but is orders of magnitude slower than plaintext.  For neural nets,")
    print("    consider CKKS (approximate HE) or hybrid HE + garbled circuits.")
    print()


if __name__ == "__main__":
    main()
