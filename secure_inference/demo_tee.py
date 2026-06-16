#!/usr/bin/env python3
"""
TEE (Trusted Execution Environment) — Secure Inference Simulation
==================================================================
Simulates the data-flow of TEE-based inference where:

    1. Client encrypts input with a shared key (established via attestation).
    2. Encrypted input is sent to the server (untrusted host).
    3. The server forwards it into the *enclave* (TEE).
    4. Inside the enclave: decrypt input -> run model -> encrypt result.
    5. Encrypted result is returned to the client.
    6. Client decrypts the result.

The untrusted host never sees plaintext data or model outputs.

NOTE: This is a *simulation*.  Real TEE implementations use:
  - AWS Nitro Enclaves (Nitro Security Module, attestation documents)
  - Intel SGX / TDX (remote attestation, sealed storage)
  - AMD SEV-SNP (encrypted VM memory, attestation)
  - ARM TrustZone / CCA

The encryption here uses AES-CTR via Python's hashlib-derived keystream
(for zero-dependency portability).  It is NOT a production AES
implementation.
"""

import hashlib
import os
import struct
import time
import math
import random


# ======================================================================
# Minimal AES-CTR-like stream cipher (for demo — NOT production crypto)
# ======================================================================

def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """Generate a keystream of `length` bytes using HMAC-SHA256 in CTR mode."""
    blocks = []
    ctr = 0
    while len(b"".join(blocks)) < length:
        data = key + nonce + struct.pack("<Q", ctr)
        blocks.append(hashlib.sha256(data).digest())
        ctr += 1
    return b"".join(blocks)[:length]


def encrypt_bytes(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt plaintext bytes. Returns nonce || ciphertext."""
    nonce = os.urandom(16)
    ks = _keystream(key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, ks))
    return nonce + ciphertext


def decrypt_bytes(key: bytes, blob: bytes) -> bytes:
    """Decrypt a nonce || ciphertext blob."""
    nonce = blob[:16]
    ciphertext = blob[16:]
    ks = _keystream(key, nonce, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, ks))


# ======================================================================
# Float <-> bytes helpers
# ======================================================================

def floats_to_bytes(values: list) -> bytes:
    """Pack a list of floats as little-endian doubles."""
    return struct.pack(f"<{len(values)}d", *values)


def bytes_to_floats(data: bytes) -> list:
    """Unpack little-endian doubles."""
    n = len(data) // 8
    return list(struct.unpack(f"<{n}d", data))


# ======================================================================
# Simple model (logistic regression on synthetic data)
# ======================================================================

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _make_model(n_features=5, seed=7):
    """Return random model weights and bias."""
    rng = random.Random(seed)
    w = [rng.gauss(0, 1) for _ in range(n_features)]
    b = rng.gauss(0, 0.5)
    return w, b


def _run_model(w, b, x):
    """Run logistic regression inference."""
    logit = sum(wi * xi for wi, xi in zip(w, x)) + b
    return _sigmoid(logit)


# ======================================================================
# Parties
# ======================================================================

class Client:
    """Represents the data owner / end user."""

    def __init__(self, session_key: bytes):
        self.key = session_key

    def prepare_request(self, features: list) -> bytes:
        """Encrypt input features for the enclave."""
        plaintext = floats_to_bytes(features)
        return encrypt_bytes(self.key, plaintext)

    def read_response(self, encrypted_result: bytes) -> list:
        """Decrypt the enclave's response."""
        plaintext = decrypt_bytes(self.key, encrypted_result)
        return bytes_to_floats(plaintext)


class UntrustedHost:
    """The cloud server — routes traffic but cannot decrypt."""

    def __init__(self, enclave):
        self.enclave = enclave

    def handle_request(self, encrypted_input: bytes) -> bytes:
        """Forward encrypted input to enclave, return encrypted output.
        The host sees only ciphertext."""
        print("    [Host]    Received encrypted input  "
              f"({len(encrypted_input)} bytes, looks random)")
        print(f"              First 32 bytes: {encrypted_input[:32].hex()}")

        encrypted_output = self.enclave.process(encrypted_input)

        print("    [Host]    Received encrypted output "
              f"({len(encrypted_output)} bytes, looks random)")
        print(f"              First 32 bytes: {encrypted_output[:32].hex()}")
        return encrypted_output


class Enclave:
    """Simulates a TEE enclave.  In practice this runs inside SGX/Nitro/etc.
    The enclave holds the session key (provisioned via attestation) and the
    model weights."""

    def __init__(self, session_key: bytes, model_weights, model_bias):
        self.key = session_key
        self.w = model_weights
        self.b = model_bias

    def process(self, encrypted_input: bytes) -> bytes:
        """Decrypt input -> run inference -> encrypt output.
        All happens inside the secure enclave memory."""
        # Decrypt (only possible inside enclave which has the key)
        plain_bytes = decrypt_bytes(self.key, encrypted_input)
        features = bytes_to_floats(plain_bytes)
        print("    [Enclave] Decrypted features (INSIDE enclave only):")
        print(f"              {[round(f, 4) for f in features]}")

        # Run model
        prediction = _run_model(self.w, self.b, features)
        print(f"    [Enclave] Model prediction: {prediction:.6f}")

        # Encrypt output
        result_bytes = floats_to_bytes([prediction])
        return encrypt_bytes(self.key, result_bytes)


# ======================================================================
# Attestation simulation
# ======================================================================

def simulate_attestation():
    """In real TEE:
      1. Enclave generates a key pair.
      2. Enclave produces an *attestation report* signed by hardware
         (e.g., Intel's quoting enclave or Nitro's NSM).
      3. Client verifies the report against the vendor's root of trust.
      4. Client sends a session key encrypted to the enclave's public key.

    Here we just create a shared key directly."""
    session_key = os.urandom(32)
    print("    Attestation: (simulated) enclave identity verified")
    print(f"    Session key established: {session_key[:8].hex()}...")
    return session_key


# ======================================================================
# Main demo
# ======================================================================

def main():
    N_FEATURES = 5
    N_SAMPLES = 3

    print("=" * 65)
    print("  TEE-Based Secure Inference Simulation")
    print("=" * 65)

    # --- Setup ---
    print("\n[1] Attestation & key establishment")
    session_key = simulate_attestation()

    print("\n[2] Loading model into enclave")
    w, b = _make_model(N_FEATURES)
    print(f"    Weights: {[round(wi, 4) for wi in w]}")
    print(f"    Bias:    {round(b, 4)}")

    enclave = Enclave(session_key, w, b)
    host = UntrustedHost(enclave)
    client = Client(session_key)

    # --- Run inference ---
    rng = random.Random(42)
    print(f"\n[3] Running {N_SAMPLES} secure inference requests\n")

    for i in range(N_SAMPLES):
        print(f"  --- Request {i + 1} ---")
        features = [rng.gauss(0, 1) for _ in range(N_FEATURES)]
        print(f"    [Client]  Input features: {[round(f, 4) for f in features]}")

        # Client encrypts
        encrypted_input = client.prepare_request(features)
        print(f"    [Client]  Encrypted and sent to server")

        # Host routes to enclave
        encrypted_output = host.handle_request(encrypted_input)

        # Client decrypts
        result = client.read_response(encrypted_output)
        print(f"    [Client]  Decrypted prediction: {result[0]:.6f}")

        # Verify against plaintext
        expected = _run_model(w, b, features)
        print(f"    [Verify]  Plaintext prediction:  {expected:.6f}")
        assert abs(result[0] - expected) < 1e-10, "Mismatch!"
        print(f"    [Verify]  Match confirmed\n")

    # --- Summary ---
    print("=" * 65)
    print("  What each party saw:")
    print("=" * 65)
    print("  Client  : plaintext input, plaintext output (owns the data)")
    print("  Host    : encrypted blobs only (cannot decrypt)")
    print("  Enclave : plaintext inside secure memory (hardware-protected)")
    print()
    print("  In production, the enclave's integrity is guaranteed by hardware")
    print("  attestation (SGX DCAP, Nitro NSM, SEV-SNP VCEK).  The host OS")
    print("  and hypervisor cannot read enclave memory.")
    print()


if __name__ == "__main__":
    main()
