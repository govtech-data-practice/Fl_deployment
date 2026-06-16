"""Homomorphic Encryption for Secure FL Inference.

Library: TenSEAL 0.3+ (OpenMined)
    https://github.com/OpenMined/TenSEAL
    pip install tenseal
    Built on Microsoft SEAL (C++): https://github.com/microsoft/SEAL

Schemes:
    - CKKS: approximate arithmetic on real/complex numbers (ML inference)
    - BFV: exact integer arithmetic (counting, aggregation)

Practical constraints:
    - CKKS supports polynomial ops only (add, multiply, rotate)
    - Suitable for: small MLPs, linear models, logistic regression
    - Not practical for: large CNNs (DenseNet, ResNet), transformers

Usage:
    from fl_pets.he import create_context, encrypt, decrypt

    ctx = create_context(scheme="ckks")
    encrypted = encrypt(ctx, [1.0, 2.0, 3.0])
    result = encrypted + encrypted  # homomorphic addition
    plain = decrypt(result)         # [2.0, 4.0, 6.0]
"""

import tenseal as ts


def create_context(scheme="ckks", poly_mod_degree=8192,
                   coeff_mod_bit_sizes=None, global_scale=2**40):
    """Create a TenSEAL encryption context.

    Args:
        scheme: "ckks" (approximate, for ML) or "bfv" (exact integers).
        poly_mod_degree: Ring dimension (4096, 8192, 16384).
            Higher = more multiplicative depth, slower.
        coeff_mod_bit_sizes: Coefficient modulus chain.
        global_scale: Encoding precision for CKKS.

    Returns:
        TenSEAL context object.
    """
    scheme_type = ts.SCHEME_TYPE.CKKS if scheme == "ckks" else ts.SCHEME_TYPE.BFV

    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]

    ctx = ts.context(
        scheme_type,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )

    if scheme == "ckks":
        ctx.global_scale = global_scale

    ctx.generate_galois_keys()
    return ctx


def encrypt(ctx, values):
    """Encrypt a list of floats under CKKS.

    Args:
        ctx: TenSEAL context from create_context().
        values: List of float values.

    Returns:
        TenSEAL CKKSVector (encrypted). Supports +, -, * operations.
    """
    return ts.ckks_vector(ctx, values)


def decrypt(encrypted):
    """Decrypt a CKKS-encrypted vector.

    Args:
        encrypted: TenSEAL CKKSVector.

    Returns:
        List of float values.
    """
    return encrypted.decrypt()


def make_context_public(ctx):
    """Drop the secret key — context can only encrypt, not decrypt.

    Use this when sending the context to an untrusted server for
    computation on encrypted data.
    """
    ctx.make_context_public()
    return ctx


__all__ = [
    "create_context",
    "encrypt",
    "decrypt",
    "make_context_public",
]
