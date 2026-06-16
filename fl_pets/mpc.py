"""Secure Multi-Party Computation for FL.

Primary library: CrypTen 0.4+ (Meta)
    https://github.com/facebookresearch/CrypTen
    pip install crypten
    PyTorch-native MPC. Validated on DenseNet-121, ResNet-50, BERT.

Alternative libraries:
    - SecretFlow/SPU (Ant Group): ABY3, Semi2k, Cheetah. LLaMA-7B capable.
      https://github.com/secretflow/secretflow
    - MP-SPDZ: research-grade, supports 30+ MPC protocols.
      https://github.com/data61/MP-SPDZ
    - Concrete ML (Zama): FHE-based, single-server, no round-trips.
      https://github.com/zama-ai/concrete-ml

This module provides:
    - CrypTen integration (if installed)
    - Encrypted tensor operations over MPC

Usage:
    from fl_pets.mpc import encrypt_tensor, decrypt_tensor

    # CrypTen MPC
    encrypted = encrypt_tensor(plain_tensor)
    result = encrypted + encrypted
    plain = decrypt_tensor(result)
"""

try:
    import crypten
    crypten.init()
    CRYPTEN_AVAILABLE = True
except (ImportError, Exception):
    crypten = None
    CRYPTEN_AVAILABLE = False


def encrypt_tensor(tensor):
    """Encrypt a PyTorch tensor using CrypTen MPC.

    Args:
        tensor: PyTorch tensor to encrypt.

    Returns:
        CrypTen encrypted tensor. Supports arithmetic operations.
    """
    if not CRYPTEN_AVAILABLE:
        raise ImportError(
            "CrypTen not installed. Run: pip install crypten\n"
            "Docs: https://github.com/facebookresearch/CrypTen"
        )
    return crypten.cryptensor(tensor)


def decrypt_tensor(encrypted):
    """Decrypt a CrypTen encrypted tensor.

    Args:
        encrypted: CrypTen CrypTensor.

    Returns:
        PyTorch tensor (plaintext).
    """
    return encrypted.get_plain_text()


def is_available():
    """Check if CrypTen MPC is available."""
    return CRYPTEN_AVAILABLE


__all__ = [
    "encrypt_tensor",
    "decrypt_tensor",
    "is_available",
    "CRYPTEN_AVAILABLE",
]
