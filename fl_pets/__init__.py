"""FL Privacy-Enhancing Technologies (PET) Toolkit.

Production-grade PET modules organised by FL lifecycle stage.

Pre-training:
    psi        — Entity alignment across parties before VFL training
                 Library: OpenMined PSI / HMAC-SHA256 ECDH-PSI

During training:
    dp         — Per-round differential privacy (gradient clipping + noise)
                 Library: Opacus 1.6+ (Meta)
    secagg     — Secure aggregation (hide individual updates from server)
                 Library: Flower SecAgg+

Post-training:
    privacy/   — Privacy attack suite (MIA, gradient leakage, model inversion)
                 Validates that DP/SecAgg controls are effective before release

Inference:
    he         — Encrypted inference on sensitive data (no decryption needed)
                 Library: TenSEAL 0.3+ (OpenMined) / Microsoft SEAL
    mpc        — Multi-party inference (split computation, no single party sees all)
                 Library: CrypTen 0.4+ (Meta)

Future (roadmap v0.3+):
    tee        — Trusted execution environments (AWS Nitro, Intel SGX)

Usage:
    # Pre-training
    from fl_pets.psi import align_entities

    # During training
    from fl_pets.dp import make_private, compute_epsilon
    from fl_pets.secagg import mask_parameters

    # Inference
    from fl_pets.he import create_context, encrypt, decrypt
    from fl_pets.mpc import encrypt_tensor
"""

# Pre-training
from fl_pets import psi

# During training
from fl_pets import dp
from fl_pets import secagg

# Inference
from fl_pets import he
from fl_pets import mpc
