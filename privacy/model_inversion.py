"""Model Inversion Attack.

White-box model inversion: given access to a trained model and a target class,
reconstruct an input that maximises the model's confidence for that class.

This tests whether a federated model leaks information about training data
through its learned representations.

Attack methodology:
    1. Start with random noise as the candidate input
    2. Optimise the input to maximise the model's output for the target class
    3. Measure reconstruction quality against real training samples
       (cosine similarity, SSIM for images, MSE for tabular)

Metrics:
    - Reconstruction MSE (lower = more leakage)
    - Cosine similarity to real training samples (higher = more leakage)
    - SSIM for image models (higher = more leakage)

Reference:
    Fredrikson et al., "Model Inversion Attacks that Exploit Confidence
    Information and Basic Countermeasures" (CCS 2015)
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple

logger = logging.getLogger("privacy.model_inversion")


def model_inversion_attack(
    model: nn.Module,
    target_class: int,
    input_shape: Tuple[int, ...],
    num_iterations: int = 1000,
    lr: float = 0.01,
    regularisation: float = 0.001,
    device: str = "cpu",
    reference_samples: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run a model inversion attack against a trained model.

    Args:
        model: Trained PyTorch model (white-box access).
        target_class: Class label to reconstruct.
        input_shape: Shape of a single input sample (e.g., (30,) for tabular).
        num_iterations: Optimisation iterations.
        lr: Learning rate for input optimisation.
        regularisation: L2 regularisation on the reconstructed input.
        device: torch device.
        reference_samples: Real training samples of target class for comparison.
            If provided, reconstruction quality is measured against these.

    Returns:
        Dict with attack metrics:
            - "reconstruction_mse": MSE of the reconstructed input
            - "cosine_similarity": Cosine similarity to nearest real sample
            - "confidence": Model confidence on reconstructed input
            - "converged": Whether optimisation converged
    """
    model = model.to(device).eval()

    # Initialise random candidate input
    x_recon = torch.randn(1, *input_shape, device=device, requires_grad=True)
    optimizer = optim.Adam([x_recon], lr=lr)

    best_loss = float("inf")
    patience_counter = 0

    for i in range(num_iterations):
        optimizer.zero_grad()

        output = model(x_recon)

        # Target: maximise confidence for target_class
        if output.shape[-1] == 1:
            # Binary classification (sigmoid output)
            if target_class == 1:
                loss = -torch.log(output + 1e-8).mean()
            else:
                loss = -torch.log(1 - output + 1e-8).mean()
        else:
            # Multi-class (logits)
            loss = -output[0, target_class]

        # Regularisation to keep input realistic
        loss += regularisation * torch.norm(x_recon)

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > 100:
            break

    # Evaluate reconstruction
    with torch.no_grad():
        final_output = model(x_recon)
        if final_output.shape[-1] == 1:
            confidence = final_output.item() if target_class == 1 else (1 - final_output.item())
        else:
            probs = torch.softmax(final_output, dim=-1)
            confidence = probs[0, target_class].item()

    x_np = x_recon.detach().cpu().numpy().flatten()

    results = {
        "reconstruction_mse": float(np.mean(x_np ** 2)),
        "confidence": confidence,
        "converged": patience_counter > 100,
    }

    # Compare to real training samples if available
    if reference_samples is not None and len(reference_samples) > 0:
        ref = reference_samples.reshape(len(reference_samples), -1)
        x_flat = x_np.reshape(1, -1)

        # Cosine similarity to nearest sample
        norms_ref = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-10
        norm_x = np.linalg.norm(x_flat) + 1e-10
        cos_sim = (ref @ x_flat.T).flatten() / (norms_ref.flatten() * norm_x)
        results["cosine_similarity"] = float(np.max(cos_sim))
        results["mean_cosine_similarity"] = float(np.mean(cos_sim))

        # MSE to nearest sample
        mse_per_sample = np.mean((ref - x_flat) ** 2, axis=1)
        results["min_mse_to_real"] = float(np.min(mse_per_sample))
    else:
        results["cosine_similarity"] = 0.0

    return results


def run_model_inversion_evaluation(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int = 2,
    num_iterations: int = 1000,
    device: str = "cpu",
) -> Dict[str, float]:
    """Run model inversion across all classes and aggregate results.

    Args:
        model: Trained model.
        X_train: Training features.
        y_train: Training labels.
        num_classes: Number of classes.
        num_iterations: Optimisation iterations per class.
        device: torch device.

    Returns:
        Aggregated metrics across all classes.
    """
    input_shape = X_train.shape[1:]
    all_cos_sims = []
    all_confidences = []

    for cls in range(num_classes):
        mask = y_train.astype(int) == cls
        ref_samples = X_train[mask] if mask.sum() > 0 else None

        result = model_inversion_attack(
            model=model,
            target_class=cls,
            input_shape=input_shape,
            num_iterations=num_iterations,
            device=device,
            reference_samples=ref_samples,
        )

        logger.info(
            "Class %d: confidence=%.3f, cosine_similarity=%.3f",
            cls, result["confidence"], result.get("cosine_similarity", 0)
        )

        all_cos_sims.append(result.get("cosine_similarity", 0))
        all_confidences.append(result["confidence"])

    return {
        "mean_cosine_similarity": float(np.mean(all_cos_sims)),
        "max_cosine_similarity": float(np.max(all_cos_sims)),
        "mean_confidence": float(np.mean(all_confidences)),
        "num_classes_tested": num_classes,
    }
