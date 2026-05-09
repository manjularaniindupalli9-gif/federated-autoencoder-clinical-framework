"""
Loss functions for the Federated Autoencoder-Based Clinical Decision Framework.

This module contains:
- Focal loss for imbalanced multi-class disease classification
- Binary cross-entropy loss for binary clinical outcome prediction
- Mean squared reconstruction loss for autoencoder training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced multi-class classification.

    Focal loss reduces the contribution of easy majority-class samples and
    increases the relative importance of hard or underrepresented samples.

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha:
        Class-balancing factor. Can be a float or class-wise tensor.
    gamma:
        Focusing parameter. Higher values increase focus on hard samples.
    reduction:
        Reduction method: "mean", "sum", or "none".
    eps:
        Small numerical stability constant.
    """

    def __init__(
        self,
        alpha: float | torch.Tensor = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError("gamma must be non-negative.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss from raw logits and integer class labels.

        Parameters
        ----------
        logits:
            Raw model logits with shape [batch_size, num_classes].
        targets:
            Ground-truth class labels with shape [batch_size].

        Returns
        -------
        torch.Tensor
            Focal loss value.
        """
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch_size, num_classes].")

        if targets.ndim != 1:
            raise ValueError("targets must have shape [batch_size].")

        if logits.shape[0] != targets.shape[0]:
            raise ValueError("logits and targets must have the same batch size.")

        targets = targets.long()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        target_probs = torch.clamp(target_probs, min=self.eps, max=1.0 - self.eps)

        focal_factor = torch.pow(1.0 - target_probs, self.gamma)

        alpha_factor = self._get_alpha_factor(
            targets=targets,
            logits=logits,
        )

        loss = -alpha_factor * focal_factor * target_log_probs

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss

    def _get_alpha_factor(
        self,
        targets: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return alpha weighting factor for each target sample.

        Parameters
        ----------
        targets:
            Target class labels.
        logits:
            Model logits used to determine device placement.

        Returns
        -------
        torch.Tensor
            Alpha factor for each sample.
        """
        if isinstance(self.alpha, torch.Tensor):
            alpha_tensor = self.alpha.to(device=logits.device, dtype=logits.dtype)

            if alpha_tensor.ndim != 1:
                raise ValueError("Class-wise alpha tensor must be one-dimensional.")

            if alpha_tensor.shape[0] != logits.shape[1]:
                raise ValueError(
                    "Class-wise alpha tensor length must match number of classes."
                )

            return alpha_tensor.gather(dim=0, index=targets)

        return torch.full(
            size=targets.shape,
            fill_value=float(self.alpha),
            device=logits.device,
            dtype=logits.dtype,
        )


def binary_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Binary cross-entropy loss for clinical outcome prediction.

    The function expects raw logits. Sigmoid activation is internally handled by
    PyTorch's BCEWithLogits formulation for numerical stability.

    Parameters
    ----------
    logits:
        Raw binary logits with shape [batch_size] or [batch_size, 1].
    targets:
        Binary ground-truth labels with shape [batch_size] or [batch_size, 1].
    pos_weight:
        Optional positive-class weight for imbalanced binary labels.

    Returns
    -------
    torch.Tensor
        Binary cross-entropy loss.
    """
    logits = logits.view(-1)
    targets = targets.float().view(-1)

    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same number of samples.")

    return F.binary_cross_entropy_with_logits(
        input=logits,
        target=targets,
        pos_weight=pos_weight,
    )


def reconstruction_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error reconstruction loss for autoencoder training.

    This loss encourages the latent representation to preserve informative
    clinical structure from the original input features.

    Parameters
    ----------
    reconstruction:
        Reconstructed feature tensor.
    original:
        Original input feature tensor.

    Returns
    -------
    torch.Tensor
        Mean squared reconstruction loss.
    """
    if reconstruction.shape != original.shape:
        raise ValueError(
            "reconstruction and original tensors must have identical shapes."
        )

    return F.mse_loss(reconstruction, original)


def weighted_multitask_loss(
    reconstruction: torch.Tensor,
    original: torch.Tensor,
    multiclass_logits: torch.Tensor,
    multiclass_targets: torch.Tensor,
    binary_logits: torch.Tensor,
    binary_targets: torch.Tensor,
    focal_loss_fn: FocalLoss,
    reconstruction_weight: float = 0.40,
    multiclass_weight: float = 1.00,
    binary_weight: float = 0.80,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the complete weighted training objective.

    The combined objective contains:
    - autoencoder reconstruction loss
    - focal loss for disease classification
    - binary cross-entropy loss for outcome prediction

    Parameters
    ----------
    reconstruction:
        Autoencoder reconstructed features.
    original:
        Original input features.
    multiclass_logits:
        Multi-class disease logits.
    multiclass_targets:
        Disease labels.
    binary_logits:
        Binary outcome logits.
    binary_targets:
        Binary outcome labels.
    focal_loss_fn:
        Focal loss instance.
    reconstruction_weight:
        Weight for reconstruction loss.
    multiclass_weight:
        Weight for multi-class classification loss.
    binary_weight:
        Weight for binary prediction loss.

    Returns
    -------
    tuple[torch.Tensor, dict[str, float]]
        Total weighted loss and detached component losses.
    """
    rec_loss = reconstruction_loss(reconstruction, original)
    mc_loss = focal_loss_fn(multiclass_logits, multiclass_targets)
    bin_loss = binary_cross_entropy_loss(binary_logits, binary_targets)

    total_loss = (
        reconstruction_weight * rec_loss
        + multiclass_weight * mc_loss
        + binary_weight * bin_loss
    )

    loss_parts = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "reconstruction_loss": float(rec_loss.detach().cpu().item()),
        "multiclass_loss": float(mc_loss.detach().cpu().item()),
        "binary_loss": float(bin_loss.detach().cpu().item()),
    }

    return total_loss, loss_parts
