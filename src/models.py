"""
Model definitions for the Federated Autoencoder-Based Clinical Decision Framework.

This module contains:
- Autoencoder for latent clinical feature extraction
- Multi-class disease classifier
- Binary clinical outcome predictor
- Integrated federated clinical model

The model follows a shared-representation design:
clinical features -> autoencoder encoder -> latent representation -> two prediction heads
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Autoencoder for structured clinical feature compression.

    The encoder learns a compact latent representation from high-dimensional
    clinical features. The decoder reconstructs the original feature vector so
    that the latent space preserves clinically meaningful information.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero.")

        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than zero.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input clinical features into latent representation.

        Parameters
        ----------
        x:
            Input feature tensor of shape [batch_size, input_dim].

        Returns
        -------
        torch.Tensor
            Latent feature tensor of shape [batch_size, latent_dim].
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to reconstructed input space.

        Parameters
        ----------
        z:
            Latent feature tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed feature tensor.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder and decoder.

        Parameters
        ----------
        x:
            Input clinical feature tensor.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing latent representation and reconstruction.
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)

        return {
            "latent": latent,
            "reconstruction": reconstruction,
        }


class MultiClassDiseaseClassifier(nn.Module):
    """
    Multi-class disease classifier.

    This classifier receives the autoencoder latent representation and predicts
    one disease category among multiple diagnostic classes.
    """

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than zero.")

        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict disease logits from latent clinical features.

        Parameters
        ----------
        latent:
            Latent clinical representation.

        Returns
        -------
        torch.Tensor
            Multi-class disease logits.
        """
        return self.classifier(latent)


class BinaryOutcomePredictor(nn.Module):
    """
    Binary clinical outcome predictor.

    This prediction head receives the same autoencoder latent representation and
    estimates the probability of a positive clinical outcome using a single-logit
    output. Sigmoid activation is applied during metric evaluation, not here.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than zero.")

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Predict binary outcome logits from latent clinical features.

        Parameters
        ----------
        latent:
            Latent clinical representation.

        Returns
        -------
        torch.Tensor
            Binary outcome logits of shape [batch_size, 1].
        """
        return self.predictor(latent)


class FederatedClinicalModel(nn.Module):
    """
    Integrated clinical model for federated training.

    The model contains:
    - shared autoencoder feature extractor
    - multi-class disease classification head
    - binary outcome prediction head

    This design allows both predictive tasks to use the same learned clinical
    representation while keeping task-specific decision layers independent.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_classes: int,
        ae_hidden_dim_1: int = 128,
        ae_hidden_dim_2: int = 64,
        ae_dropout: float = 0.20,
        classifier_hidden_dim: int = 64,
        classifier_dropout: float = 0.30,
    ) -> None:
        super().__init__()

        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim_1=ae_hidden_dim_1,
            hidden_dim_2=ae_hidden_dim_2,
            dropout=ae_dropout,
        )

        self.multiclass_classifier = MultiClassDiseaseClassifier(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
        )

        self.binary_predictor = BinaryOutcomePredictor(
            latent_dim=latent_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete clinical model.

        Parameters
        ----------
        x:
            Input clinical feature tensor.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - latent
            - reconstruction
            - multiclass_logits
            - binary_logits
        """
        ae_outputs = self.autoencoder(x)

        latent = ae_outputs["latent"]
        reconstruction = ae_outputs["reconstruction"]

        multiclass_logits = self.multiclass_classifier(latent)
        binary_logits = self.binary_predictor(latent)

        return {
            "latent": latent,
            "reconstruction": reconstruction,
            "multiclass_logits": multiclass_logits,
            "binary_logits": binary_logits,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent clinical features using the trained encoder.

        Parameters
        ----------
        x:
            Input clinical feature tensor.

        Returns
        -------
        torch.Tensor
            Latent clinical representation.
        """
        return self.autoencoder.encode(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Parameters
    ----------
    model:
        PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
