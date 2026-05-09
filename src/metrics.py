"""
Evaluation and plotting utilities for the Federated Autoencoder-Based Clinical Decision Framework.

This module contains:
- multi-class disease classification metrics
- binary clinical outcome prediction metrics
- confusion matrix export
- ROC curve export
- federated training curve export
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_multiclass_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> Dict[str, float | np.ndarray]:
    """
    Evaluate multi-class disease classification performance.

    Parameters
    ----------
    model:
        Trained federated clinical model.
    loader:
        DataLoader containing validation or test samples.
    device:
        CPU or GPU device.
    return_predictions:
        If True, return ground-truth and predicted labels.

    Returns
    -------
    Dict[str, float | np.ndarray]
        Multi-class evaluation metrics.
    """
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["disease_label"].cpu().numpy()

            outputs = model(features)
            logits = outputs["multiclass_logits"]
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    metrics: Dict[str, float | np.ndarray] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_macro": float(
            precision_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true_array,
                y_pred_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true_array,
                y_pred_array,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    if return_predictions:
        metrics["y_true"] = y_true_array
        metrics["y_pred"] = y_pred_array

    return metrics


def evaluate_binary_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    return_predictions: bool = False,
) -> Dict[str, float | np.ndarray]:
    """
    Evaluate binary clinical outcome prediction performance.

    Parameters
    ----------
    model:
        Trained federated clinical model.
    loader:
        DataLoader containing validation or test samples.
    device:
        CPU or GPU device.
    threshold:
        Decision threshold applied to sigmoid probabilities.
    return_predictions:
        If True, return ground-truth labels, predictions, and probability scores.

    Returns
    -------
    Dict[str, float | np.ndarray]
        Binary evaluation metrics.
    """
    model.eval()

    y_true: List[int] = []
    y_score: List[float] = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["outcome_label"].cpu().numpy()

            outputs = model(features)
            logits = outputs["binary_logits"].squeeze(1)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(labels.astype(int).tolist())
            y_score.extend(probabilities.tolist())

    y_true_array = np.asarray(y_true).astype(int)
    y_score_array = np.asarray(y_score)
    y_pred_array = (y_score_array >= threshold).astype(int)

    tn, fp, fn, tp = safe_binary_confusion_values(
        y_true=y_true_array,
        y_pred=y_pred_array,
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true_array)) > 1:
        roc_auc = float(roc_auc_score(y_true_array, y_score_array))
    else:
        roc_auc = float("nan")

    metrics: Dict[str, float | np.ndarray] = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(
            precision_score(
                y_true_array,
                y_pred_array,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_array,
                y_pred_array,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "roc_auc": roc_auc,
    }

    if return_predictions:
        metrics["y_true"] = y_true_array
        metrics["y_pred"] = y_pred_array
        metrics["y_score"] = y_score_array

    return metrics


def safe_binary_confusion_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Return binary confusion matrix values in a stable format.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.

    Returns
    -------
    tuple[int, int, int, int]
        True negatives, false positives, false negatives, and true positives.
    """
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return int(tn), int(fp), int(fn), int(tp)


def save_multiclass_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save the multi-class confusion matrix figure.

    Parameters
    ----------
    y_true:
        Ground-truth disease labels.
    y_pred:
        Predicted disease labels.
    output_path:
        Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Multi-Class Disease Classification Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_binary_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save the binary outcome prediction confusion matrix figure.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    output_path:
        Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Negative", "Positive"],
    )
    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Binary Clinical Outcome Confusion Matrix")
    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("True Outcome")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save ROC curve for binary clinical outcome prediction.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_score:
        Predicted positive-class probabilities.
    output_path:
        Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    if len(np.unique(y_true)) > 1:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        roc_value = auc(false_positive_rate, true_positive_rate)

        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"ROC-AUC = {roc_value:.4f}",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1,
            label="Chance",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "ROC curve unavailable: only one class present.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_title("Binary Outcome ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_curve(
    round_history: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save federated training and validation trend figure.

    Parameters
    ----------
    round_history:
        DataFrame containing federated round-wise metrics.
    output_path:
        Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {
        "round",
        "train_loss",
        "validation_multiclass_accuracy",
        "validation_binary_accuracy",
    }

    missing_columns = required_columns.difference(round_history.columns)

    if missing_columns:
        raise ValueError(
            f"Round history is missing required columns: {missing_columns}"
        )

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        round_history["round"],
        round_history["train_loss"],
        marker="o",
        linewidth=2,
        label="Training Loss",
    )
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        round_history["round"],
        round_history["validation_multiclass_accuracy"],
        marker="s",
        linewidth=2,
        label="Multi-Class Validation Accuracy",
    )
    ax2.plot(
        round_history["round"],
        round_history["validation_binary_accuracy"],
        marker="^",
        linewidth=2,
        label="Binary Validation Accuracy",
    )
    ax2.set_ylabel("Validation Accuracy")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best",
    )

    ax1.set_title("Federated Training Convergence")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_multiclass_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Build a class-wise multi-class performance report.

    Parameters
    ----------
    y_true:
        Ground-truth disease labels.
    y_pred:
        Predicted disease labels.

    Returns
    -------
    pd.DataFrame
        Class-wise precision, recall, F1-score, and support.
    """
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "class_label": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )


def build_binary_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> pd.DataFrame:
    """
    Build a binary prediction performance report.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    y_score:
        Predicted positive-class probabilities.

    Returns
    -------
    pd.DataFrame
        Binary performance report.
    """
    tn, fp, fn, tp = safe_binary_confusion_values(y_true, y_pred)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_score)
    else:
        roc_auc = np.nan

    return pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall_sensitivity": sensitivity,
                "specificity": specificity,
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc": roc_auc,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
            }
        ]
    )
