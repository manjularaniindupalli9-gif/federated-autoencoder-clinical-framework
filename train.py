"""
Main execution script for the Federated Autoencoder-Based Clinical Decision Framework.

This script supports two execution modes:

1. synthetic:
   Generates a structured clinical-like dataset for public reproducibility.

2. restricted:
   Loads an authorized private clinical CSV file from a local secure path.

The original clinical dataset is not distributed with this repository.
"""

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    apply_smote_balancing,
    create_client_loaders,
    generate_synthetic_clinical_data,
    load_restricted_dataset,
    preprocess_clinical_data,
    split_non_iid_clients,
    train_test_validation_split,
)
from src.federated import adaptive_fedavg, add_update_noise, get_model_state
from src.losses import FocalLoss, binary_cross_entropy_loss, reconstruction_loss
from src.metrics import (
    evaluate_binary_model,
    evaluate_multiclass_model,
    save_binary_confusion_matrix,
    save_multiclass_confusion_matrix,
    save_roc_curve,
    save_training_curve,
)
from src.models import FederatedClinicalModel
from src.utils import (
    get_device,
    load_yaml_config,
    make_output_dir,
    save_json,
    save_table,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated Autoencoder-Based Clinical Decision Framework"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["synthetic", "restricted"],
        help="Execution mode. Overrides config.yaml if provided.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to restricted clinical CSV file. Required only for restricted mode.",
    )

    return parser.parse_args()


def load_dataset(config: Dict, mode: str, data_path: str | None) -> pd.DataFrame:
    """
    Load either synthetic reproducibility data or restricted clinical data.

    Parameters
    ----------
    config:
        Project configuration dictionary.
    mode:
        Execution mode: synthetic or restricted.
    data_path:
        Local path to restricted clinical CSV file.

    Returns
    -------
    pd.DataFrame
        Clinical dataset containing features and target columns.
    """
    multiclass_col = config["data"]["target_columns"]["multiclass"]
    binary_col = config["data"]["target_columns"]["binary"]

    if mode == "synthetic":
        return generate_synthetic_clinical_data(
            n_samples=config["data"]["synthetic_samples"],
            n_classes=config["data"]["num_disease_classes"],
            multiclass_col=multiclass_col,
            binary_col=binary_col,
            seed=config["execution"]["seed"],
        )

    if mode == "restricted":
        restricted_path = data_path or config["data"]["restricted_data"]["path"]

        if restricted_path is None:
            raise ValueError(
                "Restricted mode requires --data-path or data.restricted_data.path in config.yaml."
            )

        return load_restricted_dataset(
            data_path=restricted_path,
            multiclass_col=multiclass_col,
            binary_col=binary_col,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def train_one_epoch(
    model: FederatedClinicalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    focal_loss_fn: FocalLoss,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Train one local client model for one epoch.

    The joint objective includes:
    - autoencoder reconstruction loss
    - focal loss for multi-class disease classification
    - binary cross-entropy loss for binary outcome prediction
    """
    model.train()

    total_loss = 0.0
    total_rec_loss = 0.0
    total_mc_loss = 0.0
    total_bin_loss = 0.0
    total_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        disease_labels = batch["disease_label"].to(device)
        outcome_labels = batch["outcome_label"].float().to(device)

        optimizer.zero_grad()

        outputs = model(features)

        rec_loss = reconstruction_loss(outputs["reconstruction"], features)
        mc_loss = focal_loss_fn(outputs["multiclass_logits"], disease_labels)
        bin_loss = binary_cross_entropy_loss(
            outputs["binary_logits"].squeeze(1),
            outcome_labels,
        )

        loss = (
            loss_weights["reconstruction_weight"] * rec_loss
            + loss_weights["multiclass_weight"] * mc_loss
            + loss_weights["binary_weight"] * bin_loss
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rec_loss += rec_loss.item()
        total_mc_loss += mc_loss.item()
        total_bin_loss += bin_loss.item()
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "reconstruction_loss": total_rec_loss / max(total_batches, 1),
        "multiclass_loss": total_mc_loss / max(total_batches, 1),
        "binary_loss": total_bin_loss / max(total_batches, 1),
    }


def train_local_client(
    global_model: FederatedClinicalModel,
    loader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float]]:
    """
    Train a local client model initialized from the current global model.

    Returns
    -------
    Tuple containing:
    - trained local model state dictionary
    - number of samples in the client
    - average local training losses
    """
    local_model = copy.deepcopy(global_model).to(device)

    optimizer = torch.optim.Adam(
        local_model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    focal_loss_fn = FocalLoss(
        alpha=config["loss"]["focal_alpha"],
        gamma=config["loss"]["focal_gamma"],
    )

    loss_weights = {
        "reconstruction_weight": config["loss"]["reconstruction_weight"],
        "multiclass_weight": config["loss"]["multiclass_weight"],
        "binary_weight": config["loss"]["binary_weight"],
    }

    epoch_losses: List[Dict[str, float]] = []

    for _ in range(config["federated"]["local_epochs"]):
        epoch_loss = train_one_epoch(
            model=local_model,
            loader=loader,
            optimizer=optimizer,
            focal_loss_fn=focal_loss_fn,
            device=device,
            loss_weights=loss_weights,
        )
        epoch_losses.append(epoch_loss)

    avg_losses = {
        key: float(np.mean([loss[key] for loss in epoch_losses]))
        for key in epoch_losses[0].keys()
    }

    return get_model_state(local_model), len(loader.dataset), avg_losses


def run_federated_training(
    model: FederatedClinicalModel,
    client_loaders: List[DataLoader],
    validation_loader: DataLoader,
    config: Dict,
    device: torch.device,
) -> Tuple[FederatedClinicalModel, pd.DataFrame]:
    """
    Run federated training across multiple clinical clients.

    Each federated round performs:
    1. local client training
    2. optional privacy-preserving update perturbation
    3. weighted FedAvg aggregation
    4. validation performance logging
    """
    round_records = []

    model = model.to(device)

    for round_idx in range(1, config["federated"]["federated_rounds"] + 1):
        client_states = []
        client_sizes = []
        client_losses = []

        for loader in client_loaders:
            local_state, client_size, local_losses = train_local_client(
                global_model=model,
                loader=loader,
                config=config,
                device=device,
            )

            if config["privacy"]["enable_update_noise"]:
                local_state = add_update_noise(
                    state_dict=local_state,
                    noise_std=config["privacy"]["dp_noise_std"],
                    device=device,
                )

            client_states.append(local_state)
            client_sizes.append(client_size)
            client_losses.append(local_losses)

        aggregated_state = adaptive_fedavg(
            client_states=client_states,
            client_sizes=client_sizes,
        )

        model.load_state_dict(aggregated_state)

        multiclass_metrics = evaluate_multiclass_model(
            model=model,
            loader=validation_loader,
            device=device,
        )

        binary_metrics = evaluate_binary_model(
            model=model,
            loader=validation_loader,
            device=device,
        )

        avg_train_loss = float(np.mean([loss["loss"] for loss in client_losses]))
        avg_rec_loss = float(
            np.mean([loss["reconstruction_loss"] for loss in client_losses])
        )
        avg_mc_loss = float(np.mean([loss["multiclass_loss"] for loss in client_losses]))
        avg_bin_loss = float(np.mean([loss["binary_loss"] for loss in client_losses]))

        round_records.append(
            {
                "round": round_idx,
                "train_loss": avg_train_loss,
                "reconstruction_loss": avg_rec_loss,
                "multiclass_loss": avg_mc_loss,
                "binary_loss": avg_bin_loss,
                "validation_multiclass_accuracy": multiclass_metrics["accuracy"],
                "validation_multiclass_f1_macro": multiclass_metrics["f1_macro"],
                "validation_binary_accuracy": binary_metrics["accuracy"],
                "validation_binary_f1": binary_metrics["f1"],
                "validation_binary_auc": binary_metrics["roc_auc"],
            }
        )

        print(
            f"Round {round_idx:02d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"MC Acc: {multiclass_metrics['accuracy']:.4f} | "
            f"BIN Acc: {binary_metrics['accuracy']:.4f} | "
            f"BIN AUC: {binary_metrics['roc_auc']:.4f}"
        )

    return model, pd.DataFrame(round_records)


def save_all_outputs(
    model: FederatedClinicalModel,
    test_loader: DataLoader,
    round_history: pd.DataFrame,
    config: Dict,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Evaluate final model and save all metrics, tables, and figures."""
    multiclass_metrics = evaluate_multiclass_model(
        model=model,
        loader=test_loader,
        device=device,
        return_predictions=True,
    )

    binary_metrics = evaluate_binary_model(
        model=model,
        loader=test_loader,
        device=device,
        return_predictions=True,
    )

    metrics_summary = {
        "multiclass": {
            "accuracy": multiclass_metrics["accuracy"],
            "precision_macro": multiclass_metrics["precision_macro"],
            "recall_macro": multiclass_metrics["recall_macro"],
            "f1_macro": multiclass_metrics["f1_macro"],
            "precision_weighted": multiclass_metrics["precision_weighted"],
            "recall_weighted": multiclass_metrics["recall_weighted"],
            "f1_weighted": multiclass_metrics["f1_weighted"],
        },
        "binary": {
            "accuracy": binary_metrics["accuracy"],
            "precision": binary_metrics["precision"],
            "recall": binary_metrics["recall"],
            "f1": binary_metrics["f1"],
            "specificity": binary_metrics["specificity"],
            "roc_auc": binary_metrics["roc_auc"],
        },
        "federated": {
            "num_clients": config["federated"]["num_clients"],
            "federated_rounds": config["federated"]["federated_rounds"],
            "local_epochs": config["federated"]["local_epochs"],
            "aggregation": config["federated"]["aggregation"],
        },
    }

    files = config["outputs"]["files"]

    save_json(
        metrics_summary,
        output_dir / files["metrics_summary"],
    )

    save_table(
        pd.DataFrame([metrics_summary["multiclass"]]),
        output_dir / files["multiclass_metrics"],
    )

    save_table(
        pd.DataFrame([metrics_summary["binary"]]),
        output_dir / files["binary_metrics"],
    )

    save_table(
        round_history,
        output_dir / files["round_history"],
    )

    save_multiclass_confusion_matrix(
        y_true=multiclass_metrics["y_true"],
        y_pred=multiclass_metrics["y_pred"],
        output_path=output_dir / files["multiclass_confusion_matrix"],
    )

    save_binary_confusion_matrix(
        y_true=binary_metrics["y_true"],
        y_pred=binary_metrics["y_pred"],
        output_path=output_dir / files["binary_confusion_matrix"],
    )

    save_roc_curve(
        y_true=binary_metrics["y_true"],
        y_score=binary_metrics["y_score"],
        output_path=output_dir / files["binary_roc_curve"],
    )

    save_training_curve(
        round_history=round_history,
        output_path=output_dir / files["training_curve"],
    )

    print("\nFinal Multi-Class Metrics")
    print(pd.DataFrame([metrics_summary["multiclass"]]).round(4).to_string(index=False))

    print("\nFinal Binary Metrics")
    print(pd.DataFrame([metrics_summary["binary"]]).round(4).to_string(index=False))

    print(f"\nSaved outputs to: {output_dir}")


def main() -> None:
    """Run the complete federated clinical learning pipeline."""
    args = parse_args()

    config = load_yaml_config(args.config)

    if args.mode is not None:
        config["execution"]["mode"] = args.mode

    set_seed(config["execution"]["seed"])

    device = get_device(config["execution"]["device"])

    output_dir = make_output_dir(config["project"]["output_dir"])

    print("Federated Autoencoder-Based Clinical Decision Framework")
    print(f"Execution mode : {config['execution']['mode']}")
    print(f"Device         : {device}")
    print(f"Output folder  : {output_dir}")

    dataset = load_dataset(
        config=config,
        mode=config["execution"]["mode"],
        data_path=args.data_path,
    )

    multiclass_col = config["data"]["target_columns"]["multiclass"]
    binary_col = config["data"]["target_columns"]["binary"]

    x, y_multiclass, y_binary, feature_names = preprocess_clinical_data(
        dataset=dataset,
        multiclass_col=multiclass_col,
        binary_col=binary_col,
        scaling=config["preprocessing"]["scaling"],
    )

    if config["preprocessing"]["apply_smote"]:
        x, y_multiclass, y_binary = apply_smote_balancing(
            x=x,
            y_multiclass=y_multiclass,
            y_binary=y_binary,
            k_neighbors=config["preprocessing"]["smote_k_neighbors"],
            seed=config["execution"]["seed"],
        )

    split_data = train_test_validation_split(
        x=x,
        y_multiclass=y_multiclass,
        y_binary=y_binary,
        test_size=config["data"]["test_size"],
        validation_size=config["data"]["validation_size"],
        seed=config["execution"]["seed"],
    )

    client_indices = split_non_iid_clients(
        labels=split_data["y_train_multiclass"],
        num_clients=config["federated"]["num_clients"],
        alpha=config["federated"]["non_iid_alpha"],
        min_client_samples=config["federated"]["min_client_samples"],
        seed=config["execution"]["seed"],
    )

    client_loaders, validation_loader, test_loader = create_client_loaders(
        split_data=split_data,
        client_indices=client_indices,
        batch_size=config["training"]["batch_size"],
    )

    input_dim = split_data["x_train"].shape[1]
    num_classes = int(len(np.unique(y_multiclass)))

    config["model"]["input_dim"] = int(input_dim)
    config["data"]["num_disease_classes"] = int(num_classes)

    model = FederatedClinicalModel(
        input_dim=input_dim,
        latent_dim=config["model"]["latent_dim"],
        num_classes=num_classes,
        ae_hidden_dim_1=config["model"]["autoencoder"]["hidden_dim_1"],
        ae_hidden_dim_2=config["model"]["autoencoder"]["hidden_dim_2"],
        ae_dropout=config["model"]["autoencoder"]["dropout"],
        classifier_hidden_dim=config["model"]["classifier"]["hidden_dim"],
        classifier_dropout=config["model"]["classifier"]["dropout"],
    )

    print(f"Features       : {input_dim}")
    print(f"Disease classes: {num_classes}")
    print(f"Clients        : {len(client_loaders)}")
    print(f"Train samples  : {len(split_data['x_train'])}")
    print(f"Valid samples  : {len(split_data['x_valid'])}")
    print(f"Test samples   : {len(split_data['x_test'])}")

    trained_model, round_history = run_federated_training(
        model=model,
        client_loaders=client_loaders,
        validation_loader=validation_loader,
        config=config,
        device=device,
    )

    save_all_outputs(
        model=trained_model,
        test_loader=test_loader,
        round_history=round_history,
        config=config,
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
