"""
Utility functions for the Federated Autoencoder-Based Clinical Decision Framework.

This module contains:
- random seed control
- YAML configuration loading
- device selection
- output directory creation
- JSON and CSV export helpers
- lightweight model and experiment utilities
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible execution.

    Parameters
    ----------
    seed:
        Random seed value.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    validate_config(config)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required top-level configuration sections.

    Parameters
    ----------
    config:
        Configuration dictionary.
    """
    required_sections = [
        "project",
        "execution",
        "data",
        "preprocessing",
        "federated",
        "model",
        "training",
        "loss",
        "privacy",
        "outputs",
    ]

    missing_sections = [
        section for section in required_sections if section not in config
    ]

    if missing_sections:
        raise ValueError(
            f"Configuration file is missing required sections: {missing_sections}"
        )

    if config["federated"]["num_clients"] < 2:
        raise ValueError("federated.num_clients must be at least 2.")

    if config["federated"]["federated_rounds"] < 1:
        raise ValueError("federated.federated_rounds must be at least 1.")

    if config["federated"]["local_epochs"] < 1:
        raise ValueError("federated.local_epochs must be at least 1.")

    if config["training"]["batch_size"] < 1:
        raise ValueError("training.batch_size must be at least 1.")

    if config["training"]["learning_rate"] <= 0:
        raise ValueError("training.learning_rate must be greater than 0.")

    if config["model"]["latent_dim"] < 1:
        raise ValueError("model.latent_dim must be at least 1.")


def get_device(device_setting: str = "auto") -> torch.device:
    """
    Select execution device.

    Parameters
    ----------
    device_setting:
        Device setting. Supported values are "auto", "cpu", and "cuda".

    Returns
    -------
    torch.device
        Selected PyTorch device.
    """
    normalized = str(device_setting).lower().strip()

    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            print("CUDA was requested but is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be one of: auto, cpu, cuda")


def make_output_dir(output_dir: str | Path) -> Path:
    """
    Create the output directory if it does not exist.

    Parameters
    ----------
    output_dir:
        Output directory path.

    Returns
    -------
    Path
        Created output directory path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def save_json(data: Dict[str, Any], output_path: str | Path) -> None:
    """
    Save a dictionary as a formatted JSON file.

    Parameters
    ----------
    data:
        Dictionary to save.
    output_path:
        Target JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_data = make_json_serializable(data)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_data, file, indent=4)


def save_table(table: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a pandas DataFrame as a CSV file.

    Parameters
    ----------
    table:
        DataFrame to save.
    output_path:
        Target CSV file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(output_path, index=False)


def make_json_serializable(value: Any) -> Any:
    """
    Convert common NumPy, pandas, and PyTorch objects into JSON-safe values.

    Parameters
    ----------
    value:
        Object to convert.

    Returns
    -------
    Any
        JSON-serializable object.
    """
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if isinstance(value, pd.Series):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    return value


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.

    Parameters
    ----------
    model:
        PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def save_experiment_config(
    config: Dict[str, Any],
    output_dir: str | Path,
    filename: str = "used_config.json",
) -> None:
    """
    Save the active experiment configuration to the output directory.

    Parameters
    ----------
    config:
        Active configuration dictionary.
    output_dir:
        Output directory.
    filename:
        Output filename.
    """
    output_path = Path(output_dir) / filename
    save_json(config, output_path)


def ensure_clean_output_dir(output_dir: str | Path) -> Path:
    """
    Create the output directory without deleting existing files.

    This function is intentionally conservative to prevent accidental removal
    of reviewer-generated outputs.

    Parameters
    ----------
    output_dir:
        Output directory path.

    Returns
    -------
    Path
        Output directory path.
    """
    return make_output_dir(output_dir)


def get_project_root() -> Path:
    """
    Return the current working directory as the project root.

    Returns
    -------
    Path
        Current working directory.
    """
    return Path(os.getcwd()).resolve()
