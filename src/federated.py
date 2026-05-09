"""
Federated learning utilities for the Federated Autoencoder-Based Clinical Decision Framework.

This module contains:
- model state extraction
- sample-weighted FedAvg aggregation
- adaptive FedAvg wrapper
- optional privacy-aware update perturbation
- client weight calculation

Raw clinical records are never exchanged in this module. Only model parameters
are handled, matching the privacy-preserving federated learning workflow.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn


StateDict = Dict[str, torch.Tensor]


def get_model_state(model: nn.Module) -> StateDict:
    """
    Return a detached CPU copy of a model state dictionary.

    Parameters
    ----------
    model:
        PyTorch model.

    Returns
    -------
    Dict[str, torch.Tensor]
        Detached model state dictionary.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def set_model_state(model: nn.Module, state_dict: StateDict) -> nn.Module:
    """
    Load a state dictionary into a model.

    Parameters
    ----------
    model:
        PyTorch model.
    state_dict:
        Model state dictionary.

    Returns
    -------
    nn.Module
        Model with loaded parameters.
    """
    model.load_state_dict(state_dict)
    return model


def compute_client_weights(client_sizes: List[int]) -> List[float]:
    """
    Compute sample-proportional client aggregation weights.

    Parameters
    ----------
    client_sizes:
        Number of samples available at each federated client.

    Returns
    -------
    List[float]
        Normalized aggregation weights.
    """
    if not client_sizes:
        raise ValueError("client_sizes cannot be empty.")

    if any(size <= 0 for size in client_sizes):
        raise ValueError("Every client must contain at least one sample.")

    total_samples = float(sum(client_sizes))

    return [float(size) / total_samples for size in client_sizes]


def fedavg(
    client_states: List[StateDict],
    client_sizes: List[int],
) -> StateDict:
    """
    Perform sample-weighted Federated Averaging.

    This implements the standard FedAvg update:

        global_weight = sum_k (n_k / n_total) * local_weight_k

    where n_k is the number of samples held by client k.

    Parameters
    ----------
    client_states:
        List of local client model state dictionaries.
    client_sizes:
        Number of samples available at each client.

    Returns
    -------
    Dict[str, torch.Tensor]
        Aggregated global model state dictionary.
    """
    validate_client_inputs(client_states, client_sizes)

    client_weights = compute_client_weights(client_sizes)
    aggregated_state: StateDict = copy.deepcopy(client_states[0])

    for key in aggregated_state.keys():
        first_tensor = client_states[0][key]

        if torch.is_floating_point(first_tensor):
            aggregated_tensor = torch.zeros_like(first_tensor, dtype=first_tensor.dtype)

            for state, weight in zip(client_states, client_weights):
                aggregated_tensor += state[key].to(first_tensor.dtype) * weight

            aggregated_state[key] = aggregated_tensor
        else:
            aggregated_state[key] = first_tensor.clone()

    return aggregated_state


def adaptive_fedavg(
    client_states: List[StateDict],
    client_sizes: List[int],
) -> StateDict:
    """
    Adaptive FedAvg wrapper.

    In this implementation, adaptation is expressed through proportional client
    weighting based on local sample size. This is suitable for heterogeneous
    clinical clients because larger local datasets contribute more strongly to
    the global model while still preserving the federated setup.

    Parameters
    ----------
    client_states:
        List of local model states.
    client_sizes:
        Number of local samples per client.

    Returns
    -------
    Dict[str, torch.Tensor]
        Aggregated global state dictionary.
    """
    return fedavg(client_states=client_states, client_sizes=client_sizes)


def add_update_noise(
    state_dict: StateDict,
    noise_std: float,
    device: torch.device | str = "cpu",
) -> StateDict:
    """
    Add small Gaussian perturbation to floating-point model parameters.

    This function provides a lightweight privacy-aware perturbation mechanism
    for model updates. It does not modify integer buffers such as batch counter
    variables.

    Parameters
    ----------
    state_dict:
        Local model state dictionary.
    noise_std:
        Standard deviation of Gaussian noise.
    device:
        Device used for noise generation.

    Returns
    -------
    Dict[str, torch.Tensor]
        Perturbed state dictionary.
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    if noise_std == 0:
        return {
            key: value.detach().cpu().clone()
            for key, value in state_dict.items()
        }

    noisy_state: StateDict = {}

    for key, value in state_dict.items():
        tensor = value.detach().clone()

        if torch.is_floating_point(tensor):
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=tensor.shape,
                device=device,
            ).to(tensor.device)

            tensor = tensor + noise.cpu()

        noisy_state[key] = tensor.cpu()

    return noisy_state


def model_delta(
    global_state: StateDict,
    local_state: StateDict,
) -> StateDict:
    """
    Compute the difference between a local client state and the global state.

    Parameters
    ----------
    global_state:
        Global model state before local training.
    local_state:
        Local model state after client training.

    Returns
    -------
    Dict[str, torch.Tensor]
        Difference between local and global model states.
    """
    validate_state_compatibility(global_state, local_state)

    delta: StateDict = {}

    for key in global_state.keys():
        if torch.is_floating_point(global_state[key]):
            delta[key] = local_state[key] - global_state[key]
        else:
            delta[key] = local_state[key].clone()

    return delta


def apply_delta(
    global_state: StateDict,
    delta_state: StateDict,
    scale: float = 1.0,
) -> StateDict:
    """
    Apply a model delta to a global state.

    Parameters
    ----------
    global_state:
        Base global model state.
    delta_state:
        Delta update state.
    scale:
        Scaling factor for the delta.

    Returns
    -------
    Dict[str, torch.Tensor]
        Updated model state.
    """
    validate_state_compatibility(global_state, delta_state)

    updated_state: StateDict = {}

    for key in global_state.keys():
        if torch.is_floating_point(global_state[key]):
            updated_state[key] = global_state[key] + scale * delta_state[key]
        else:
            updated_state[key] = global_state[key].clone()

    return updated_state


def average_deltas(
    delta_states: List[StateDict],
    client_sizes: List[int],
) -> StateDict:
    """
    Average local model deltas using sample-weighted aggregation.

    Parameters
    ----------
    delta_states:
        List of client model deltas.
    client_sizes:
        Number of samples per client.

    Returns
    -------
    Dict[str, torch.Tensor]
        Aggregated delta state.
    """
    validate_client_inputs(delta_states, client_sizes)

    client_weights = compute_client_weights(client_sizes)
    averaged_delta: StateDict = copy.deepcopy(delta_states[0])

    for key in averaged_delta.keys():
        first_tensor = delta_states[0][key]

        if torch.is_floating_point(first_tensor):
            aggregated_tensor = torch.zeros_like(first_tensor)

            for delta, weight in zip(delta_states, client_weights):
                aggregated_tensor += delta[key] * weight

            averaged_delta[key] = aggregated_tensor
        else:
            averaged_delta[key] = first_tensor.clone()

    return averaged_delta


def validate_client_inputs(
    client_states: List[StateDict],
    client_sizes: List[int],
) -> None:
    """
    Validate client states and client sample sizes.

    Parameters
    ----------
    client_states:
        List of client model states.
    client_sizes:
        Number of samples per client.
    """
    if not client_states:
        raise ValueError("client_states cannot be empty.")

    if len(client_states) != len(client_sizes):
        raise ValueError(
            "client_states and client_sizes must have the same length."
        )

    reference_keys = set(client_states[0].keys())

    for idx, state in enumerate(client_states):
        if set(state.keys()) != reference_keys:
            raise ValueError(
                f"Client state at index {idx} has incompatible parameter keys."
            )

    reference_shapes = {
        key: tensor.shape for key, tensor in client_states[0].items()
    }

    for idx, state in enumerate(client_states):
        for key, tensor in state.items():
            if tensor.shape != reference_shapes[key]:
                raise ValueError(
                    f"Client state at index {idx} has incompatible shape for key '{key}'."
                )

    if any(size <= 0 for size in client_sizes):
        raise ValueError("All client sizes must be positive.")


def validate_state_compatibility(
    state_a: StateDict,
    state_b: StateDict,
) -> None:
    """
    Validate that two model states have identical keys and tensor shapes.

    Parameters
    ----------
    state_a:
        First state dictionary.
    state_b:
        Second state dictionary.
    """
    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError("State dictionaries have different parameter keys.")

    for key in state_a.keys():
        if state_a[key].shape != state_b[key].shape:
            raise ValueError(
                f"State tensor shape mismatch for key '{key}': "
                f"{state_a[key].shape} vs {state_b[key].shape}"
            )


def summarize_client_distribution(client_sizes: List[int]) -> Dict[str, float]:
    """
    Summarize federated client sample distribution.

    Parameters
    ----------
    client_sizes:
        Number of samples per client.

    Returns
    -------
    Dict[str, float]
        Summary statistics for client distribution.
    """
    if not client_sizes:
        raise ValueError("client_sizes cannot be empty.")

    tensor = torch.tensor(client_sizes, dtype=torch.float32)

    return {
        "num_clients": int(len(client_sizes)),
        "total_samples": int(tensor.sum().item()),
        "min_client_samples": int(tensor.min().item()),
        "max_client_samples": int(tensor.max().item()),
        "mean_client_samples": float(tensor.mean().item()),
        "std_client_samples": float(tensor.std(unbiased=False).item()),
    }
