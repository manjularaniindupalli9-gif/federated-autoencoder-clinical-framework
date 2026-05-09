"""
Data handling utilities for the Federated Autoencoder-Based Clinical Decision Framework.

This module supports:
- synthetic structured clinical data generation for public reproducibility
- restricted clinical CSV loading for authorized users
- preprocessing of numerical and categorical clinical features
- SMOTE-based class balancing
- non-IID federated client partitioning
- PyTorch dataset and dataloader construction

The original clinical dataset is not distributed with this repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


class ClinicalDataset(Dataset):
    """
    PyTorch dataset for structured clinical records.

    Each item contains:
    - features
    - multi-class disease label
    - binary outcome label
    """

    def __init__(
        self,
        x: np.ndarray,
        y_multiclass: np.ndarray,
        y_binary: np.ndarray,
    ) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_multiclass = torch.tensor(y_multiclass, dtype=torch.long)
        self.y_binary = torch.tensor(y_binary, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.x[index],
            "disease_label": self.y_multiclass[index],
            "outcome_label": self.y_binary[index],
        }


def generate_synthetic_clinical_data(
    n_samples: int,
    n_classes: int,
    multiclass_col: str,
    binary_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a structured clinical-like dataset for reproducibility.

    The synthetic data are not real patient records. They are designed only to
    reproduce the computational workflow of the manuscript.

    Parameters
    ----------
    n_samples:
        Number of synthetic records.
    n_classes:
        Number of disease classes.
    multiclass_col:
        Name of the multi-class disease label column.
    binary_col:
        Name of the binary outcome label column.
    seed:
        Random seed.

    Returns
    -------
    pd.DataFrame
        Synthetic structured clinical dataset.
    """
    rng = np.random.default_rng(seed)

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")

    base_probs = np.linspace(1.0, 0.35, n_classes)
    class_probs = base_probs / base_probs.sum()

    disease_labels = rng.choice(
        np.arange(n_classes),
        size=n_samples,
        replace=True,
        p=class_probs,
    )

    age = rng.normal(44 + disease_labels * 3.5, 12, n_samples).clip(18, 90)
    systolic_bp = rng.normal(118 + disease_labels * 4.0, 13, n_samples).clip(85, 210)
    diastolic_bp = rng.normal(76 + disease_labels * 2.0, 9, n_samples).clip(50, 130)
    heart_rate = rng.normal(78 + disease_labels * 1.5, 10, n_samples).clip(45, 145)
    cholesterol = rng.normal(178 + disease_labels * 8.0, 30, n_samples).clip(90, 340)
    glucose = rng.normal(96 + disease_labels * 7.0, 22, n_samples).clip(55, 280)
    bmi = rng.normal(25 + disease_labels * 0.9, 4.5, n_samples).clip(15, 45)
    respiratory_rate = rng.normal(17 + disease_labels * 0.8, 3, n_samples).clip(10, 35)
    oxygen_saturation = rng.normal(97 - disease_labels * 0.7, 2.2, n_samples).clip(75, 100)
    temperature = rng.normal(36.8 + disease_labels * 0.12, 0.7, n_samples).clip(34, 41)
    hemoglobin = rng.normal(13.7 - disease_labels * 0.12, 1.4, n_samples).clip(7, 18)
    wbc_count = rng.normal(7200 + disease_labels * 380, 1600, n_samples).clip(2500, 18000)
    platelet_count = rng.normal(250000 - disease_labels * 3500, 52000, n_samples).clip(
        80000, 520000
    )
    creatinine = rng.normal(0.9 + disease_labels * 0.06, 0.25, n_samples).clip(0.35, 3.5)
    crp = rng.gamma(shape=2.0 + disease_labels * 0.25, scale=2.0, size=n_samples).clip(
        0.1, 45
    )

    sex = rng.choice(["female", "male"], size=n_samples, p=[0.48, 0.52])
    smoking_status = rng.choice(
        ["never", "former", "current"],
        size=n_samples,
        p=[0.57, 0.25, 0.18],
    )

    hypertension_history = (
        rng.random(n_samples)
        < sigmoid_numpy(-4.0 + 0.035 * age + 0.012 * systolic_bp)
    ).astype(int)

    diabetes_history = (
        rng.random(n_samples) < sigmoid_numpy(-5.0 + 0.032 * age + 0.018 * glucose)
    ).astype(int)

    medication_adherence = rng.choice(
        ["low", "moderate", "high"],
        size=n_samples,
        p=[0.22, 0.48, 0.30],
    )

    risk_score = (
        -8.0
        + 0.035 * age
        + 0.018 * systolic_bp
        + 0.014 * glucose
        + 0.030 * bmi
        + 0.0012 * crp
        + 0.22 * hypertension_history
        + 0.25 * diabetes_history
        + 0.18 * disease_labels
        - 0.030 * oxygen_saturation
    )

    outcome_probability = sigmoid_numpy(risk_score)
    binary_outcome = (rng.random(n_samples) < outcome_probability).astype(int)

    disease_names = [f"disease_class_{idx}" for idx in range(n_classes)]
    disease_text_labels = np.array([disease_names[idx] for idx in disease_labels])

    data = pd.DataFrame(
        {
            "age": age,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "cholesterol": cholesterol,
            "glucose": glucose,
            "bmi": bmi,
            "respiratory_rate": respiratory_rate,
            "oxygen_saturation": oxygen_saturation,
            "temperature": temperature,
            "hemoglobin": hemoglobin,
            "wbc_count": wbc_count,
            "platelet_count": platelet_count,
            "creatinine": creatinine,
            "crp": crp,
            "sex": sex,
            "smoking_status": smoking_status,
            "hypertension_history": hypertension_history,
            "diabetes_history": diabetes_history,
            "medication_adherence": medication_adherence,
            multiclass_col: disease_text_labels,
            binary_col: binary_outcome,
        }
    )

    return data


def sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function for NumPy arrays."""
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def load_restricted_dataset(
    data_path: str,
    multiclass_col: str,
    binary_col: str,
) -> pd.DataFrame:
    """
    Load an authorized restricted clinical CSV dataset from a local secure path.

    The dataset must not be committed to GitHub.

    Parameters
    ----------
    data_path:
        Local path to the restricted CSV file.
    multiclass_col:
        Name of the disease label column.
    binary_col:
        Name of the binary outcome label column.

    Returns
    -------
    pd.DataFrame
        Loaded clinical dataset.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Restricted dataset not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Restricted dataset must be provided as a CSV file.")

    dataset = pd.read_csv(path)

    required_columns = {multiclass_col, binary_col}
    missing_columns = required_columns.difference(dataset.columns)

    if missing_columns:
        raise ValueError(
            f"Restricted dataset is missing required target columns: {missing_columns}"
        )

    if len(dataset) == 0:
        raise ValueError("Restricted dataset is empty.")

    return dataset


def preprocess_clinical_data(
    dataset: pd.DataFrame,
    multiclass_col: str,
    binary_col: str,
    scaling: str = "standard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Clean, encode, scale, and transform structured clinical data.

    Parameters
    ----------
    dataset:
        Raw clinical dataset.
    multiclass_col:
        Disease label column.
    binary_col:
        Binary outcome label column.
    scaling:
        Scaling strategy: standard, minmax, or none.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]
        Processed features, multi-class labels, binary labels, and feature names.
    """
    if multiclass_col not in dataset.columns:
        raise ValueError(f"Missing multi-class target column: {multiclass_col}")

    if binary_col not in dataset.columns:
        raise ValueError(f"Missing binary target column: {binary_col}")

    dataset = dataset.copy()

    y_multiclass_raw = dataset[multiclass_col]
    y_binary_raw = dataset[binary_col]

    features = dataset.drop(columns=[multiclass_col, binary_col])

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        col for col in features.columns if col not in numeric_columns
    ]

    processed_arrays: List[np.ndarray] = []
    feature_names: List[str] = []

    if numeric_columns:
        numeric_data = features[numeric_columns].replace([np.inf, -np.inf], np.nan)

        numeric_imputer = SimpleImputer(strategy="median")
        numeric_array = numeric_imputer.fit_transform(numeric_data)

        if scaling == "standard":
            scaler = StandardScaler()
            numeric_array = scaler.fit_transform(numeric_array)
        elif scaling == "minmax":
            scaler = MinMaxScaler()
            numeric_array = scaler.fit_transform(numeric_array)
        elif scaling == "none":
            pass
        else:
            raise ValueError("scaling must be one of: standard, minmax, none")

        processed_arrays.append(numeric_array.astype(np.float32))
        feature_names.extend(numeric_columns)

    if categorical_columns:
        categorical_data = features[categorical_columns].astype("object").fillna("missing")

        encoder = build_one_hot_encoder()
        categorical_array = encoder.fit_transform(categorical_data)

        processed_arrays.append(categorical_array.astype(np.float32))

        encoded_names = encoder.get_feature_names_out(categorical_columns).tolist()
        feature_names.extend(encoded_names)

    if not processed_arrays:
        raise ValueError("No usable feature columns found in the dataset.")

    x = np.concatenate(processed_arrays, axis=1).astype(np.float32)

    y_multiclass = LabelEncoder().fit_transform(y_multiclass_raw.astype(str)).astype(
        np.int64
    )

    y_binary = encode_binary_labels(y_binary_raw)

    return x, y_multiclass, y_binary, feature_names


def build_one_hot_encoder() -> OneHotEncoder:
    """
    Build a OneHotEncoder compatible with recent scikit-learn versions.

    Returns
    -------
    OneHotEncoder
        Configured encoder.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def encode_binary_labels(labels: pd.Series) -> np.ndarray:
    """
    Encode binary labels into 0 and 1.

    Parameters
    ----------
    labels:
        Raw binary labels.

    Returns
    -------
    np.ndarray
        Binary labels as int64.
    """
    clean_labels = labels.copy()

    if clean_labels.isna().any():
        raise ValueError("Binary outcome labels contain missing values.")

    unique_values = pd.Series(clean_labels.unique())

    if len(unique_values) != 2:
        raise ValueError(
            "Binary outcome column must contain exactly two unique classes."
        )

    if pd.api.types.is_numeric_dtype(clean_labels):
        sorted_values = sorted(clean_labels.unique())
        mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
        return clean_labels.map(mapping).to_numpy(dtype=np.int64)

    return LabelEncoder().fit_transform(clean_labels.astype(str)).astype(np.int64)


def apply_smote_balancing(
    x: np.ndarray,
    y_multiclass: np.ndarray,
    y_binary: np.ndarray,
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance the multi-class disease labels.

    The binary outcome labels for synthetic samples are assigned from the nearest
    original sample to preserve consistency between disease labels and binary outcomes.

    Parameters
    ----------
    x:
        Feature matrix.
    y_multiclass:
        Multi-class disease labels.
    y_binary:
        Binary outcome labels.
    k_neighbors:
        Number of SMOTE neighbors.
    seed:
        Random seed.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Balanced feature matrix, balanced disease labels, and aligned binary labels.
    """
    class_counts = np.bincount(y_multiclass)
    min_class_count = int(class_counts[class_counts > 0].min())

    if min_class_count < 2:
        raise ValueError(
            "SMOTE requires at least two samples in every disease class."
        )

    adjusted_k = min(k_neighbors, min_class_count - 1)

    smote = SMOTE(
        sampling_strategy="not majority",
        k_neighbors=adjusted_k,
        random_state=seed,
    )

    x_balanced, y_multiclass_balanced = smote.fit_resample(x, y_multiclass)

    if len(x_balanced) == len(x):
        return x_balanced.astype(np.float32), y_multiclass_balanced.astype(np.int64), y_binary

    y_binary_balanced = align_binary_labels_after_smote(
        x_original=x,
        y_binary_original=y_binary,
        x_balanced=x_balanced,
    )

    return (
        x_balanced.astype(np.float32),
        y_multiclass_balanced.astype(np.int64),
        y_binary_balanced.astype(np.int64),
    )


def align_binary_labels_after_smote(
    x_original: np.ndarray,
    y_binary_original: np.ndarray,
    x_balanced: np.ndarray,
) -> np.ndarray:
    """
    Assign binary labels to SMOTE-generated samples using nearest original samples.

    SMOTE generates new samples for the multi-class target only. This helper aligns
    the secondary binary label by assigning each balanced sample the binary label of
    the closest original sample in feature space.

    Parameters
    ----------
    x_original:
        Original feature matrix.
    y_binary_original:
        Original binary outcome labels.
    x_balanced:
        Feature matrix after SMOTE.

    Returns
    -------
    np.ndarray
        Binary labels aligned with the balanced feature matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(x_original)

    _, indices = nearest.kneighbors(x_balanced)
    aligned_binary = y_binary_original[indices.flatten()]

    return aligned_binary.astype(np.int64)


def train_test_validation_split(
    x: np.ndarray,
    y_multiclass: np.ndarray,
    y_binary: np.ndarray,
    test_size: float,
    validation_size: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Split data into training, validation, and test subsets.

    Stratification is performed using the multi-class disease labels.

    Parameters
    ----------
    x:
        Feature matrix.
    y_multiclass:
        Multi-class disease labels.
    y_binary:
        Binary outcome labels.
    test_size:
        Test fraction.
    validation_size:
        Validation fraction from the remaining training data.
    seed:
        Random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Split arrays.
    """
    (
        x_train_valid,
        x_test,
        y_train_valid_mc,
        y_test_mc,
        y_train_valid_bin,
        y_test_bin,
    ) = train_test_split(
        x,
        y_multiclass,
        y_binary,
        test_size=test_size,
        random_state=seed,
        stratify=y_multiclass,
    )

    validation_fraction = validation_size / max(1.0 - test_size, 1e-8)

    (
        x_train,
        x_valid,
        y_train_mc,
        y_valid_mc,
        y_train_bin,
        y_valid_bin,
    ) = train_test_split(
        x_train_valid,
        y_train_valid_mc,
        y_train_valid_bin,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train_valid_mc,
    )

    return {
        "x_train": x_train.astype(np.float32),
        "x_valid": x_valid.astype(np.float32),
        "x_test": x_test.astype(np.float32),
        "y_train_multiclass": y_train_mc.astype(np.int64),
        "y_valid_multiclass": y_valid_mc.astype(np.int64),
        "y_test_multiclass": y_test_mc.astype(np.int64),
        "y_train_binary": y_train_bin.astype(np.int64),
        "y_valid_binary": y_valid_bin.astype(np.int64),
        "y_test_binary": y_test_bin.astype(np.int64),
    }


def split_non_iid_clients(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_client_samples: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Split training samples into non-IID federated clients using Dirichlet allocation.

    Parameters
    ----------
    labels:
        Multi-class labels for training samples.
    num_clients:
        Number of federated clients.
    alpha:
        Dirichlet concentration parameter. Lower values increase non-IID behavior.
    min_client_samples:
        Minimum acceptable samples per client.
    seed:
        Random seed.

    Returns
    -------
    List[np.ndarray]
        List of sample indices for each client.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    unique_classes = np.unique(labels)

    for attempt in range(100):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for cls in unique_classes:
            class_indices = np.where(labels == cls)[0]
            rng.shuffle(class_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        client_indices_array = [
            np.array(sorted(indices), dtype=np.int64) for indices in client_indices
        ]

        client_sizes = [len(indices) for indices in client_indices_array]

        if min(client_sizes) >= min_client_samples:
            return client_indices_array

    return balanced_fallback_client_split(
        n_samples=len(labels),
        num_clients=num_clients,
        seed=seed,
    )


def balanced_fallback_client_split(
    n_samples: int,
    num_clients: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Fallback balanced client split if Dirichlet partitioning produces very small clients.

    Parameters
    ----------
    n_samples:
        Number of training samples.
    num_clients:
        Number of clients.
    seed:
        Random seed.

    Returns
    -------
    List[np.ndarray]
        Balanced client sample indices.
    """
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)

    return [
        np.array(split, dtype=np.int64)
        for split in np.array_split(all_indices, num_clients)
    ]


def create_client_loaders(
    split_data: Dict[str, np.ndarray],
    client_indices: List[np.ndarray],
    batch_size: int,
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    Create PyTorch dataloaders for federated clients, validation, and testing.

    Parameters
    ----------
    split_data:
        Dictionary returned by train_test_validation_split.
    client_indices:
        List of client-specific training indices.
    batch_size:
        Batch size.

    Returns
    -------
    Tuple[List[DataLoader], DataLoader, DataLoader]
        Client loaders, validation loader, and test loader.
    """
    client_loaders: List[DataLoader] = []

    x_train = split_data["x_train"]
    y_train_mc = split_data["y_train_multiclass"]
    y_train_bin = split_data["y_train_binary"]

    for indices in client_indices:
        dataset = ClinicalDataset(
            x=x_train[indices],
            y_multiclass=y_train_mc[indices],
            y_binary=y_train_bin[indices],
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        client_loaders.append(loader)

    validation_dataset = ClinicalDataset(
        x=split_data["x_valid"],
        y_multiclass=split_data["y_valid_multiclass"],
        y_binary=split_data["y_valid_binary"],
    )

    test_dataset = ClinicalDataset(
        x=split_data["x_test"],
        y_multiclass=split_data["y_test_multiclass"],
        y_binary=split_data["y_test_binary"],
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return client_loaders, validation_loader, test_loader
