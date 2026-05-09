# Federated Autoencoder-Based Clinical Decision Framework with Hybrid Class Balancing

This repository contains the reproducibility implementation for the manuscript:

**Federated Autoencoder-Based Clinical Decision Framework with Hybrid Class Balancing**

The manuscript is currently under review. Publication details will be updated after acceptance.

---

## Overview

This project implements a privacy-preserving federated clinical decision framework for structured healthcare data. The framework is designed to support disease classification without requiring direct sharing of sensitive patient records across healthcare institutions.

The implemented pipeline combines:

- Federated learning for decentralized model training
- Autoencoder-based latent feature extraction
- SMOTE-based class imbalance correction
- Multi-class disease classification
- Binary clinical outcome prediction
- Focal loss for imbalanced multi-class learning
- Binary cross-entropy loss for outcome prediction
- FedAvg-based model aggregation
- Synthetic reproducibility mode for public execution
- Restricted-data mode for authorized clinical datasets

The repository is intentionally minimal and contains only the files required to reproduce the computational workflow.

---

## Repository Structure

```text
federated-autoencoder-clinical-framework/
│
├── README.md
├── requirements.txt
├── config.yaml
├── train.py
│
├── src/
│   ├── data.py
│   ├── models.py
│   ├── federated.py
│   ├── losses.py
│   ├── metrics.py
│   └── utils.py
│
├── scripts/
│   └── run_reproducibility.sh
│
└── outputs/
    └── .gitkeep
