#!/usr/bin/env bash

set -e

echo "Running Federated Autoencoder-Based Clinical Decision Framework"
echo "Mode: synthetic reproducibility"

python train.py \
  --config config.yaml \
  --mode synthetic

echo "Execution completed successfully."
echo "Generated outputs are available in the outputs/ directory."
