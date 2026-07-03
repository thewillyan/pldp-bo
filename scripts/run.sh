#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/default.yaml}"
NUM_CLIENTS="${2:-10}"

flwr run . --stream \
    --run-config "config-path=\"$CONFIG_PATH\"" \
    --federation-config "num-supernodes=$NUM_CLIENTS"
