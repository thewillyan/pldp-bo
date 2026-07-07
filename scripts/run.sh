#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/default.yaml}"
NUM_CLIENTS="${2:-10}"
shift 2 || true

RUN_CONFIG_ARGS=("config-path=$CONFIG_PATH")
for override in "$@"; do
  RUN_CONFIG_ARGS+=("$override")
done

flwr run . --stream \
    --run-config "${RUN_CONFIG_ARGS[@]}" \
    --federation-config "num-supernodes=$NUM_CLIENTS"
