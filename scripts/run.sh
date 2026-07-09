#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/default.yaml}"
NUM_CLIENTS="${2:-10}"
shift 2 || true

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file '$CONFIG_PATH' not found" >&2
  exit 1
fi

if [[ ! "$NUM_CLIENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: NUM_CLIENTS must be a positive integer, got '$NUM_CLIENTS'" >&2
  exit 1
fi

for override in "$@"; do
  if [[ "$override" =~ ^[A-Z_][A-Za-z0-9_]*= ]]; then
    export "${override?}"
  else
    echo "Warning: ignoring invalid override '$override' (must be KEY=value)" >&2
  fi
done

flwr run . --stream \
    --run-config "config-path='$CONFIG_PATH'" \
    --federation-config "num-supernodes=$NUM_CLIENTS"
