#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/default.yaml}"
NUM_CLIENTS="${2:-10}"
shift 2 || true

for override in "$@"; do
  if [[ "$override" =~ ^[A-Z_][A-Za-z0-9_]*= ]]; then
    export "${override?}"
  else
    echo "Warning: ignoring invalid override '$override' (must be KEY=value)" >&2
  fi
done

flwr run . --stream \
    --run-config "config-path=\"$CONFIG_PATH\"" \
    --federation-config "num-supernodes=$NUM_CLIENTS"
