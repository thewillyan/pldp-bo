#!/usr/bin/env bash
set -euo pipefail

NUM_CLIENTS="${1:-10}"
shift || true

if [[ $# -gt 0 ]]; then
  EXPERIMENTS=()
  for pattern in "$@"; do
    while IFS= read -r f; do
      EXPERIMENTS+=("$f")
    done < <(ls config/experiments/${pattern}_*.yaml 2>/dev/null || true)
  done
else
  EXPERIMENTS=(config/experiments/*.yaml)
fi

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
  echo "No experiments matched."
  exit 1
fi

for CONFIG in "${EXPERIMENTS[@]}"; do
  echo "=== Running experiment: $CONFIG ==="
  ./scripts/run.sh "$CONFIG" "$NUM_CLIENTS"
  echo ""
done

echo "All experiments completed."
