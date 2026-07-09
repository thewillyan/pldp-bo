#!/usr/bin/env bash
set -uo pipefail

CLEANUP_DONE=false
CLEANUP_PIDS=()

cleanup() {
  [[ "$CLEANUP_DONE" == "true" ]] && return
  CLEANUP_DONE=true
  local exit_code=$?
  echo ""
  echo "=== Graceful shutdown triggered ==="
  if [[ ${#CLEANUP_PIDS[@]} -gt 0 ]]; then
    echo "  Killing tracked child processes: ${CLEANUP_PIDS[*]}"
    kill "${CLEANUP_PIDS[@]}" 2>/dev/null || true
  fi
  command -v pkill &>/dev/null && pkill -P $$ 2>/dev/null || true
  echo "=== Shutdown complete (exit code: $exit_code) ==="
  exit "$exit_code"
}
trap cleanup EXIT SIGTERM SIGINT

NUM_CLIENTS="${1:-10}"
shift || true

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [num_clients] <pattern1> [pattern2 ...]"
  echo "  Globs: config/experiments/*<pattern>*.yaml"
  echo "  Examples:"
  echo "    $0 5 cifar100_noniid"
  echo "    $0 10 fedavg_cifar100 fedprox_cifar100 pldp_bo_cifar100"
  exit 1
fi

VENV_PYTHON="${VIRTUAL_ENV:-.venv}/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    VENV_PYTHON="python3"
fi
PLOT_SCRIPT="scripts/plot_results.py"

OVERALL_FAILED=()
OVERALL_SUCCESS=()

for pattern in "$@"; do
  echo ""
  echo "========================================================================"
  echo "Pattern: $pattern"
  echo "========================================================================"

  shopt -s nullglob
  configs=(config/experiments/*"${pattern}"*.yaml)
  shopt -u nullglob

  if [[ ${#configs[@]} -eq 0 ]]; then
    echo "WARNING: No config files matching pattern '$pattern'"
    continue
  fi

  pattern_dir="mlruns/$pattern"
  plots_dir="$pattern_dir/plots"
  tracking_uri="sqlite:///${pattern_dir}/mlflow.db"

  mkdir -p "$plots_dir"

  run_ids=()
  run_names_list=()

  for config in "${configs[@]}"; do
    config_name=$(basename "$config" .yaml)
    exp_plots="$plots_dir/$config_name"
    mkdir -p "$exp_plots"

    echo ""
    echo "--- Config: $config_name ---"

    run_name=$($VENV_PYTHON -c "
import yaml
with open('$config') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('logging', {}).get('run_name', '') or '')
")
    echo "  run_name: $run_name"

    if ! MLFLOW_TRACKING_URI="$tracking_uri" \
         ./scripts/run.sh "$config" "$NUM_CLIENTS"; then
      echo "  ERROR: Experiment '$run_name' (config: $config_name) failed."
      OVERALL_FAILED+=("$pattern/$config_name")
      command -v pkill &>/dev/null && pkill -P $$ 2>/dev/null || true
      continue
    fi
    OVERALL_SUCCESS+=("$pattern/$config_name")

    echo "  Fetching run_id for run_name='$run_name'..."
    run_id=$($VENV_PYTHON -m scripts.plot_results \
      --tracking-uri "$tracking_uri" \
      get-run-id --run-name "$run_name" 2>/dev/null || echo "")

    if [[ -z "$run_id" ]]; then
      echo "  WARNING: Could not find run with name '$run_name' in $tracking_uri"
      echo "  Skipping plots for this experiment."
      continue
    fi

    echo "  run_id: $run_id"
    run_ids+=("$run_id")
    run_names_list+=("$config_name")

    echo "  Generating plots..."
    $VENV_PYTHON -m scripts.plot_results \
         --tracking-uri "$tracking_uri" \
         plot "$run_id" --type all --save-dir "$exp_plots" 2>&1 | sed 's/^/    /'
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
      echo "  WARNING: Plot generation failed for run '$run_id'"
    fi
  done

  if [[ ${#run_ids[@]} -gt 1 ]]; then
    echo ""
    echo "--- Generating comparison plots for pattern: $pattern ---"

    comp_dir="$plots_dir/comparison"
    mkdir -p "$comp_dir"

    $VENV_PYTHON -m scripts.plot_results \
         --tracking-uri "$tracking_uri" \
         compare \
         --runs "${run_ids[@]}" \
         --names "${run_names_list[@]}" \
         --type all \
         --save-dir "$comp_dir" 2>&1 | sed 's/^/    /'
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
      echo "  WARNING: Comparison plot generation failed for pattern '$pattern'"
    fi
  elif [[ ${#run_ids[@]} -eq 1 ]]; then
    echo "  Only one experiment found — skipping comparison plots."
  fi

  echo ""
  echo "=== Pattern '$pattern' done. Results in $pattern_dir ==="
done

echo ""
echo "========================================"
echo "Summary:"
if [[ ${#OVERALL_FAILED[@]} -gt 0 ]]; then
  echo "  Failed (${#OVERALL_FAILED[@]}):"
  for exp in "${OVERALL_FAILED[@]}"; do
    echo "    - $exp"
  done
fi
if [[ ${#OVERALL_SUCCESS[@]} -gt 0 ]]; then
  echo "  Successful (${#OVERALL_SUCCESS[@]}):"
  for exp in "${OVERALL_SUCCESS[@]}"; do
    echo "    - $exp"
  done
fi
echo "========================================"
