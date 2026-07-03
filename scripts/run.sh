#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config/default.yaml}"

flwr run . --stream --run-config "config-path=$CONFIG_PATH"
