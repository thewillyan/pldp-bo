# PLDP-BO: Per-Update Local Differential Privacy with Bayesian Optimization

Federated Learning framework comparing FedAvg, FedProx, and PLDP-BO strategies under per-update local differential privacy.

**Key features:**

- **Per-update DP mechanism** — clips and noises the model delta once per round (no per-gradient DP-SGD, no opacus dependency)
- **RDP accountant** — native per-round Rényi differential privacy accounting with serialization for Flower state persistence
- **Epsilon schedulers** — `FixedEpsilonScheduler`, `UniformRandomEpsilonScheduler`, and `PLDPBOScheduler` (warm-up + Gaussian Process + Expected Improvement)
- **Budget enforcement** — binary search (`enforce_epsilon_budget()`) that respects per-round budget constraints
- **Median robust aggregation** — `MedianRobustAggregation(FedAvg)` with median-weight attenuation + server learning rate (PLDP-BO only)
- **Two optimization metrics** — NUN (normalized update norm) and utility loss, both computed per-round per-client regardless of config
- **MLflow tracking** — per-client metrics (`epsilon`, `update_norm`, `utility_loss`, `cumulative_epsilon`) logged with cross-client stats per round

---

## Installation

Requires Python 3.14+.

```bash
# Clone + enter
git clone <repo-url> pldp-bo
cd pldp-bo

# Create virtual environment and install
uv sync

# Activate
source .venv/bin/activate
```

---

## Quick Start

### 1. Run the smoke test

```bash
./scripts/run single config/experiments/smoke_test.yaml 5
```

This runs 2 rounds with 5 MNIST clients (no privacy) to verify everything is wired correctly.

### 2. Browse results

```bash
mlflow ui
```

Open `http://localhost:5000` in your browser.

### 3. Run a real experiment

```bash
# PLDP-BO with NUN metric, MNIST IID, 10 clients, 50 rounds
./scripts/run single config/experiments/pldp_bo_mnist_iid_nun.yaml 10

# FedAvg + DP baseline
./scripts/run single config/experiments/fedavg_mnist_iid.yaml 10

# Run all experiments in a group (e.g. cifar100_iid)
./scripts/run group cifar100_iid 10

# List available groups
./scripts/run list
```

---

## Experiment Configs

All configs inherit from `config/default.yaml` and override specific fields.

### PLDP-BO (8 configs)

| Config | Strategy | Dataset | Partition | Metric |
|---|---|---|---|---|
| `pldp_bo_mnist_iid_nun.yaml` | pldp_bo | MNIST | IID | NUN |
| `pldp_bo_mnist_iid_utility.yaml` | pldp_bo | MNIST | IID | utility |
| `pldp_bo_mnist_noniid_nun.yaml` | pldp_bo | MNIST | Non-IID | NUN |
| `pldp_bo_mnist_noniid_utility.yaml` | pldp_bo | MNIST | Non-IID | utility |
| `pldp_bo_cifar100_iid_nun.yaml` | pldp_bo | CIFAR-100 | IID | NUN |
| `pldp_bo_cifar100_iid_utility.yaml` | pldp_bo | CIFAR-100 | IID | utility |
| `pldp_bo_cifar100_noniid_nun.yaml` | pldp_bo | CIFAR-100 | Non-IID | NUN |
| `pldp_bo_cifar100_noniid_utility.yaml` | pldp_bo | CIFAR-100 | Non-IID | utility |

### Baselines (8 configs)

| Config | Strategy | Dataset | Partition |
|---|---|---|---|
| `fedavg_mnist_iid.yaml` | fedavg | MNIST | IID |
| `fedavg_mnist_noniid.yaml` | fedavg | MNIST | Non-IID |
| `fedavg_cifar100_iid.yaml` | fedavg | CIFAR-100 | IID |
| `fedavg_cifar100_noniid.yaml` | fedavg | CIFAR-100 | Non-IID |
| `fedprox_mnist_iid.yaml` | fedprox | MNIST | IID |
| `fedprox_mnist_noniid.yaml` | fedprox | MNIST | Non-IID |
| `fedprox_cifar100_iid.yaml` | fedprox | CIFAR-100 | IID |
| `fedprox_cifar100_noniid.yaml` | fedprox | CIFAR-100 | Non-IID |

### Personalization (3 configs)

`personalized_custom.yaml`, `personalized_data_proportional.yaml`, `personalized_heterogeneity.yaml`

### Special

`smoke_test.yaml` (2 rounds, 5 clients, no privacy), `baseline.yaml` (minimal FedAvg), `dp_example.yaml`

---

## Configuration Reference

All defaults are in `config/default.yaml`. Configs are deep-merged at runtime (experiment overrides default).

```yaml
data:
  name: cifar10           # Dataset: mnist, cifar10, cifar100
  num_clients: 10
  partition_type: iid     # iid or noniid
  batch_size: 64
  val_split: 0.1

model:
  name: cnn               # cnn, mlp, resnet18
  num_classes: 10

federated:
  num_rounds: 50
  fraction_fit: 0.5
  fraction_evaluate: 0.2
  local_epochs: 5
  strategy: fedavg        # fedavg, fedprox, pldp_bo
  server_learning_rate: 1.0  # Used only when strategy == pldp_bo

optimizer:
  name: sgd
  lr: 0.01
  momentum: 0.9

privacy:
  enabled: false
  mechanism: gaussian
  max_grad_norm: 1.0
  delta: 1e-5

bo:
  enabled: false
  warmup_rounds: 20
  epsilon_min: 0.1
  epsilon_max: 10.0
  epsilon_budget: 8.0
  optimization_metric: nun  # nun or utility
  grid_points: 100
  acquisition_penalty: 0.1
  gp_kernel: matern52       # matern52, rbf, matern32
  observation_noise: 0.01

personalization:
  enabled: false
  strategy: uniform         # uniform, data_proportional, heterogeneity
  epsilon_min: 0.1
  epsilon_max: 10.0

logging:
  tracker: mlflow
  experiment_name: pldp-bo
  tracking_uri: sqlite:///./mlflow.db

seed: 42
```

---

## Results & Analysis

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

Each run logs server loss/accuracy per round, and for PLDP-BO runs additionally logs per-client epsilon, update_norm, utility_loss, cumulative_epsilon, and cross-client statistics.

### Plotting Functions

All functions return a `matplotlib.figure.Figure`; pass `save_path` to save to disk.

```python
from src.plotting import (
    plot_convergence,
    plot_privacy_budget,
    plot_epsilon_schedules,
    plot_metric_vs_epsilon,
    plot_comparison_convergence,
    plot_comparison_privacy,
)

# Single-run convergence (loss + accuracy)
fig = plot_convergence("mlflow_run_id")
fig.savefig("convergence.png")

# Privacy budget spend over rounds
fig = plot_privacy_budget("mlflow_run_id")

# Per-client epsilon schedules with mean ± σ band
fig = plot_epsilon_schedules("mlflow_run_id")
fig = plot_epsilon_schedules("mlflow_run_id", client_ids=[0, 1, 2], show_mean_std=False)

# Epsilon vs metric scatter (warm-up vs BO phases)
fig = plot_metric_vs_epsilon("mlflow_run_id", client_id=3, metric="utility_loss", warmup_rounds=20)

# Cross-experiment comparison
fig = plot_comparison_convergence(["run_id_1", "run_id_2"], labels=["FedAvg", "PLDP-BO"])
fig = plot_comparison_privacy(["run_id_1", "run_id_2"], labels=["FedAvg", "PLDP-BO"])
```

---

## Project Structure

```
pldp-bo/
  config/
    default.yaml                     # Default configuration
    experiments/                     # Experiment-specific configs (22 files)
  docs/
    PLDP-BO.md                       # Algorithm specification
  plots/                             # Output directory for generated plots
  scripts/
    run                              # Experiment runner: ./scripts/run single|group|list
    plot                             # Plot tool: ./scripts/plot [--group <name>] plot|compare|list-runs
  src/
    client/
      __init__.py                    # Client factory (FlowerClient or PerUpdateDPClient)
      per_update_dp_client.py        # PerUpdateDPClient — trains, clips, noises, returns metrics
      flower_client.py               # Standard FlowerClient (no privacy)
    client_app.py                    # Flower ClientApp — scheduler/accountant lifecycle, budget enforcement
    config/
      loader.py                      # YAML loader, dataclasses (FederatedConfig, BOConfig, etc.)
    data/
      dataset.py                     # Dataset loading (MNIST, CIFAR-10/100)
      partition.py                   # IID/Non-IID partitioning
    logging/
      tracker.py                     # MLflow tracking
    models/
      cnn.py, mlp.py, resnet.py      # Model definitions
    plotting/
      __init__.py                    # Exports all 6 plotting functions
      _helpers.py                    # MLflow extraction helpers
      bo.py                          # plot_epsilon_schedules, plot_metric_vs_epsilon
      comparison.py                  # plot_comparison_convergence, plot_comparison_privacy
      convergence.py                 # plot_convergence
      privacy.py                     # plot_privacy_budget
    privacy/
      accountant.py                  # RDPAccountant (no opacus)
      bo_scheduler.py                # PLDPBOScheduler (warm-up + GP + EI)
      epsilon_scheduler.py           # EpsilonScheduler ABC, Fixed, UniformRandom
      metrics.py                     # compute_nun(), compute_utility_loss()
      per_update_dp.py               # PerUpdateGaussianMechanism, clip, noise, calibrate, enforce_budget
    server/
      strategy.py                    # MedianRobustAggregation(FedAvg) with client metric logging
    server_app.py                    # Flower ServerApp — strategy dispatch
    utils.py                         # set_seed, misc
  tests/                             # 120 tests across 10 test files
```

---

## Testing

```bash
# Full suite
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Specific test file
python -m pytest tests/test_bo_scheduler.py -v
```

---

## Algorithm Documentation

See [`docs/PLDP-BO.md`](docs/PLDP-BO.md) for the detailed algorithm specification, including:

- FL workflow with per-update DP
- RDP accounting and budget enforcement
- Epsilon scheduling (fixed, uniform random, BO)
- Median-based robust aggregation
- NUN and utility loss metrics
