# PLDP-BO Experiment Implementation Plan

**Status:** approved — ready for implementation
**Source spec:** [`EXPERIMENTS-TODO.md`](../EXPERIMENTS-TODO.md) (§2 locked constants, §3 matrix, §4 MLflow schema, §9 gap-closure worklist)
**Repo audited at:** HEAD `8b925cd` (2026-08-11)

This document translates the gap-closure worklist (`EXPERIMENTS-TODO.md` §9.1–§9.12) into a set of
feature/fix issues, ordered by dependency, so the whole team understands the problem and the fix for
each one. Every issue has: problem statement, current behavior (with file references), required
change, acceptance criteria, and files touched.

---

## 0. Background

`pldp-bo` is the experimentation repository for the PLDP-BO paper. The paper requires a specific
experiment matrix (10 methods × 10 dataset/partition cells × 12 seeds ≈ 1,200 MLflow runs) with a
strict configuration contract (§2), a strict MLflow logging schema (§4), and a verification pass
(§5.1). An audit (2026-08-14, recorded in `EXPERIMENTS-TODO.md` §9.0) found the implementation is
**not spec-conformant**:

- Locked constants: **1/13 PASS**
- §4 MLflow schema: **0/8 PASS**
- §3 matrix runner: PARTIAL
- Run data: **0 spec-conformant runs exist** (all legacy runs predate the spec)
- FEMNIST data: not yet available (pipeline must be implemented now, counts verified at extraction)

No experiment data can be trusted until the issues below are closed.

### Key facts verified during the audit

| Area | Current state (HEAD 8b925cd) |
|---|---|
| Config defaults | T=50, K=10, ρ=0.5, E=5; experiment configs deviate further (T=15–150, K=8, momentum=0.0, C=5.0, η_s=0.5, total_budget=120–6000, λ_aq=0.3, G=100, min_warmup=5) |
| per_example accounting | per-step: `num_steps = E × len(trainloader)`; σ carries an extra √num_steps factor (`src/client_app.py:488-495`, `src/client/per_example_dp_client.py:284-290`) |
| Warm-up grid | `np.linspace(rdp_min, rdp_max, warmup_rounds)` (`src/privacy/bo_scheduler.py:344`) |
| Momentum / FedProx | `ValueError` guards reject momentum≠0 and proximal_mu>0 in per_example mode (`src/client/per_example_dp_client.py:124-136`); proximal term is L2 not squared (`src/client/base_client.py:68-73`) |
| Fixed baselines | `FixedRDPScheduler` exists but is never constructed; fallback divides budget by T (`src/client_app.py:169`); attenuation gated on `strategy == "pldp_bo"` (`src/server_app.py:245`) |
| Validation | global 10% hold-out of the full train set shared by all clients (`src/data/__init__.py:28-33`); L_clean from pre-round global model — wrong semantics |
| Evaluation | `accuracy`/`server_loss` on the hold-out; no official test set, no F1 (`src/server_app.py:200-204`) |
| Partitions | only `iid` / `noniid` (α=0.5); no pathological, no min-30, no writer partition (`src/data/partitioner.py`) |
| FEMNIST | absent from registry/`_MODEL_DATA_COMPAT`/`_INPUT_CHANNELS_MAP`; data not yet downloaded (the `data/femnist-dataset-PyTorch/` repo only ships the downloader) |
| MLflow | experiment always `pldp-bo`; params under renamed keys (`data.*`, `federated.*`, …); no timing/bytes; `log_artifact` never called |
| Runner | `scripts/run` single/group only; seeds `base+idx`; no FINISHED filter; `start_run` duplicates on rerun |
| Formula fidelity | SNR uses clipped norm; PerRemaining falls back to R_t; Agreement kept as complement (minimization-equivalent) |

Models already match the paper: MLP 200-200 (`mlp200x2`), CNN conv32→conv64→FC512→dropout0.25.

---

## 1. Approved decisions (2026-08-14)

| Decision | Outcome |
|---|---|
| Archive legacy configs | All 12 active `config/experiments/*.yaml` move to `config/experiments/archive/`; the new matrix is generated fresh |
| Matrix configs | **Generated YAMLs** — `scripts/gen_matrix_configs` emits 100 files (`<dataset>_<partition>_<method>.yaml`), deterministic and reviewable; `config_version` hashed per file |
| Verification module | `scripts/verify` implements §5.1 checks 1–5 against MLflow (aggregation/report script remains user-owned per spec §5/§6) |
| Locked-config escape hatch | Top-level `assert_locked_config: bool` (default `true`); smoke configs set `false` |
| Privacy budget | Per-client flat `B_RDP = 10.0`; data-proportional/equal-division personalization disabled |

---

## 2. Issue index (dependency order)

| Issue | Title | Spec item | Kind | Phase |
|---|---|---|---|---|
| IMPL-01 | Locked-config layer | §9.1 | feature | 1 |
| IMPL-02 | Per-round RDP accounting + σ calibration | §9.2 | fix | 2 |
| IMPL-03 | Log-spaced warm-up grid | §9.3 | fix | 2 |
| IMPL-04 | Momentum DP-SGD + squared FedProx term | §9.4 | fix | 2 |
| IMPL-05 | Partition types (Dirichlet α, pathological, min-30) | §9.8 | feature | 3 |
| IMPL-06 | Per-client validation subset + clean pre-DP pass | §9.6 | fix | 3 |
| IMPL-07 | FEMNIST pipeline | §9.9 | feature | 3 |
| IMPL-08 | Official test-set evaluation + macro-F1 | §9.7 | fix | 3 |
| IMPL-09 | Fixed baselines + aggregation routing | §9.5 | feature | 4 |
| IMPL-10 | Formula-fidelity fixes (SNR, PerRemaining, Agreement) | §9.12 | fix | 3/4 |
| IMPL-11 | MLflow logging schema (§4) | §9.10 | feature | 5 |
| IMPL-12 | Matrix configs + runner + run policy | §9.11, §3 | feature | 6 |
| IMPL-13 | Verification module (§5.1) | §5.1 | feature | 6 |
| IMPL-14 | Tests, lint, smoke validation | — | chore | 7 |

Dependency graph:

```
IMPL-01 (config contract)
   ├── IMPL-02 → IMPL-03 → IMPL-04 (privacy core)
   │       └───────────────► IMPL-09 (routing consumes schedulers)
   ├── IMPL-05 → IMPL-06 → IMPL-07 (data; 07 parallel) → IMPL-08 (eval)
   │        └──► IMPL-10 (SNR needs clean pass from IMPL-06)
   └── IMPL-11 (schema consumes everything above)
           └── IMPL-12 (configs+runner) ──► IMPL-13 (verification)
                     └── IMPL-14 (validation of all)
```

---

## 3. Issues

---

### IMPL-01 — Locked-config layer

**Spec:** `EXPERIMENTS-TODO.md` §9.1 — locked constants asserted at startup.
**Kind:** feature · **Phase:** 1 · **Depends on:** —

#### Problem

The paper's numbers are only comparable if every run uses the exact §2 constants. Today config
defaults (`src/config/loader.py`) and shipped configs (`config/experiments/*`) deviate wildly
(T=15–150, K=8, ρ=0.5–0.6, E=1–4, momentum=0.0, C=5.0, η_s=0.5, λ_aq=0.3, G=100, min_warmup=5).
Per-client budgets are derived by equal division of `total_budget` or data-proportional
personalization (`src/server_app.py:53-168`). Nothing stops a run from silently using non-spec
constants, producing numbers that must not appear in the paper.

#### Required change

1. Add config fields in `src/config/loader.py`:
   - top-level `method: str` (one of the 10 §3 method names)
   - `federated.aggregation: "attenuation" | "plain"`
   - `privacy.enforce_budget: bool = True`
   - `privacy.fixed_rdp_target: float = 0.5`
   - top-level `assert_locked_config: bool = True`
2. New `src/config/locked.py` with `LOCKED_CONSTANTS` and `assert_locked_config(cfg) -> None`
   that raises `RuntimeError` on any deviation:
   - T=200, K=100, ρ=0.1, E=5, B=64, lr=0.01, momentum=0.9, clip_norm C=1.0
   - α₀=10.0 (rdp_native, fixed order), per-client flat B_RDP=10.0, [R_min,R_max]=[0.01,2.0]
   - λ_aq=0.1, G=50, kernel=matern52; `weight_decay=0.0`, `gradient_clip_norm=0.0`
   - personalization disabled (data-proportional/heterogeneity budgets forbidden)
3. Call from `src/server_app.py:main()` and `src/client_app.py:train()` right after
   `load_config`, guarded by `config.assert_locked_config`.
4. `BOConfig.bounds_strategy` locked to `"global"` (no per-client bounds) when asserting.

#### Acceptance

- Any non-spec config fails fast at startup with a message naming the offending constant(s).
- `config/smoke/*.yaml` (assertion off) load and run.
- §5.1 budget-match check runnable (utilization 0.98–1.02 for private methods).

**Files:** `src/config/loader.py`, `src/config/locked.py` (new), `src/server_app.py`, `src/client_app.py`, `tests/test_config.py`.

---

### IMPL-02 — Per-round RDP accounting + σ calibration

**Spec:** §9.2 + §2 accounting convention (1 round = 1 Gaussian release), σ_t = √(α₀·q²/(2·R_t)).
**Kind:** fix · **Phase:** 2 · **Depends on:** IMPL-01

#### Problem

In `clipping_mode="per_example"` the code accounts **per optimization step**: `_resolve_rdp`
converts the candidate to a per-step cost (`candidate/num_steps` with
`num_steps = E×len(trainloader)`) and `enforce_rdp_budget` composes `num_steps` releases
(`src/client_app.py:484-495`). The client then calls `accountant.step(num_steps=total_steps_per_round)`
(`src/client/per_example_dp_client.py:284-290`) and reports `rdp_cost = privacy_param × num_steps`
(`:334`). Result: σ carries an extra √num_steps factor and the accountant over-spends the budget,
so `B_RDP=10.0` would not be reached and the warm-up sum check (1.3995) cannot pass.

#### Required change

1. `src/client_app.py::_resolve_rdp`, per_example branch: drop the per-step conversion.
   - `num_steps = 1` for enforcement; `enforce_rdp_budget(candidate, current_rdp, budget, lower_bound, alpha, sampling_rate, clipping_mode="per_example", num_steps=1)`.
   - Direct calibration when no budget check: `sigma = _sigma_for_rdp_target_dp_sgd(candidate, alpha, sampling_rate)`.
   - `sampling_rate q = B / n_i` where `n_i` is the client's **training** size (after IMPL-06's per-client hold-out).
2. `src/client/per_example_dp_client.py`:
   - `accountant.step(sigma=sigma, clipping_norm=self._sampling_rate, num_steps=1, mode="per_example")`.
   - `rdp_cost` metric = `privacy_param` (the enforced R_t), not × steps.
   - Keep `_total_steps_per_round` only for the `num_opt_steps` diagnostic metric.
3. Add `acct_cost` (accountant-reported round cost) and `r_t_final` (enforced R_t) to the metrics
   that flow to the server (consumed by IMPL-11); they must be identical.

#### Acceptance

- `acct_cost ≡ r_t_final` (relative error ≤ 1e-6).
- §5.1 warm-up sum mean in [1.33, 1.47] over clients × runs.
- `rdp_cost` metric matches the scheduler's proposed value after enforcement.

**Files:** `src/client_app.py`, `src/client/per_example_dp_client.py`, `src/privacy/per_update_dp.py` (verify no other caller relies on per-step semantics), `tests/test_rdp_native.py`, `tests/test_privacy.py`.

---

### IMPL-03 — Log-spaced warm-up grid

**Spec:** §9.3 — 10 log-spaced points over [0.01, 0.5], ratio 50^(1/9) ≈ 1.5444.
**Kind:** fix · **Phase:** 2 · **Depends on:** IMPL-01

#### Problem

Warm-up uses `np.linspace(rdp_min, rdp_max, warmup_rounds)` (`src/privacy/bo_scheduler.py:344`)
with `min_warmup=5`, spending ≈10.05 RDP of the budget by warm-up (linear grid over [0.01, 2.0]).
The paper fixes warm-up to 10 log-spaced points summing to ≈1.40 (≈14% of B_RDP), leaving ≈8.60
for BO.

#### Required change

1. In `PLDPBORDPScheduler.__init__` (`src/privacy/bo_scheduler.py`), replace `_warmup_rdp` with a
   hardcoded module-level constant:
   `[0.01, 0.0154, 0.0239, 0.0368, 0.0569, 0.0879, 0.1357, 0.2095, 0.3236, 0.4998]` (sum ≈ 1.3995).
2. Warm-up count = 10: enforce `warmup_rounds == 10` when constructing the scheduler under the
   locked config (IMPL-01) — i.e. BOConfig.min_warmup must be 10.
3. Mirror the same grid in `PLDPBOScheduler` (epsilon path) so the two schedulers stay in sync if
   the epsilon path is ever used again.
4. Expose the grid for logging: the runner/config emits `warmup_points` (JSON) and
   `warmup_sum_nominal = 1.3995` (§4.2 params, consumed by IMPL-11).

#### Acceptance

- Scheduler warm-up sequence equals the constant grid exactly.
- §5.1 warm-up sum mean in [1.33, 1.47] (each client's first 10 participations).

**Files:** `src/privacy/bo_scheduler.py`, `tests/test_bo_scheduler.py`.

---

### IMPL-04 — Momentum DP-SGD + squared FedProx proximal term

**Spec:** §9.4 — SGD momentum 0.9; FedProx μ=0.01; record-level per-example DP (C1).
**Kind:** fix · **Phase:** 2 · **Depends on:** IMPL-02

#### Problem

`PerExampleDPClient.__init__` raises `ValueError` if `optimizer.momentum > 0` or
`federated.proximal_mu > 0` (`src/client/per_example_dp_client.py:124-136`), so the locked config
(momentum 0.9) cannot even run. The proximal term in `src/client/base_client.py:68-73` is
(μ/2)·Σ‖w−w_global‖₂ (L2 norms, not squared), which is not the standard FedProx regularizer.

#### Required change

1. Remove both guards.
2. Momentum (Opacus-style, DP-safe): maintain a per-parameter momentum buffer; at each step
   `v = momentum·v + g_avg_clipped`, then add noise to `v` and take one optimizer step.
   - The invariant **momentum applied post-clip / pre-noise** must be documented in a comment:
     applying momentum after noise would compound noise; applying it to per-sample gradients
     would require per-sample buffers.
   - Seed the buffers (deterministic per run/seed).
3. Proximal term: change to (μ/2)·Σ‖w−w_global‖² (sum of squared L2 norms) in both
   `src/client/base_client.py` and `src/client/per_example_dp_client.py` (the latter needs the
   proximal term added inside the per-batch loop before clipping).
4. Add `mu_fedprox` param plumbing (logged by IMPL-11).

#### Acceptance

- Config with momentum=0.9 + proximal_mu=0.01 runs (smoke).
- Unit test: with momentum=0.9 and no noise, weights match a reference PyTorch implementation
  applying momentum to the clipped gradient.
- Accounting unchanged (noise added after momentum; cost formula untouched).

**Files:** `src/client/per_example_dp_client.py`, `src/client/base_client.py`, `tests/test_client.py`, `tests/test_privacy.py`.

---

### IMPL-05 — Partition types: Dirichlet α, pathological, min-30

**Spec:** §9.8, §3 — Dirichlet α∈{1.0,0.5,0.1}; pathological = 2 classes/client
(MNIST 20 clients/class, CIFAR-100 2 clients/class); min 30 samples/client.
**Kind:** feature · **Phase:** 3 · **Depends on:** IMPL-01

#### Problem

`src/data/partitioner.py` only supports `iid` and `noniid` (hardcoded α=0.5 configs). No
pathological branch, no minimum-sample enforcement. Under-sampled clients (possible with small α)
break the sampling-rate q and give degenerate results.

#### Required change

1. `partition_type` values: `iid`, `dirichlet` (parameterized by `partition_alpha`; α=1.0, 0.5, 0.1),
   `pathological`, and `writer` (FEMNIST, IMPL-07).
   - Keep `noniid` as a deprecated alias of `dirichlet` with α=0.5 (configs in archive still use it).
2. `pathological`: each client receives exactly 2 non-overlapping classes;
   class → client assignment such that every class is covered by ⌈2·K/C⌉ clients
   (MNIST: 20 clients/class; CIFAR-100: 2 clients/class), deterministic given seed.
3. Min-30 enforcement (all partitions): clients with < 30 training samples are merged/dropped
   with a rule defined once (e.g. redistribute their samples to the smallest-neighbor client)
   and documented; log `partition_kwargs`.
4. `partition_single` must produce the **same** client partition as the full `partition_dataset`
   (single-client determinism — currently only guaranteed for iid/dirichlet).
5. `DataConfig` gains `partition_min_samples: int = 30` (locked by IMPL-01).

#### Acceptance

- Per-cell client counts documented; no client below 30 samples for MNIST/CIFAR-100 cells.
- `partition_kwargs` param logged (e.g. `{"type": "pathological", "classes_per_client": 2}`).
- Unit tests: pathological coverage counts (MNIST: 20/class), determinism, min-30 merge.

**Files:** `src/data/partitioner.py`, `src/config/loader.py`, `tests/test_data.py`.

---

### IMPL-06 — Per-client validation subset + clean pre-DP pass

**Spec:** §9.6, §2 — each client holds out a fixed 10% of **its own** train data (never trained on);
reference variants (Retention, Efficiency, PerRemaining, SNR, Agreement) compute L_clean by training
the local model **without DP noise** (same seed) and evaluating both models on the subset.
**Kind:** fix · **Phase:** 3 · **Depends on:** IMPL-05

#### Problem

Validation is a single global 10% hold-out of the full train set, shared by all clients
(`src/data/__init__.py:28-33`, `create_validation_loader` used by the server), and clean stats are
computed on the **pre-round global model** (`src/client/per_example_dp_client.py:217-220`). This
breaks the objective semantics: L_clean must come from the locally-trained clean model, and each
client's L_val must be its own data (never seen in training).

#### Required change

1. `src/data/__init__.py`:
   - Remove the global `random_split`; the full official train set feeds the partitioner.
   - `create_client_dataloader` returns: client train loader, client val loader (10% of the
     client's partition, drawn seeded and fixed across rounds), client train subset (post
     hold-out), client val subset, total train size.
   - Server-side `create_validation_loader` is replaced by the test loader (IMPL-08).
2. `src/client/per_example_dp_client.py`:
   - Evaluate L_val on the client's own val loader.
   - For variants retention/efficiency/perremaining/snr/agreement: after the DP pass, run a
     **clean pass** — train the local model from the global weights with the same E, B, lr,
     momentum, seed, no clipping/noise — then compute L_clean on the val subset and the clean
     update norm ‖Δ_clean‖ (consumed by IMPL-10).
   - NUN/Utility compute only from the privatized model (no clean pass).
3. `validation_frac = 0.1` param logging (IMPL-11). Sampling rate q uses the post-hold-out
   client train size (IMPL-02).

#### Acceptance

- L_clean varies per client; each client's val data is disjoint from its train data and never
  used for training.
- Accounting unaffected (clean pass consumes no privacy budget).
- Smoke run shows expected 1.5–2× local cost for C–G variants.

**Files:** `src/data/__init__.py`, `src/client/per_example_dp_client.py`, `src/client/per_update_dp_client.py`, `src/client_app.py`, `tests/test_data.py`, `tests/test_client.py`.

---

### IMPL-07 — FEMNIST pipeline

**Spec:** §9.9 — LEAF 80/20 sample split → ≈654,281 / ≈163,570 / 3,598 writers; 62 classes;
CNN 28×28; writer partition on train split; <10-sample writers merged.
**Kind:** feature · **Phase:** 3 · **Depends on:** IMPL-05

#### Problem

FEMNIST is absent: not in `DATASET_REGISTRY`, `_MODEL_DATA_COMPAT` rejects it, no
`_INPUT_CHANNELS_MAP` entry, no 62-class mapping. The data is not yet downloaded — only the
downloader repo exists at `data/femnist-dataset-PyTorch/` (the `.tar.gz` is an LFS pointer).
Per the spec, the pipeline is implemented **now** and counts verified at extraction.

#### Required change

1. Loader `FEMNISTDataset` (root/data_dir → `FEMNIST/processed/femnist_train.pt`,
   `femnist_test.pt`, `femnist_user_keys.pt`), modeled on `data/femnist-dataset-PyTorch/femnist_dataset.py`
   (each sample carries a writer id; `femnist_user_keys.pt` maps writer keys → ids).
   - Samples are 28×28 grayscale (1 channel); 62 classes.
2. Registry/compat/channels:
   - `DATASET_REGISTRY["femnist"]`, `TRANSFORMS_MAP["femnist"]` (28×28 ToTensor+normalize),
     `NUM_CLASSES_MAP["femnist"] = 62`.
   - `src/models/__init__.py::_MODEL_DATA_COMPAT["cnn"] += "femnist"`;
     `src/models/cnn.py::_INPUT_CHANNELS_MAP["femnist"] = 1` (FC input = 3136, verified by test).
   - `src/models/mlp.py` needs `_INPUT_SIZE_MAP["femnist"]` only if MLP is used with FEMNIST (it is not in the matrix — add anyway for safety or explicitly reject).
3. Partitions: `partition_type="writer"` on the **train** split — one client per writer (3,598
   writers > K=100; K clients are the largest writer clusters, or a deterministic grouping rule —
   define once and document); writers with <10 train samples merged into the nearest writer
   cluster by label distribution.
4. Official test split evaluation (with IMPL-08) and per-client test accuracy: test samples are
   grouped by writer; each client evaluates the global model on its own writers' test samples.
5. Exact counts re-read from the `.pt` files at extraction (verification-only item, reported by
   IMPL-13): train ≈654,281, test ≈163,570, writers 3,598.

#### Acceptance

- Loader raises a clear error when data is missing (no silent download attempts in CI).
- Counts check (`scripts/verify`) passes once data is extracted.
- `dataset_root` and `data_hash` (sha256 over the dataset files) params logged (IMPL-11).

**Files:** `src/data/dataloaders.py`, `src/data/partitioner.py`, `src/models/__init__.py`, `src/models/cnn.py`, `tests/test_data.py`, `tests/test_models.py`.

---

### IMPL-08 — Official test-set evaluation + macro-F1

**Spec:** §9.7, §4.3 — per-round `acc_test`/`f1_test` on the official test set; FEMNIST per-client
test accuracy artifact.
**Kind:** fix · **Phase:** 3 · **Depends on:** IMPL-06, IMPL-07 (FEMNIST part)

#### Problem

The server evaluates on the 10% train hold-out (`src/server_app.py::_run_global_evaluate`,
`accuracy`/`server_loss`). There is no test loader and no F1 anywhere. Paper numbers must come
from the official test sets (MNIST/CIFAR-100 `train=False`; FEMNIST test split).

#### Required change

1. `src/data/__init__.py`: `create_test_loader(config)` — official test splits.
2. `src/server_app.py::_run_global_evaluate`:
   - Evaluate the global model on the test loader each round.
   - Log `acc_test` (top-1) and `f1_test` (macro-F1) at `step=round` (IMPL-11 consumes).
   - Drop or rename the hold-out evaluation to `acc_val` (kept only for diagnostics; not in §4.3).
3. FEMNIST: per-client test accuracy — each client evaluates the final global model on its
   writers' test samples and returns the mean; the server writes `client_test_acc.json`
   (final step) with per-client values (IMPL-11).
   - Rationale per spec: per-client evaluation only for FEMNIST (avoid 200 rounds × per-client evals elsewhere).

#### Acceptance

- `acc_test` logged every round; MNIST sanity check in expected FedAvg range (smoke).
- Macro-F1 matches a hand-computed value on a small test set (unit test).

**Files:** `src/data/__init__.py`, `src/server_app.py`, `src/client_app.py` (FEMNIST client test eval), `tests/test_data.py`, `tests/test_server.py`.

---

### IMPL-09 — Fixed baselines + aggregation routing

**Spec:** §9.5, §2 — `dpfedavg_fixed` and `fedprox_fixed` with R = B_RDP/(ρ·T) = 0.5/round;
median attenuation for **all** private methods; plain averaging only for `nonprivate`.
**Kind:** feature · **Phase:** 4 · **Depends on:** IMPL-02, IMPL-04

#### Problem

`FixedRDPScheduler` is never constructed; the personalization fallback divides by T
(`src/client_app.py:169` → 0.05 instead of 0.5). Median attenuation is gated on
`strategy == "pldp_bo"` (`src/server_app.py:245`), so fixed baselines would get plain FedAvg —
breaking the "attenuation for all private methods" requirement.

#### Required change

1. `src/client_app.py::_make_scheduler`:
   - `method in {"dpfedavg_fixed", "fedprox_fixed"}` → `FixedRDPScheduler(0.5)`
     (from `privacy.fixed_rdp_target`, itself asserted = B_RDP/(ρ·T) by IMPL-01).
   - 7 BO variants → `PLDPBORDPScheduler` (rdp_min/rdp_max/λ_aq/G/kernel from locked config).
   - Remove the personalization-derived budget paths from the active flows.
2. `src/server_app.py`:
   - Route aggregation on the new `federated.aggregation` key, independent of strategy name:
     `attenuation` → `MedianRobustAggregation`, `plain` → `SafeFedAvg`.
   - `fedprox_fixed` uses SafeFedAvg-style aggregation **with attenuation** — it must NOT use
     `SafeFedProx` server class (the proximal term lives client-side, IMPL-04).
   - Flat per-client budget: `_compute_per_client_budgets` returns `{cid: B_RDP}` for all K
     clients (drops the QUERY round and data-proportional weights).
3. `nonprivate` → `aggregation: plain`, privacy disabled (no scheduler, no accountant).
4. Log `aggregation` param per run (IMPL-11). N/A rules (§4.4): fixed baselines log
   `r_t_candidate == r_t_final == 0.5`, `bo_time_round = 0`, no `observed_m`,
   `enforcement_count = 0`, `phase = bo` throughout.

#### Acceptance

- 4-cell strategy smoke: nonprivate, dpfedavg_fixed, fedprox_fixed, pldpbo_nun (assertion off).
- Fixed baselines hit cumulative ≈10.0 at T=200 (budget match).
- `aggregation` logged per run.

**Files:** `src/client_app.py`, `src/server_app.py`, `src/server/strategy.py`, `tests/test_server.py`, `tests/test_integration.py`.

---

### IMPL-10 — Formula-fidelity fixes (SNR, PerRemaining, Agreement)

**Spec:** §9.12 — m_snr = ‖Δ‖₂²/σ² with the **clean unclipped** update; m_rem uses R_remaining
(no R_t fallback); m_agr = logit agreement (code's `1 − cos_sim` is minimization-equivalent — keep).
**Kind:** fix · **Phase:** 3/4 · **Depends on:** IMPL-06 (clean pass), IMPL-09 (server wiring)

#### Problem

- SNR uses the **clipped** norm (`src/client/per_update_dp_client.py:164-166`); the paper uses the
  clean unclipped update norm.
- PerRemaining falls back to `privacy_param` (=R_t) when the server sends no remaining budget
  (`src/client/per_example_dp_client.py:312-317`, `src/client/per_update_dp_client.py:203-208`);
  the paper defines R_remaining = B_RDP − cum_rdp.
- Agreement: code reports `1 − cos_sim`; the paper's m_agr maximizes logit agreement. Equivalent
  under minimization — keep the code, document the mapping.

#### Required change

1. SNR (per_example + per_update): `m_snr = ‖Δ_clean‖² / σ²`, where Δ_clean is the local weight
   update from the clean pass (IMPL-06) or the raw unclipped update (per_update mode).
2. PerRemaining: remove the fallback; the server must send `remaining_rdp` per client each round
   (IMPL-09 flat-budget server wiring; computed as B_RDP − cum_rdp). If the value is missing,
   the client must raise (no silent fallback).
3. Agreement: keep `logit_disagreement = 1 − cos_sim`; document the minimization equivalence in
   `meta.display_names` (the aggregation script will present it as agreement).

#### Acceptance

- Unit tests pin the formulas against the paper's definitions (`solution.tex` §Optimization Objectives).
- No `remaining_budget` fallback path remains.

**Files:** `src/client/per_example_dp_client.py`, `src/client/per_update_dp_client.py`, `src/server/strategy.py`, `tests/test_client.py`.

---

### IMPL-11 — MLflow logging schema (§4)

**Spec:** §4 in full; §9.10.
**Kind:** feature · **Phase:** 5 · **Depends on:** IMPL-02…IMPL-10

#### Problem

`src/tracking/tracker.py` logs a single `pldp-bo` experiment, params under renamed keys
(`data.*`, `federated.*`, …), per-round metrics = val accuracy/loss + legacy stats
(`epsilon_mean`, `rdp_cost_mean`, …), no timing/bytes instrumentation, and `log_artifact` is never
called. The aggregation script (and the paper) cannot consume any of it.

#### Required change

1. Experiments/runs/tags:
   - Experiment `<dataset>_<partition>` (e.g. `mnist_dirichlet_0.5`, `cifar100_pathological`, `femnist_natural`).
   - Run `<method>_seed<NN>`.
   - Tags: `dataset`, `partition`, `method`, `seed`, `config_version` (hash of §2 constants),
     `code_git_hash` (from `git rev-parse HEAD`), `group`.
2. Params under §4.2 names (all of them, incl. JSON params): `T`, `K`, `rho`, `E`, `B`,
   `eta_server`, `local_opt` (=`sgd_momentum0.9`), `clip_norm`, `alpha0`, `B_RDP`, `R_min`,
   `R_max`, `warmup_points`, `warmup_sum_nominal`, `lambda_aq`, `kernel`, `G`, `N` (=seed),
   `mu_fedprox`, `model` (`mlp200x2`|`cnn`), `dataset_sizes`, `partition_kwargs`, `seeds`,
   `validation_frac`, `aggregation`, `enforce_budget`, `dataset_root`, `data_hash`.
   - `data_hash` = sha256 over the dataset files; `dataset_sizes` = JSON train/test/writers.
3. Per-round metrics at `step=round` (server-side; step 0 = before training):
   `acc_test`, `f1_test`, `n_participants`, `mean_r_t` (mean enforced R_t over selected clients),
   `mean_cum_rdp`, `budget_utilization` (final: mean cum_rdp/B_RDP), `bytes_round`
   (server sends+receives, summed array bytes), `bo_time_round` (`perf_counter` around GP fit +
   acquisition + selection, per-client mean), `acct_time_round` (around budget check + σ
   calibration, per-client mean), `bo_overhead_pct` (final: cumulative BO time / run wall time).
4. Artifact `client_state.json` (final step): server accumulates per-client per-participation
   arrays from client replies — `r_t_candidate`, `r_t_final`, `cum_rdp`, `remaining_rdp`, `phase`
   (`warmup|bo|exhausted`), `warmup_rounds` (global rounds of warm-up participations),
   `dropout_round` (first refused global round, `null` if survives to T), `observed_m`,
   `acct_cost`, `enforcement_count`, plus variant components (`L_clean`, `L_noisy`,
   `update_norm_noisy`, `update_norm_clean`, `sigma`, `agreement`).
   §4.4 N/A rules: `nonprivate` logs no privacy fields; fixed baselines log the §4.4 constants.
5. Artifact `client_test_acc.json` (FEMNIST, final step).
6. Client reports the new fields per round (r_t_candidate/r_t_final/acct_cost/phase/observed_m/
   enforcement_count + components) via metrics; the strategy accumulates them.

#### Acceptance

- A single smoke run logs every §4 key with the right names/values.
- IMPL-13 verification checks runnable against that run.

**Files:** `src/tracking/tracker.py`, `src/server/strategy.py`, `src/server_app.py`, `src/client_app.py`, `src/client/*.py`, `tests/test_tracker.py`, `tests/test_server.py`.

---

### IMPL-12 — Matrix configs + runner + run policy

**Spec:** §9.11, §3 — 10 methods × 10 cells × seeds 0–11; FINISHED-only; same-seed rerun overwrites.
**Kind:** feature · **Phase:** 6 · **Depends on:** IMPL-01, IMPL-11

#### Problem

`scripts/run` only does `single`/`group` with seeds `base+idx` and configs seeded at 42; it has no
FINISHED filter and `start_run` duplicates runs on rerun. There are no §3 matrix configs.

#### Required change

1. `scripts/gen_matrix_configs`: emits 100 YAMLs into `config/experiments/`
   (`<dataset>_<partition>_<method>.yaml`) from (a) the 10 locked-constant templates and (b) a
   method table: `nonprivate`, `dpfedavg_fixed`, `fedprox_fixed`, `pldpbo_nun`, `pldpbo_utility`,
   `pldpbo_retention`, `pldpbo_efficiency`, `pldpbo_perremaining`, `pldpbo_snr`, `pldpbo_agreement`
   (mapped to existing `optimization_metric` keys: nun, utility, utility_retention,
   utility_efficiency, utility_per_remaining, snr, logit_disagreement).
   - Cells: MNIST {iid, dirichlet_1.0, dirichlet_0.5, dirichlet_0.1, pathological} (600 runs),
     CIFAR-100 {iid, dirichlet_0.5, dirichlet_0.1, pathological} (480), FEMNIST {natural} (120).
   - Each file self-contained, `assert_locked_config: true`, and a stable `config_version` hash.
2. `scripts/run matrix`:
   - Iterate cells × methods × seeds 0–11.
   - Inventory first: query MLflow (experiment = `<dataset>_<partition>`) — include a run only if
     status FINISHED **and** `config_version` tag matches the current config.
   - Rerun policy: `mlflow.delete_run(run_id)` on the existing (experiment, run_name) then start a
     new run with the same name and seed (so names stay unique and reruns overwrite).
   - `--dry-run` prints the §6.1-style inventory (done/failed/missing per cell) without running.
   - Optional `--gpu-parallel N` runs N cells concurrently; log which GPU each cell uses.
3. Remove/replace the old `group`/`replot` paths for non-archive configs (archive configs remain
   usable via `single` for legacy exploration).
4. Archive step: move the 12 current configs to `config/experiments/archive/`.

#### Acceptance

- `--dry-run` over an empty tracking DB shows 1,200 cells with seeds 0–11 missing.
- Re-running the same cell after a crash shows the crashed run replaced (same run name, new id).
- §6.1 inventory achievable: n=12 per cell or explicit failures.

**Files:** `scripts/gen_matrix_configs` (new), `scripts/run`, `config/experiments/*` (generated), `tests/test_scripts.py`.

---

### IMPL-13 — Verification module (§5.1)

**Spec:** §5.1 — warm-up sum, budget match, drop-out, FEMNIST counts, accounting parity.
**Kind:** feature · **Phase:** 6 · **Depends on:** IMPL-11, IMPL-12

#### Problem

§5.1 checks are the acceptance criteria for IMPL-02/IMPL-03/IMPL-09/IMPL-11 but nothing runs them.
The full aggregation/report script (§5/§6) is user-owned; a lightweight verifier closes the loop
for development.

#### Required change

New `scripts/verify` (reads MLflow URI + filters from argv/config):
1. **Warm-up sum:** per private method × run × client, Σ over the client's first 10
   participations of `acct_cost`; mean ± SD vs nominal 1.3995; pass if mean within ±5% (1.33–1.47).
   Per-round parity: `acct_cost` vs `r_t_final` (median and max relative error).
2. **Budget matching:** mean final cumulative RDP ≈ 10.0 for all private methods
   (utilization 1.00 ± 0.02); nonprivate = 0.
3. **Drop-out:** per method: distribution of `dropout_round` (never = T+1), fraction never
   dropping out, mean ± SD drop-out round, mean final cumulative RDP.
4. **FEMNIST counts:** read the `.pt` files; report exact train/test/writer counts
   vs ≈654,281 / ≈163,570 / 3,598.
5. **Accounting convention:** covered by check 1 (per-round cost == one Gaussian release).
6. Prints a PASS/FAIL list (exit code 1 on any FAIL) — consumption-ready for CI on smoke runs.

#### Acceptance

- Runs against a smoke-run MLflow DB and produces the §5.1 checklist with numbers.
- Same output shape as §6.1 `verification` block (so the aggregation script can reuse it).

**Files:** `scripts/verify` (new), `tests/test_scripts.py`.

---

### IMPL-14 — Tests, lint, smoke validation

**Spec:** §9.4/§9.5 acceptance smoke tests; repo conventions (`pyproject.toml`: ruff, mypy strict).
**Kind:** chore · **Phase:** 7 · **Depends on:** IMPL-01…IMPL-13

#### Problem

≈4,400 lines of tests assert legacy behavior (linspace warm-up, momentum guard, per-step
accounting, global val split, budget division) and will break under IMPL-01…IMPL-11.

#### Required change

1. Update broken tests; delete tests for removed behavior (e.g. momentum `ValueError`).
2. New tests per issue (already listed in each issue's acceptance):
   - locked-config assertion (fail + pass cases),
   - warm-up grid constants, per-round accounting parity,
   - momentum DP-SGD math, squared proximal term,
   - pathological partition coverage, min-30 merge, per-client val hold-out, clean-pass semantics,
   - macro-F1, §4 param/metric/artifact presence in a tracked run (in-process MLflow test DB),
   - matrix runner dry-run inventory + rerun overwrite.
3. Smoke configs: `config/smoke/*.yaml` (`assert_locked_config: false`, tiny T/K) including the
   4-cell strategy smoke (nonprivate, dpfedavg_fixed, fedprox_fixed, pldpbo_nun) and a FEMNIST
   data-less loader check.
4. `ruff check`, `mypy`, `pytest` green.

#### Acceptance

- CI-style: `ruff`, `mypy`, `pytest` all green on the working tree.
- Smoke run produces a run whose §4 schema passes IMPL-13.

**Files:** `tests/*`, `config/smoke/*` (new).

---

## 4. Definition of done (repo side)

Everything below is the repo-side subset of `EXPERIMENTS-TODO.md` §8:

1. IMPL-01…IMPL-11 merged; `assert_locked_config` active for all matrix configs.
2. `scripts/gen_matrix_configs` produces the 100-cell matrix; `scripts/run matrix --dry-run`
   shows the full §3 inventory (1,200 runs).
3. `scripts/verify` PASSes on a smoke run (warm-up 1.33–1.47, utilization 0.98–1.02, parity ≤1e-6).
4. FEMNIST pipeline in place; counts verified at extraction (data blocked on user download).
5. Legacy configs archived; `tests/*` green; lint/typecheck clean.
6. The full §3 matrix is then runnable on GPU; the aggregation/report script (user-owned) consumes
   MLflow per §5/§6.

## 5. Out of scope

| Item | Owner / reason |
|---|---|
| Aggregation script, `report.md`/`report.json`, figures, curves | user-owned (spec §5/§6); IMPL-13 only implements §5.1 checks |
| Running the full 1,200-run matrix | user (GPU hours); runner support is delivered |
| Paper writing (results/discussion/conclusion) | assistant, only after §8 done criteria met |
| λ_aq sweep, multi-budget sweep, FedAWA/FedStrag, ε/δ reporting, per-step accounting, client-level LDP | explicitly out of scope (spec §1.1) |
| FEMNIST data download | user decision; pipeline + verification implemented now |

## 6. Status log

- 2026-08-14: document created from the approved implementation plan (user decisions: generated
  matrix YAMLs, verification module included, config-flag assertion escape hatch).
- 2026-08-14: **IMPL-01 closed.** Added `method`/`assert_locked_config` (top level),
  `federated.aggregation`, `privacy.enforce_budget`, `privacy.fixed_rdp_target`; new
  `src/config/locked.py` (`LOCKED_CONSTANTS`, method contract, `collect_violations`,
  `assert_locked_config`, `config_version`); wired into `server_app.main` and `client_app.train`;
  fixed `load_config` to merge top-level scalar YAML keys (previously `seed` was silently ignored).
  66 new/updated tests; ruff/mypy baseline unchanged (no new errors); 415 tests green.
- 2026-08-14: **IMPL-02 closed.** Per-round accounting: `_resolve_rdp` per_example branch now
  calibrates σ_t = √(α₀·q²/(2·R_t)) against the per-round candidate (no per-step conversion;
  `total_steps_per_round` param dropped from `_resolve_rdp`/`_resolve_epsilon`); mirrored in the
  epsilon path (user-approved); `per_example_dp_client` steps the accountant once per round
  (`num_steps=1`), reports `rdp_cost = R_t`, and adds `r_t_final`/`acct_cost` metrics (identical
  by construction, rel ≤ 1e-6, verified at fit level). 14 new tests; 424 tests green; ruff/mypy
  baseline unchanged. `acct_cost`/`r_t_final` server-side logging deferred to IMPL-11.
