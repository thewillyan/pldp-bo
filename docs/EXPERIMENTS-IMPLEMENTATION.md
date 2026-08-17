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

**Spec:** §9.9 — LEAF 80/20 sample split → ≈654,281 / ≈163,570 / 3,597 writers; 62 classes;
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
3. Partitions: `partition_type="writer"` on the **train** split — one client per writer (3,597
   writers > K=100; K clients are the largest writer clusters, or a deterministic grouping rule —
   define once and document); writers with <10 train samples merged into the nearest writer
   cluster by label distribution.
4. Official test split evaluation (with IMPL-08) and per-client test accuracy: test samples are
   grouped by writer; each client evaluates the global model on its own writers' test samples.
5. Exact counts re-read from the `.pt` files at extraction (verification-only item, reported by
   IMPL-13): train ≈654,281, test ≈163,570, writers 3,597.

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
   vs ≈654,281 / ≈163,570 / 3,597.
5. **Accounting convention:** covered by check 1 (per-round cost == one Gaussian release).
6. Prints a PASS/FAIL list (exit code 1 on any FAIL) — consumption-ready for CI on smoke runs.

#### Acceptance

- Runs against a smoke-run MLflow DB and produces the §5.1 checklist with numbers.
- Same output shape as §6.1 `verification` block (so the aggregation script can reuse it).

**Files:** `scripts/verify` (new), `tests/test_scripts.py`.

##### Status log

- **2026-08-15 — IMPL-13 closed.** `scripts/verify` implemented, TDD (21 new tests;
  599 → 620 green). `VerifyRun` (uri, experiment, run name, run id, seed, tags, params
  incl. `param_float`, client-state dicts via `_load_client_state` — `list_artifacts` +
  `download_artifacts`, expects `client_state.json`). Discovery: FINISHED runs whose
  `config_version` tag matches the locked hash (mirrors IMPL-12 inventory), filterable by
  `--dataset`/`--partition`/`--method`/`--seeds`. Check 1 warm-up: per client, Σ `acct_cost`
  over the first 10 participations (or available), mean±SD vs nominal 1.3995 within ±5%
  (1.33–1.47); parity `|acct_cost−r_t_final|/r_t_final` median+max, zero-valued `r_t_final`
  (refused rounds) excluded. Check 2 budget: mean final `cum_rdp` per method, utilization
  1.00±0.02 vs B_RDP 10.0; nonprivate reports 0.0 with a note (§4.4 N/A rule — no privacy
  fields logged, `PASS`). Check 3 drop-out: fraction never (dropout_round None → T+1),
  mean±SD round, mean final RDP; descriptive `pass=True` when data present (per approved
  plan). Check 4 FEMNIST: exact counts vs 654,281/163,570/3,597; SKIP (no exit effect)
  when root/`dataset_root` absent or read fails; non-FEMNIST datasets reported as a note.
  Enforcement (mean `enforcement_count` per client, fraction of reduced rounds) reported
  in the §6.1 `verification` JSON block, not pass/fail. CLI: `--tracking-uri` (default
  `MLFLOW_TRACKING_URI` → `sqlite:///./mlflow.db`), filters, `--json` (emits only the
  `verification` object, shape per §6.1). Text output: per method×check PASS/FAIL/SKIP
  lines with numbers, `=== totals === runs / failed_checks`, exit 1 on any FAIL. `_aggregate`
  folds per-run checks per method (nonprivate budget and FEMNIST matches aggregate through;
  artifact-less private runs stay SKIP). Manual acceptance: synthetic smoke DB (test-built
  §4-schema sqlite: healthy grid-warm-up cell, over-budget cell, nonprivate, wrong-version
  runs) produces the checklist with numbers — 6 PASS / 2 FAIL / 4 SKIP (FEMNIST absent by
  design), exit 1, `failed_checks: 2`; `--json` verified; filters and empty-DB (exit 0)
  paths checked. Ruff zero new findings; mypy zero new errors on `scripts/verify`
  (pre-existing baselines unchanged).

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
- 2026-08-14: **IMPL-03 closed.** Fixed 10-point log-spaced warm-up grid
  `WARMUP_GRID` (spec §9.3, sum 1.3995) + `WARMUP_SUM_NOMINAL` exported from
  `src/privacy/bo_scheduler.py` (consumed by IMPL-11's `warmup_points`/`warmup_sum_nominal`
  params); both scheduler paths (RDP + epsilon mirror) use the grid, capped at 10
  (`min(warmup_rounds, len(grid))`); grid is absolute RDP over [0.01, 0.5], independent of the
  BO search bounds. Fixes beyond the plan: (a) `normalize_ei` flatness floor (1e-4 relative)
  — GP float noise (~1e-6) was amplified to full scale, defeating the documented degenerate
  case so flat-utility BO proposed R_max instead of R_min; (b) test alignment to the grid
  domain (`test_integration.py` lifecycle bounds, `test_bo_scheduler.py` warm-up expectations,
  `test_rdp_native.py`). 5 new grid-integrity tests; 429 tests green; ruff/mypy baseline
  unchanged.
- 2026-08-14: **IMPL-04 closed.** Momentum DP-SGD + squared FedProx (spec §9.4). Removed both
  guards in `PerExampleDPClient.__init__`; manual per-parameter momentum buffer applied
  post-clip / pre-noise (Opacus-style, DP-safe — each example appears once with weight m^(t-i) ≤ 1,
  sensitivity stays 2C/n; invariant documented in a comment); optimizer momentum forced 0.0 via
  new `_get_optimizer(..., momentum=None)` kwarg to avoid double momentum; FedProx term changed
  to (μ/2)·Σ‖w−w_global‖² (sum of squared L2 norms) in `base_client.py`, added before clipping
  per-example in `per_example_dp_client.py` (deterministic public shift, same for all examples),
  and mirrored in `per_update_dp_client.py` (user-approved). `mu_fedprox` param plumbing deferred
  to IMPL-11 (value already at `federated.proximal_mu`, locked per method). Acceptance verified:
  momentum=0.9 + proximal_mu=0.01 runs (rdp_native smoke); reference-match tests prove
  momentum-equals-torch-SGD and shift-before-clip ordering (noise disabled via monkeypatch); σ
  and RDP cost identical with/without momentum (accounting unchanged). 8 new tests, 1 removed
  (test_momentum_rejected); 436 tests green; ruff/mypy baseline unchanged (only the 3 pre-existing
  line-shifted errors).
- 2026-08-14: **IMPL-05 closed.** Partition types (spec §3). `src/data/partitioner.py` rewritten:
  index-plan based (`_plan_partition` + per-client materialization), so `partition_single` and the
  full `partition_dataset` are structurally identical — single-client parity is guaranteed, not
  tested. `noniid` kept as a deprecated alias of `dirichlet` with α=0.5 (archive YAMLs use it);
  `writer` recognized but raises `NotImplementedError` ("requires the FEMNIST pipeline, IMPL-07").
  Pathological: greedy least-covered class-pair assignment (tie: lowest class id), each class
  covered ⌈2K/C⌉ times (MNIST cell: 20 clients × 2 classes, 600/client; CIFAR-100 cell: 2,
  500/client), balanced seeded chunks. Min-30: `_enforce_min_samples` tops up deficient clients
  (smallest-first, tie: lowest id) from the largest donor's trailing slice (donors never drop
  below 30; K preserved); degenerate exhaustion impossible for paper cells. `partition_dataset`
  gained `seed=` (previously global RNG — determinism bug); `build_partition_kwargs` helper added
  for IMPL-11 logging; `DataConfig.partition_min_samples: int = 30`. 15 new tests (alias/parity/
  pathological coverage/empty-fill/min-30 non-vacuous checks: sizes 1→30, donors 201→85); 451
  tests green; ruff/mypy zero new errors (mypy 64→52: rewrite dropped bare generics). Note:
  `src/data/` is gitignored/untracked — the partitioner rewrite is not in git history.
- 2026-08-14: **IMPL-06 closed.** Per-client validation subset + clean pre-DP pass (spec §9.6).
  `src/data/__init__.py` rewritten: `create_dataset` now returns the full official train set (no
  `random_split`), `_cached_dataset(name, data_dir)`; `create_client_dataloader` returns a
  5-tuple `(train_loader, val_loader, train_subset, val_subset, total_train_size)`;
  `create_validation_loader` deleted (client-side hold-out instead of server-side global split).
  New `partitioner.split_holdout(subset, val_frac, seed)`: dedicated `np.random.RandomState`,
  floor(10%) of the client partition, val/train Subsets of the same underlying dataset; seed =
  `seed + partition_id` (fixed across rounds); `total_train_size` stays global (data_proportional
  weighting); min-30 applies pre-holdout. `server_app.py`: global-model validation evaluation
  (`_run_global_evaluate` + `evaluate_fn`) removed — `server_loss`/`accuracy` absent until IMPL-08
  official test-set eval. `client_app.py` call sites unpack the 5-tuple; `client_subset` is now the
  post-holdout train_subset. `per_example_dp_client.py`: clean pass only for
  retention/efficiency/perremaining/snr/agreement (`CLEAN_PASS_METHODS`); clean pass = fresh
  training of a deepcopy of the pristine global net taken before the DP pass (same E/B/lr/
  momentum/seed, no clip/noise/accountant); NUN/Utility/nonprivate/`""` skip → clean-derived
  metrics reported as 0.0; new additive metric `update_norm_clean` (per_update: `= delta_norm`).
  Verified: L_clean varies per client, accounting (σ, cumulative_epsilon) unaffected by the clean
  pass, clean optimizer steps exactly double DP steps. 21 new tests (14 data, 7 client); 472 tests
  green; ruff/mypy zero new errors vs git HEAD (mypy 196 total; only removals: 3 old server_app
  errors). `src/data/` still untracked/gitignored.
- 2026-08-14: **IMPL-07 closed.** FEMNIST pipeline (spec §9.9; data unavailable — `femnist.tar.gz`
  is an LFS pointer — pipeline implemented now, counts verified at extraction via
  `femnist_counts`, wired to §5.1 at IMPL-13). New `src/data/femnist.py`: `FEMNISTDataset` reads
  `FEMNIST/processed/femnist_{train,test}.pt` (`[data, targets, users]`) + `femnist_user_keys.pt`;
  missing files raise a clear `FileNotFoundError` (never downloads; `download=` accepted only for
  torchvision-compatible signature); stored 0–255 floats are rescaled to [0, 1] (the reference
  repo feeds F-mode floats through MNIST normalize unscaled — loader normalizes robustly,
  both scales tested); samples returned as (1, 28, 28) tensors + int label; `.users`/`.user_keys`
  exposed for writer partition and per-client test grouping (IMPL-08). Registry: `femnist` in
  `DATASET_REGISTRY`/`TRANSFORMS_MAP` (Normalize 0.1307/0.3081 only)/`NUM_CLASSES_MAP=62`.
  Models: CNN gains `_INPUT_DIMS_MAP` (femnist 28×28 — without it the FC input would be 4096,
  not 3136; verified by test); `_MODEL_DATA_COMPAT["cnn"] += femnist`; mlp `_INPUT_SIZE_MAP`
  28×28 for safety (mlp still rejected for femnist). Partitioner: `partition_type="writer"`
  (user-approved rule): clients = the K largest writers by train-sample count; writers with <10
  samples merged into the nearest client by minimum JS divergence on label histograms (initial
  client histograms; ties lowest client id; merges smallest-first); other writers dropped;
  seed-independent pure function of (users, targets, K) so `partition_single` ≡ full parity;
  min-30 skipped for writer (FEMNIST exempt); `build_partition_kwargs` logs
   `{type: writer, merge_threshold: 10}`. 23 new tests (loader 8, writer 9, registry 1, models 5);
   495 tests green; ruff/mypy zero new errors (femnist.py clean; partitioner back to its 3
   pre-existing len errors). `src/data/` still untracked/gitignored.
- 2026-08-14: **IMPL-08 closed.** Official test-set evaluation + macro-F1 (spec §3; §9.7).
   `src/data/__init__.py`: `create_dataset(config, train=True)` gains the train flag;
   `_cached_dataset` key extended to `(name, data_dir, train)` (maxsize=2 still exactly fits
   train+test); new `create_test_dataset(config)` (official test split, `train=False`) and
   `create_test_loader(config)` (no shuffle). `server_app.py`: `_macro_f1` (sklearn-style
   macro-F1, zero denominators → 0, zero-support classes excluded) + `_run_global_test_evaluate`
   wired via Flower 1.33's official `evaluate_fn` hook of `Strategy.start()` — called after every
   round's aggregation with the updated model; logs `acc_test` (top-1) + `f1_test` (macro-F1) at
   `step=round` for rounds 1..T (round 0 skipped so the untrained model never pollutes the
   curves); returns the MetricRecord. FEMNIST per-client test accuracy (§9.7): after `start()`
   returns, the server sends the final global model via QUERY task `client_test_accuracy`;
   each client (guarded to FEMNIST) evaluates only the test samples of its own writers (writer
   set from its train partition's `.users`), returns `partition_id`/`test_accuracy`/`n_test`;
   server writes a deterministic `client_test_acc.json` (sorted by partition_id) and
   `tracker.log_artifact`s it (temp file, unlinked after; mean/sd/n_clients stay in the
   aggregation script per §6.1). `strategy.py`: `MetricLoggingMixin` now inherits `FedAvg`
   (MRO already matched runtime) and overrides `aggregate_evaluate` to log the aggregated
   client hold-out metrics as `val_loss_mean`/`acc_val_mean` — diagnostics only, not in §4.3
   (the rename allowed by the spec; drop on request). Acceptance verified: acc_test/f1_test
   logged per round; macro-F1 hand-computed unit tests (3-class mixed batch, never-correct
   class, single class, empty); server eval end-to-end on a fixed-weight Linear model
   (acc=0.75, F1=(0.8+2/3)/2); deterministic JSON artifact test; client writer-set filtering
   test (n=4 of 6 test samples, acc=0.75). 16 new tests (3 data, 10 server, 3 client); 511 tests
   green; ruff/mypy zero new errors vs git HEAD (ruff 54 pre-existing findings unchanged;
   mypy 261 = 261; only line-shifted pre-existing notes). `src/data/` still untracked/gitignored.
- 2026-08-14: **IMPL-10 closed.** Formula-fidelity fixes (spec §9.12).
   SNR (both clients): `m_snr = ‖Δ_clean‖²/σ²` — per_update now uses the raw unclipped
   `delta_norm` (previously `min(delta_norm, clip)`); per_example uses the clean-pass
   `update_norm_clean` (previously a gradient-scale `mean_after/(σ·C)²`), 0.0 for non-clean-pass
   methods (N/A rule, consistent with other clean-derived metrics). PerRemaining: no fallback
   path remains — the server tracks per-client `cumulative_rdp` (eps fallback) from fit replies
   and injects `remaining_rdp = max(0, B_RDP − cum)` into every fit config alongside
   `per_client_budget` (`_add_budgets_to_messages` + `MetricLoggingMixin._remaining_rdp_map`;
   round 1 = full budget; node/partition keying matches the budget map); `client_app`
   reads it via `_read_remaining_rdp(msg)` and forwards through `create_client`; both DP
   clients raise `ValueError` when it is missing (per_example only in the clean-pass branch
   where m_rem is computed; per_update always — budget-exhausted early return unaffected).
   The client's own `remaining_budget` stays solely for the BO scheduler. Agreement: kept
   `1 − cos_sim`, documented the minimization equivalence in comments at both computation
   sites (the report schema's `meta.display_names` presents it as agreement, §6.1).
   Acceptance verified: SNR formula pinned by identity tests (per_example clean-pass and
   per_update with `update_clip_norm=1e-6` proving the raw norm is used), 0.0 without clean
   pass, raise-on-missing in both clients (and no raise for non-clean per_example), server
   round-1/round-2 injection, node-keyed partition resolution, `_read_remaining_rdp`
   read logic, create_client forwarding. 17 new tests; 528 tests green; ruff/mypy zero new
   errors vs git HEAD (ruff 55 = 55; mypy 255 lines, only line-shifted pre-existing notes).
   `src/data/` still untracked/gitignored.

- 2026-08-14: **IMPL-11 closed.** MLflow logging schema (spec §4; §9.10).
   Tracker (`src/tracking/tracker.py`): experiments/runs derived as `<dataset>_<partition>` /
   `<method>_seed<NN>` (`partition_label`: iid / `dirichlet_<alpha>` / `noniid`→`dirichlet_0.5` /
   `pathological` / `writer`→`natural`); tags dataset, partition, method, seed, `config_version`
   (locked hash), `code_git_hash` (git HEAD, `unknown` fallback), group; §4.2 params under spec
   names (T, K, rho, E, B, eta_server, local_opt=`sgd_momentum<M>`, clip_norm, alpha0, B_RDP,
   enforce_budget, R_min, R_max, warmup_points, warmup_sum_nominal, lambda_aq, kernel, G, N=seed,
   mu_fedprox, model, dataset_sizes, partition_kwargs, seeds, validation_frac, aggregation,
   dataset_root, data_hash); `data_hash` = sha256 over the sorted dataset files (relpath +
   `\x00` + content + `\x00` per file; param absent when the dataset dir is missing — user
   decision); FEMNIST sizes via `femnist_counts` (real data now extracted, 5.6G). Legacy logging
   kept alongside (user decision). Clients (`client_app.py`): `_resolve_rdp`/`_resolve_epsilon`
   return 5-tuples adding `r_t_candidate` + `bo_time` (perf_counter around GP fit + acquisition +
   selection) + `acct_time` (around budget check + σ calibration); reply metrics add
   phase/bo_time/acct_time/r_t_candidate/observed_m (`_OPTIMIZATION_METRIC_KEY_MAP`; only when BO
   enabled and not exhausted); phase sent as a numeric code (MetricRecord rejects strings):
   0=warmup, 1=bo, 2=exhausted — the server decodes. per_update client adds `r_t_final` +
   `acct_cost` (=`compute_rdp_cost`, mirroring per_example). Strategy (`src/server/strategy.py`,
   all 3 classes): §4.3 per-round metrics at `step=round` — n_participants, mean_r_t,
   mean_cum_rdp, bo_time_round, acct_time_round, bytes_round (configure_train materializes
   messages and counts sent array bytes; received counted in `_log_client_metrics`); §4.4
   per-client per-participation accumulation in `MetricLoggingMixin` from ALL replies
   (`_record_exhausted` scans raw replies because `_filter_valid_replies` drops budget-exhausted
   ones) — r_t_candidate, r_t_final (0.0 for refused rounds), cum_rdp, remaining_rdp (the value
   the server sent that round), phase (decoded), observed_m, acct_cost, components (L_clean,
   L_noisy, update_norm_noisy, update_norm_clean, sigma, agreement), warmup_rounds,
   dropout_round (first refused round, null default), enforcement_count (candidate≠final on
   non-exhausted participations; enforced rounds still report candidate+phase so the script can
   tell enforcement from refusal); accessors `get_client_state`/`get_bo_time_total`/
   `get_acct_time_total`. Nonprivate skips privacy fields automatically (no phase marker) and the
   artifact; fixed baselines accumulate naturally to the §4.4 constants (candidate=final=0.5,
   phase `bo`, enforcement 0). Server (`server_app.py`): `perf_counter` wall time around
   `strategy.start()`; `_write_client_state_artifact` writes a deterministic `client_state.json`
   (sorted by client id) via tempfile + `tracker.log_artifact` and logs the final metrics
   budget_utilization (mean final cum_rdp / B_RDP) and bo_overhead_pct (cumulative BO time / wall
   time) at `step=T`. Note: enforcement_count is derived server-side from candidate≠final rather
   than client-reported (equivalent, robust to restarts); the doc's parenthetical list is not
   binding on the source. 50 new/updated tests (tracker 28, client 6, strategy 12, artifact 4);
   569 tests green; ruff/mypy zero new errors vs git HEAD (ruff 53 = 53; mypy 174 = 174; only
   line-shifted pre-existing notes). `src/data/` still untracked/gitignored.
- 2026-08-15: **IMPL-09 closed.** Fixed baselines + aggregation routing (spec §9.5; §2 contract).
  Client (`client_app.py`): `_make_rdp_native_scheduler` routes `method in FIXED_METHODS`
  (`dpfedavg_fixed`, `fedprox_fixed`) → `FixedRDPScheduler(rdp_target=privacy.fixed_rdp_target)`
  (asserted = B_RDP/(ρ·T) = 0.5 by IMPL-01) before the personalization guard; BO methods →
  `PLDPBORDPScheduler`; `bo_time` measured only when `getattr(scheduler, "_phase", None)` is not
  None, so fixed baselines log `bo_time_round = 0` (§4.4 N/A rule); removed the personalization
  divide-by-T fallback in `_resolve_rdp`/`_resolve_epsilon` (the scheduler-None path now raises
  ValueError). Server (`server_app.py`): new `_make_strategy(config, tracker, budgets,
  node_to_partition)` helper routes on `federated.aggregation` — `attenuation` →
  `MedianRobustAggregation` (all private methods incl. fixed baselines), `plain` → `SafeFedAvg`
  (nonprivate); `fedprox_fixed` no longer uses the `SafeFedProx` server class (proximal term is
  client-side, IMPL-04; class kept in strategy.py for legacy); `_compute_per_client_budgets` is
  now flat — `{nid: total_budget}` for all K clients — dropping the QUERY round and
  data-proportional weights; `_discover_node_to_partition` removed (dead). Tests: new
  `TestSchedulerRouting` (5), `TestStrategyRouting` (3), `TestFourCellMatrixSmoke` (4-cell:
  nonprivate→SafeFedAvg/None scheduler, dpfedavg_fixed+fedprox_fixed→Median/FixedRDPScheduler,
  pldpbo_nun→Median/PLDPBORDPScheduler, assertion off), `TestFixedBaselineBudgetMatch` (20
  participations × 0.5 = 10.0 = B_RDP; 21st refused); `test_bo_time_zero_without_scheduler`
  rewritten as `test_bo_time_zero_for_fixed_scheduler`; budget tests updated to flat semantics
  (custom/data_proportional/QUERY-branch tests deleted per user decision). 575 tests green;
  ruff/mypy zero new errors vs git HEAD (ruff 38 = 38; mypy zero new, net −9 lines from deleted
  legacy tests). Client `personalization_metadata` query handler kept for legacy (personalization
  module remains), now unreachable from the server.
- 2026-08-15: **IMPL-12 closed.** Matrix configs + runner + run policy (spec §9.11, §3).
  `scripts/gen_matrix_configs`: emits 100 deterministic self-contained YAMLs
  (`config/experiments/<dataset>_<partition>_<method>.yaml`) from a shared `CELLS` table —
  MNIST {iid, dirichlet_1.0, dirichlet_0.5, dirichlet_0.1, pathological} ×5, CIFAR-100
  {iid, dirichlet_0.5, dirichlet_0.1, pathological} ×4, FEMNIST {natural} ×1 — × 10 methods
  (`nonprivate`, `dpfedavg_fixed`, `fedprox_fixed`, `pldpbo_nun`, `pldpbo_utility`,
  `pldpbo_retention`, `pldpbo_efficiency`, `pldpbo_perremaining`, `pldpbo_snr`,
  `pldpbo_agreement`), each locked (§2 constants), `assert_locked_config: true`, `seed: 0`,
  `aggregation: attenuation` (nonprivate: plain), and a stable `config_version` hash embedded
  as YAML metadata (computed from the global §2 constants, matching the tracker tag; ignored
  by `load_config`). `scripts/run matrix`: inventory (per experiment `<dataset>_<partition>`,
  run `<method>_seed<NN>`, done = FINISHED **and** matching `config_version` tag, failed =
  exists but not done, missing = absent) printed per cell×method with a totals line;
  `--dry-run`/`--dataset`/`--partition`/`--method`/`--seeds` (default `0-11`)/`--tracking-uri`/
  `--num-clients`. `run_matrix` (rerun policy): for each pending run, `mlflow.delete_run` the
  stale run id, then relaunch via `cmd_single` with `seed=<N>` (client count from the config
  unless overridden) with `MLFLOW_TRACKING_URI` exported; post-launch id resolution is
  experiment-scoped (`_resolve_run_id(..., experiment=)`, `cmd_single` gained the kwarg) since
  run names repeat across cells — the legacy all-experiments lookup returned None and
  misreported launches as failed; final status printed per run. `--gpu-parallel` deferred
  (user decision). Legacy `group`/`replot group` paths removed (cmd_group, cmd_replot_group,
  argparse entries, dispatch branches); `list` now prints config filenames; 12 archive configs
  moved to `config/experiments/archive/` (28 → 40 files, `git mv`; still usable via `single`).
  Config generation rewritten from scratch for determinism (no env/BE name, fixed hash) so
  inventory is reproducible across machines. 14 new tests (parse-seeds 2, inventory 5, scoped
  resolve 2, run loop 5: delete-and-relaunch with new id verified against a real sqlite store,
  done-runs skipped, num_clients from config and override, scoped experiment filtering);
  581 tests green; ruff zero new findings (pre-existing 4 remain, all in untouched legacy
  replot code; the removed group/replot code took its 10 findings with it); mypy zero new
  errors (only pre-existing loader.py/test_server.py notes). Dry-run over an empty DB reports
  1,200 missing. `src/data/` still untracked/gitignored.
- 2026-08-17: **IMPL-14 closed.** Tests, lint, smoke validation (spec §9.4/§9.5). Gates:
  ruff 0 findings (57 files), mypy 0 errors, **651 tests green** (baseline 581 at IMPL-12
  closure). Smoke acceptance (fresh sqlite DB, 4 cells × 4 clients × 1 run, real Ray+Flower
  execution): `scripts/verify` → `runs: 4  failed_checks: 0`, exit 0. Per cell:
  `nonprivate` budget PASS (nonprivate), other checks SKIP; `dpfedavg_fixed` /
  `fedprox_fixed` budget PASS utilization=1.0000 (final_rdp 10.0000, 20×0.5 exactly),
  drop-out PASS never=100.00% round=21.0 (T+1), warm-up SKIP (no warm-up phase, §4.4);
  `pldpbo_nun` warm-up PASS mean=1.3995±0.0000 (grid sum, parity 0.0000), budget PASS
  utilization=0.9813 (final_rdp 9.8132; in the 0.98–1.02 band), drop-out PASS
  never=100.00% round=32.0 (T+1). Bugs found by real-DB verification and fixed: (1) the
  server logged `client_state` under the mkstemp basename (`client_state_<rand>.json`) —
  `tracker.log_artifact` keeps basenames — now staged as canonical `client_state.json` in a
  per-run temp dir (also removes a parallel-run `os.replace` race); (2) `scripts/verify`
  iterated the wrapped payload's `.values()` so every check saw one bogus entry — payload
  unwrapped at load, per-run load failures degrade to SKIP instead of aborting the report;
  (3) the warm-up sum counted any client's first-10 participations regardless of phase —
  now phase-filtered ("warmup" participations only), which stays correct for
  `fraction_fit < 1`; this surfaced the client's transition-round phase tag (the warm-up
  grid spend of round 10 was tagged "bo" because the phase was read post-step) — the client
  now tags the spend's phase (pre-step), pinned by a 12-round reply test. Documented
  deviations from the IMPL-14 expected table: fixed cells SKIP warm-up (nominal 1.3995 is
  BO-specific; impossible at utilization 1.0 with constant r_t); pldpbo_nun drop-out round
  = 32.0 not 21.0 (T raised 20→31 per the plan's contingency so the budget is spent, §4.5
  fill unreachable under the production 0.1 budget margin — the band, not the nominal,
  defines acceptance); FEMNIST writers = 3,597 not 3,598 (on-disk ground truth; LEAF
  canonical; `femnist_counts` unwraps the dict-shaped keys file; expected counts
  654,281/163,570/3,597 re-verified on disk). CI: `.github/workflows/ci.yml` (uv sync +
  ruff + mypy + pytest; runs on push to main / pull requests — will execute on the user's
  next push; branch merged locally, never pushed per repo policy). `src/data/` still
  untracked/gitignored.
