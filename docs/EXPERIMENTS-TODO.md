# EXPERIMENTS-TODO — Experiment Specification and Paper Data Contract

**Status:** active (pre-run). **Owner (runs):** user (implementation repo). **Owner (aggregation script):** user (this file specifies the output contract). **Owner (paper):** assistant — consumes the report artifact to write Results/Discussion/Conclusion and update Introduction claims.

This file is the **binding contract** between the experiment code (MLflow), the aggregation script, and the paper (`paper/sections/*.tex`). Every number that will appear in the paper must be traceable to a block of the report artifact produced by the aggregation script. The authoritative references are `docs/PLDP-BO.md` v0.3 (algorithm) and `paper/sections/exp_methodology.tex` (current, post-fix). Where this file and the paper disagree, **the paper text wins** — tell the assistant so the discrepancy can be resolved.

---

## 1. Workflow

```
implementation repo (training) ──► MLflow tracking server
                                        │
                    aggregation script (reads MLflow)
                                        │
          paper/experiments/report.md  +  report.json  +  figures/*.pdf  +  curves/*.csv
                                        │
                              assistant (paper writer) ──► Results/Discussion/Conclusion,
                                        │                    Introduction I4, memos
                                        └─────► paper/sections/*.tex
```

1. Run the experimental matrix (§3) with the MLflow logging schema (§4). One MLflow run per (method, seed) cell.
2. Run the aggregation script (§5) → produces the report artifact (§6).
3. Assistant consumes the artifact, updates the paper, and marks this file's status section (§9).

### 1.1 Out of scope — do NOT run

| Item | Reason |
|---|---|
| λ_aq sweep (ablation) | M7: λ_aq fixed at 0.1; no tuning study. "Ablation Study" subsection of Results will be removed. |
| Multi-budget sweep (B_RDP ∈ {5, 20, …}) | M1/M1b: single matched budget B_RDP = 10.0; strong/weak baseline rows removed. |
| FedAWA / FedStrag runs | B3: removed from paper and bibliography. |
| ε / δ reporting | C3: privacy reported in RDP space at α₀ only. |
| Per-step DP-SGD accounting | M11: the implemented accountant treats one communication round as one Gaussian release. |
| Client-level LDP variants | C1: paper and implementation are record-level per-example DP-SGD. |

---

## 2. Locked configuration (the script must assert these)

| Constant | Value |
|---|---|
| Communication rounds `T` | 200 |
| Clients `K` | 100 |
| Participation fraction `ρ` | 0.1 (≈ 10 clients/round; ≈ 20 participations/client) |
| Local epochs `E` | 5 |
| Local batch size `B` | 64 |
| Local optimizer | SGD, momentum 0.9 |
| Server learning rate `η` | 0.01 |
| Clipping norm `C` | 1.0 (per-example) |
| RDP order `α₀` | 10.0 (fixed; no conversion, no δ) |
| Total RDP budget `B_RDP` | 10.0 per client, hard constraint |
| RDP search interval `[R_min, R_max]` | [0.01, 2.0] |
| Subsampling rate `q` | q = B / n_i (per client) |
| Sigma calibration | σ_t = √(α₀·q²/(2·R_t)) |
| **Fixed baselines** | R = B_RDP/(ρ·T) = **0.5 per round** → cumulative ≈ 10.0 by T (budget-matched) |
| **Warm-up grid** | 10 log-spaced points over [0.01, 0.5], ratio r = 50^(1/9) ≈ 1.5444:<br>0.01, 0.0154, 0.0239, 0.0368, 0.0569, 0.0879, 0.1357, 0.2095, 0.3236, 0.4998<br>**sum ≈ 1.3995 ≈ 1.40** (≈14% of B_RDP; ≈ 8.60 remains for BO) |
| BO kernel / grid `G` / penalty `λ_aq` | Matérn 5/2 / 50 points / 0.1 |
| GP observation noise | auto-learned (σ_n²); distinct from DP σ² |
| Seeds `N` | 12 runs per cell, **seeds 0–11** (min Wilcoxon p = 1/2¹² ≈ 2.4e-4); NumPy, PyTorch, and client-selection RNGs all seeded per run |
| Target thresholds | 80%, 90%, 95% of **non-private FedAvg final accuracy**, computed per (dataset, **partition**) from that cell's non-private runs |
| Fixed-round anchors | rounds 50, 100, 150, 200 |
| Validation subset | 10% hold-out of each client's train data, never used for local training; fixed per client; supplies L_val / L_clean / L_noisy for variants B–G |
| Clean (pre-DP) pass | Variants C–G additionally train the local model **without** DP noise (same seed) to obtain L_clean / clean logits → ≈2× local training cost for these variants |
| Aggregation | median-based robust-weighting attenuation for **all private methods** (PLDP-BO variants and both fixed-cost baselines); plain FedAvg averaging only for `nonprivate` |
| Accounting convention | each communication round = one Gaussian release; per-round cost = accountant unit |

**Datasets** (Table 1 of the paper):

| Dataset | Train | Test | Classes | Model |
|---|---|---|---|---|
| MNIST | 60,000 | 10,000 | 10 | MLP 200–200 (≈200K params) |
| CIFAR-100 | 50,000 | 10,000 | 100 | CNN 2-conv/FC-512 (≈2.17M params) |
| FEMNIST | ≈654,281 | ≈163,570 | 62 | CNN 2-conv/FC-512 (≈1.66M params) |

- FEMNIST: 817,851 total images, 3,598 writers (`femnist_user_keys.pt`), 80/20 **sample-based** split (LEAF preprocessing `-t sample`). Client construction uses the writer partition on the **train** split. Exact counts must be re-read from the `.pt` files during extraction (verification check, §5.1).
- CNN: conv 32 → conv 64 (3×3, pad 1), ReLU, 2×2 max-pool ×2 → FC 512 → dropout 0.25 → softmax. FEMNIST input 28×28 (FC input 3136); CIFAR-100 input 32×32 (FC input 4096).

---

## 3. Experimental matrix (full methods everywhere — user decision 2026-08-14)

**Methods (10):** `nonprivate` (plain FedAvg averaging), `dpfedavg_fixed` (R = 0.5/round), `fedprox_fixed` (μ = 0.01, R = 0.5/round), `pldpbo_nun`, `pldpbo_utility`, `pldpbo_retention`, `pldpbo_efficiency`, `pldpbo_perremaining`, `pldpbo_snr`, `pldpbo_agreement`.

**Cells (dataset × partition × 10 methods × 12 seeds ≈ 1,200 runs):**

| Dataset | Partitions | Cells | Runs |
|---|---|---|---|
| MNIST | IID, Dirichlet α=1.0, α=0.5, α=0.1, pathological | 5 | 600 |
| CIFAR-100 | IID, Dirichlet α=0.5, α=0.1, pathological | 4 | 480 |
| FEMNIST | natural (writer) | 1 | 120 |

- Pathological: 2 classes/client — MNIST 20 clients/class, CIFAR-100 2 clients/class. FEMNIST exempt.
- Dirichlet: symmetric Dir(α·1_K) label proportions (protocol of li2020federated).
- Min 30 samples/client (MNIST, CIFAR-100); FEMNIST writers with <10 samples merged with the nearest writer cluster.
- The **core cells** for headline claims: MNIST α=0.5, CIFAR-100 α=0.5, FEMNIST natural.

**Run policy:** only MLflow runs with status FINISHED are included; partial runs (crashed before round T = 200) are excluded and flagged in the §6.1 inventory, then rerun with the same seed. Reruns overwrite the same run name; `config_version` must match.

**Compute note:** ≈1,200 runs. MNIST MLP ≈5–15 min/run, CIFAR-100 CNN ≈30–60 min, FEMNIST CNN ≈1–2 h (GPU); variants C–G cost ≈1.5–2× the others (clean pass + extra validation evaluations). Plan for several hundred GPU-hours total.

**Verification-only items (no training runs needed):**
1. FEMNIST exact train/test/writer counts from `femnist_train.pt`, `femnist_test.pt`, `femnist_user_keys.pt`.
2. Warm-up sum and per-round accountant-cost parity (computed from logged data, §5.1).

---

## 4. MLflow logging schema

### 4.1 Experiments and runs

- **Experiment name:** `<dataset>_<partition>` — e.g. `mnist_dirichlet_0.5`, `cifar100_pathological`, `femnist_natural`.
- **Run name:** `<method>_seed<NN>`.
- **Tags (must be set):** `dataset`, `partition`, `method`, `seed`, `config_version` (hash of §2 constants), `code_git_hash`.

### 4.2 Params (log all — the script asserts equality with §2)

`T`, `K`, `rho`, `E`, `B`, `eta_server`, `local_opt` = `sgd_momentum0.9`, `clip_norm` = 1.0, `alpha0` = 10.0, `B_RDP` = 10.0, `R_min` = 0.01, `R_max` = 2.0, `warmup_points` (JSON list of 10 values), `warmup_sum_nominal` = 1.3995, `lambda_aq` = 0.1, `kernel` = `matern_5_2`, `G` = 50, `N` = 12 (run index), `mu_fedprox` = 0.01 (FedProx only), `model` (mlp200x2 | cnn), `dataset_sizes` (JSON: train/test/writers), `partition_kwargs` (JSON: alpha or pathological), `seeds` (JSON: seed per RNG), `validation_frac` = 0.1, `aggregation` (`attenuation` | `plain`), `enforce_budget` = true, `dataset_root`, `data_hash` (sha256 over the dataset files).

### 4.3 Metrics (log with `step = round`; `step=0` = before training)

| Metric | Definition |
|---|---|
| `acc_test` | global model test accuracy at round t (all datasets) |
| `f1_test` | global model macro-F1 at round t |
| `n_participants` | clients selected this round |
| `mean_r_t` | mean of selected clients' R_t this round |
| `mean_cum_rdp` | mean cumulative RDP across selected clients |
| `budget_utilization` | final cumulative RDP / B_RDP (log at final step) |
| `bytes_round` | total bytes transmitted server↔clients this round |
| `bo_time_round` | per-round wall time: GP fit + acquisition + selection (per-client, mean) |
| `acct_time_round` | per-round wall time: budget check + σ calibration (per-client, mean) |
| `bo_overhead_pct` | cumulative BO time / total training time (final step) |

**Per-client test accuracy (FEMNIST only):** log as artifact `client_test_acc.json` at the final step (not a metric). Rationale: per-client evaluation is reported for FEMNIST; avoid 200-round × per-client evaluations for MNIST/CIFAR-100.

### 4.4 Artifacts (per run, one JSON/npz per client per round is NOT required — store per-client arrays)

Log one artifact `client_state.json` (or `.npz`) with keys per client id `i`:

| Key (per client `i`, arrays indexed by participation round) | Description |
|---|---|
| `r_t[i]` | selected RDP cost per participation round |
| `cum_rdp[i]` | accountant-reported cumulative RDP after each participation |
| `remaining_rdp[i]` | B_RDP − cum_rdp after each participation |
| `phase[i]` | `warmup` / `bo` / `exhausted` (marker per participation) |
| `warmup_rounds[i]` | the global round indices of the client's warm-up participations |
| `dropout_round[i]` | first global round participation is refused because remaining < R_min; `null` if never (survives to T = 200) |
| `observed_m[i]` | the scalar objective observation m_{i,t} fed to BO (all variants) |
| `acct_cost[i]` | **accountant-reported per-round RDP cost** (must ≈ r_t_final; drives the warm-up sum check) |
| `r_t_candidate[i]` | BO-proposed R_t **before** budget enforcement |
| `r_t_final[i]` | enforced R_t actually used (σ_t calibrated from this; binary search in [R_min, r_t_candidate]) |
| `enforcement_count[i]` | number of times budget enforcement reduced r_t for this client |

**Variant components** (needed for schedule analysis and validation; log once per client per participation, same file):

| Component | Used by |
|---|---|
| `L_clean[i]`, `L_noisy[i]` | Utility, Retention, Efficiency, PerRemaining |
| `update_norm_noisy[i]` = ‖Δ̃_{i,t}‖₂ | NUN |
| `update_norm_clean[i]` = ‖Δ_{i,t}‖₂, `sigma[i]` | SNR |
| `agreement[i]` (logit agreement clean vs noisy) | Agreement |
| `R_remaining` (same as above) | PerRemaining |

Formulas in the paper (solution.tex §Optimization Objectives): `m_nun = ‖Δ̃‖₂`; `m_util = L_val(w_DP)`; `m_ret = L_noisy/L_clean`; `m_eff = −max(0, L_noisy−L_clean)/(L_clean·R_t)`; `m_rem = −max(0, L_noisy−L_clean)/(L_clean·R_remaining)`; `m_snr = ‖Δ‖₂²/σ²`; `m_agr = logit agreement`. All L_* evaluations use the client's validation subset (§2).

**N/A rules:** `nonprivate` logs no privacy fields (`r_t_candidate`, `r_t_final`, `cum_rdp`, `remaining_rdp`, `phase`, `dropout_round`, `acct_cost`, `observed_m`, variant components, `enforcement_count` — all null/absent). Fixed baselines: `r_t_candidate == r_t_final == 0.5` constant, `bo_time_round` = 0, no `observed_m` (no BO), `enforcement_count` = 0, `phase` = `bo` throughout. The script must not average N/A fields.

---

## 5. Aggregation script — duties

Reads MLflow (URI + filters from provenance), computes, writes the artifact (§6). **Every aggregation must include run counts `n`** — no `n<12` cell may be silently averaged.

### 5.1 Verification checks (report first, before any result tables)

1. **Warm-up sum:** for each private method × run × client: Σ over the client's first 10 participations of `acct_cost`; report mean ± SD across clients and runs vs nominal 1.3995; pass criterion: mean within ±5% (1.33–1.47). Also per-round parity: `acct_cost` vs `r_t_final` relative error (median, max).
2. **Budget matching:** mean final cumulative RDP per method ≈ 10.0 for all private methods (utilization ≈ 1.00 ± tolerance 0.02); non-private = 0.
3. **Drop-out:** per method: distribution of `dropout_round` (never = T+1), fraction never dropping out, mean ± SD drop-out round, mean final cumulative RDP.
4. **FEMNIST counts:** read the `.pt` files; report exact train/test/writer counts (compare vs ≈654,281 / ≈163,570 / 3,598).
5. **Accounting convention:** confirm per-round cost logged by the accountant equals the RDP cost of the round's single Gaussian release (covered by check 1).

### 5.2 Aggregations

- Per (dataset, partition, method): mean ± SD over the 12 runs of: final accuracy, final macro-F1, final cumulative RDP, utilization, drop-out stats, accuracy at rounds {50, 100, 150, 200}, rounds-to-target {80%, 90%, 95%} (per that cell's non-private final accuracy; mean, SD, `reached_n`, `not_reached_n`; "not reached" is a legal value, not an error).
- Participation: per (dataset, partition, method): mean ± SD of participations per client (`mean_participations`; expected ≈20 for non-private, less for methods with drop-out), enforcement stats (mean `enforcement_count` per client, fraction of participation rounds where enforcement reduced R_t), phase-wise mean R_t per variant (warm-up / early BO [first 30 BO participations] / late BO [last 30]) — RQ1 evidence that BO adapts the schedule.
- FEMNIST only: per-client test accuracy distribution (mean ± SD across clients, and across runs) from `client_test_acc.json`.
- Per-round curves: mean ± SD of `acc_test`, `f1_test` over runs.
- Statistics (per dataset, **at the α=0.5 core partition** and per partition for heterogeneity cells):
  - Wilcoxon signed-rank, all-pairs over the 10 methods, **on final accuracy only** → one 10×10 matrix (7×7 sub-matrix for variants).
  - Cohen's d (pooled SD) matrices for the same pairs.
  - Bonferroni-adjusted α = 0.05 / C(10,2) = 0.05/45 ≈ 0.001111; also compute the 7-variant-only adjustment 0.05/21 ≈ 0.002381 (RQ2 comparisons). Report both thresholds.
  - Significance markers: `*` = p < adjusted α.
  - **No hypothesis test on cumulative RDP:** under the matched budget all private methods finish at ≈10.0 with near-zero variance → Wilcoxon is degenerate (ties). Cumulative RDP / utilization are reported **descriptively only** (mean ± SD, distributions, schedules below). Matches the paper's statistical methodology (2026-08-14 edit).
- Schedules: per (dataset, partition, method): per-round mean R_t over clients and runs + quartiles (for box/violin plots), per-client final cumulative RDP distribution, warm-up vs BO phase boundary (round indices), drop-out histogram data.
- Timing: per (dataset, method): mean ± SD of `bo_time_round`, `acct_time_round`, `bo_overhead_pct`, `bytes_round`.
- Trade-off: per (dataset, partition, method): mean final accuracy vs mean final cumulative RDP (scatter data), and accuracy delta vs non-private at matched cumulative RDP.

### 5.3 Figure rendering (matplotlib, journal style)

Two-column width ≈ 3.5 in, font ≥ 8 pt, 300 dpi, saved as **PDF** (vector) + PNG; axis labels with units; series legend; mean lines with ±SD shaded band (α=0.3 fill).

| Figure file | Content |
|---|---|
| `fig-convergence-<dataset>.pdf` | accuracy vs round, 10 method curves (core partition per dataset) |
| `fig-schedules-<dataset>.pdf` | per-round mean R_t curves (7 PLDP-BO variants), quartile bands, warm-up region shaded |
| `fig-dropout-<dataset>.pdf` | histogram of drop-out rounds per method (or empirical CDF) |
| `fig-heterogeneity.pdf` | final accuracy bars per partition, two panels (MNIST: 5 partitions; CIFAR-100: 4), methods nonprivate, dpfedavg_fixed, pldpbo_nun, pldpbo_utility, pldpbo_snr |
| `fig-tradeoff-<dataset>.pdf` | scatter: mean final accuracy vs mean final cumulative RDP per method (±SD error bars) |
| `fig-timing.pdf` | bar chart: BO overhead % per method per dataset (grouped) |

Violin / per-round box plots for schedules are rendered from **raw MLflow per-client data** (not from report aggregates); the curves CSVs carry quartiles for re-plotting.

### 5.4 Curves CSVs

One CSV per (dataset, partition, method): columns `round, acc_mean, acc_sd, f1_mean, f1_sd, r_t_mean, r_t_q25, r_t_q50, r_t_q75, n`. Naming: `curves/<dataset>_<partition>_<method>.csv`. (Enables re-plotting with different styling.) The `r_t_*` columns are empty for `nonprivate`.

---

## 6. Report artifact — output template (the contract with the paper)

Output location: `paper/experiments/` → `report.md`, `report.json`, `figures/*.pdf`, `curves/*.csv`.

### 6.1 `report.json` schema (machine-readable; exact numbers, ≥6 significant figures)

```jsonc
{
  "provenance": {                    // §6.2
    "mlflow_tracking_uri": "...", "experiments": {...}, "extracted_at": "...",
    "code_git_hash": "...", "config_version": "...", "papers_commit": "...",
    "versions": {"python": "...", "torch": "...", "gpytorch": "..."}
  },
  "meta": {                          // display names for report tables
    "display_names": { "nonprivate": "Non-private FedAvg", "dpfedavg_fixed": "DP-FedAvg (fixed)",
      "fedprox_fixed": "FedProx (fixed)", "pldpbo_nun": "PLDP-BO-NUN",
      "pldpbo_utility": "PLDP-BO-Utility", "pldpbo_retention": "PLDP-BO-Retention",
      "pldpbo_efficiency": "PLDP-BO-Efficiency", "pldpbo_perremaining": "PLDP-BO-PerRemaining",
      "pldpbo_snr": "PLDP-BO-SNR", "pldpbo_agreement": "PLDP-BO-Agreement" }
  },
  "inventory": {                     // completeness matrix
    "dataset_partition": { "method": { "seeds_done": [1..12], "failed": [], "missing": [] } }
  },
  "verification": {                  // §5.1
    "warmup": { "method": { "sum_mean": 0.0, "sum_sd": 0.0, "n": 0, "pass": true, "per_round_parity_median": 0.0 } },
    "budget_match": { "method": { "final_rdp_mean": 0.0, "utilization_mean": 0.0, "n": 0 } },
    "dropout": { "method": { "fraction_never": 0.0, "dropout_round_mean": 0.0, "dropout_round_sd": 0.0 } },
    "enforcement": { "method": { "mean_count_per_client": 0.0, "fraction_reduced_rounds": 0.0 } },
    "femnist_counts": { "train": 0, "test": 0, "writers": 0 }
  },
  "overall": {                       // per cell
    "dataset_partition": { "method": {
      "acc_final": {"mean": 0.0, "sd": 0.0, "n": 12},
      "f1_final": {"mean": 0.0, "sd": 0.0, "n": 12},
      "cum_rdp_final": {"mean": 0.0, "sd": 0.0, "n": 12},
      "utilization": {"mean": 0.0, "sd": 0.0},
      "mean_participations": {"mean": 0.0, "sd": 0.0},
      "acc_at": {"50": {...}, "100": {...}, "150": {...}, "200": {...}},
      "rounds_to_target": {"80": {"mean": 0.0, "sd": 0.0, "reached_n": 0, "not_reached_n": 0},
                           "90": {...}, "95": {...}}
    } }
  },
  "curves": { "dataset_partition_method": { "rounds": [...], "acc_mean": [...], "acc_sd": [...],
             "f1_mean": [...], "f1_sd": [...], "r_t_mean": [...], "r_t_q25": [...],
             "r_t_q50": [...], "r_t_q75": [...], "n": 12 } },
  "stats": { "dataset_partition": {
      "wilcoxon_acc_p": { "methodA": { "methodB": 0.0 } },
      "cohens_d_acc": { "methodA": { "methodB": 0.0 } },
      "adj_alpha_all": 0.001111, "adj_alpha_variants": 0.002381,
      "rdp_reported": "descriptive only (degenerate under matched budget)" } },
  "schedules": { "dataset_partition_method": {
      "dropout_rounds": [...], "final_rdp": [...], "warmup_rounds_used": [...],
      "phasewise_rt": {"warmup": {"mean": 0.0, "sd": 0.0}, "early_bo": {...}, "late_bo": {...}},
      "enforcement": {"mean_count_per_client": 0.0, "fraction_reduced_rounds": 0.0},
      "mean_participations": {"mean": 0.0, "sd": 0.0},
      "client_test_acc": {"mean": 0.0, "sd": 0.0, "n_clients": 0} } },  // FEMNIST only
  "timing": { "dataset_method": { "bo_time_per_round": {...}, "acct_time_per_round": {...},
             "bo_overhead_pct": {...}, "bytes_per_round": {...} } },
  "tradeoff": { "dataset_partition_method": { "acc_final_mean": 0.0, "cum_rdp_final_mean": 0.0,
              "acc_delta_vs_nonprivate": 0.0 } }
}
```

### 6.2 `report.md` sections (prose-ready; the assistant reads this first)

1. **Provenance** — §6.1 block, plus the asserted §2 constants and the pass/fail list of §5.1 checks.
2. **Run inventory** — completeness table (rows = dataset×partition, cols = methods, values = seeds done/failed/missing). Any failed run → cause + rerun decision.
3. **Verification results** — §5.1 outputs (incl. enforcement); each check labeled PASS/FAIL with the numbers.
4. **Overall performance** — one table per dataset (core partition): rows = 10 methods, columns = final acc (mean±SD), final macro-F1, final cum. RDP, utilization, never-dropout %, significance markers vs dpfedavg_fixed (from §5.2).
5. **Heterogeneity** — same table per extra partition (MNIST × 5, CIFAR-100 × 4), plus fig-heterogeneity data.
6. **Objective comparison (RQ2)** — 7×7 sub-matrix of Wilcoxon p (final acc) among variants with `*` markers at adj_alpha_variants; per-variant schedule summaries (mean R_t by phase, final cumulative RDP, drop-out).
7. **Convergence** — accuracy at {50,100,150,200} tables; rounds-to-target table with `not reached (n/N)` cells; curve CSVs referenced.
8. **Privacy–utility trade-off** — matched-budget table: acc delta vs non-private at equal cumulative RDP; scatter data.
9. **Efficiency** — timing table per dataset×method; bytes/round (expected identical across methods — flag if not).
10. **Figures** — the file list with one-line caption text suggested for each (assistant may rewrite).

### 6.3 Formatting conventions (script output)

- Accuracy / F1: 3 decimals (`87.312`); mean±SD as `87.312 ± 1.234`.
- RDP: 2 decimals (cumulative), 3 decimals (R_t); utilization as `0.98`.
- p-values: 3 decimals, `p < 0.001` beyond; `*` = significant at the stated adjusted α; report the adjusted α with every table.
- Cumulative RDP / utilization: **descriptive only** (mean±SD, distributions) — no hypothesis tests (degenerate under the matched budget).
- Missing / not applicable: `—`; unreached threshold: literal `not reached` + count.
- Every table cell carries `n` when < 12 (flagged in §2 inventory too).

---

## 7. Paper mapping (what each block feeds)

| Report block | Paper target |
|---|---|
| §3 verification (warm-up, budget, drop-out, enforcement) | Results "Privacy Adaptation Across Clients" + drop-out claims; Introduction drop-out clause (I8); M2/M11 verification note |
| §4 overall performance tables | Results "Overall Performance" (Table: final acc/F1 per method per dataset) |
| §5 heterogeneity tables | Results "Impact of Data Heterogeneity" |
| §6 objective comparisons | Results "Impact of the Optimization Objective" (RQ2); Discussion "Effect of the Optimization Objective" |
| §7 convergence + rounds-to-target | Results "Convergence Analysis" (+ figure) |
| §4/§8 trade-off | Results "Privacy–Utility Trade-off" (matched budget) |
| §9 timing/bytes | Results "Computational and Communication Efficiency" (RQ4) |
| figures | Results figures; Discussion; Introduction I4 empirical claims (final accuracy numbers, matched-budget claim, overhead claim) |
| schedule data | Results "Privacy Adaptation Across Clients" (RQ1) — violin/box plots, warm-up/BO phase boundary |
| `overall` numbers | Table 1 FEMNIST exact counts (if they differ from ≈), Conclusion "Summary of Findings" |

Results outline actions for the assistant once data exists: **"Impact of the Privacy Budget" → replaced by "Budget Utilization and Client Drop-out"**; **"Ablation Study" → removed** (no λ_aq sweep).

---

## 8. Definition of done (assistant can only write Results when:)

1. `report.md` + `report.json` + all `figures/*.pdf` + `curves/*.csv` exist in `paper/experiments/`.
2. Inventory: every cell of §3 complete with n = 12; no failed runs unaccounted.
3. All §5.1 checks PASS (warm-up sum 1.33–1.47, utilization 0.98–1.02 private methods, FEMNIST counts consistent).
4. Statistics computed with the two stated adjusted α thresholds; no undefined p-values.
5. `n` present in every aggregate; no silent imputation.
6. Implementation is spec-conformant: the §9 worklist is closed — config assertion active, per-round accounting, log-grid warm-up, momentum DP-SGD, per-client validation subset + clean pass, official test-set evaluation, full §4 logging schema live, FEMNIST counts verified at extraction (9.9).

Then the assistant will: write `results.tex` (9 content subsections after outline revision), `discussion.tex` (7 subsections incl. Threats to Validity: budget-match caveat M1, warm-up/budget interaction M2, per-round-vs-per-step approximation C5/M11, record-level semantics C1), `conclusion.tex` (mirroring the 4 contributions, contribution 4 now evidence-based), update `introduction.tex` I4/abstract-level claims, update Table 1 if exact FEMNIST counts differ, update memos (06-placeholders → Stage 3 re-review), and report back.

---

## 9. Implementation gap-closure worklist (audit 2026-08-14)

### 9.0 Audit verdict

Audited: implementation repo `/home/will/Codigo/Pesquisa/pldp-bo` (HEAD `8b925cd`, 2026-08-11 — predates this spec). Result: **not spec-conformant; no experiment data exists.**

| Area | Result |
|---|---|
| §2 locked constants | 1/13 PASS (α₀ = 10.0 RDP-native, no ε/δ) — gaps in 9.1–9.5 |
| §3 datasets/partitions | PARTIAL — MNIST/CIFAR-100, IID + Dirichlet(α=0.5) exist; FEMNIST, pathological, α∈{1.0, 0.1}, min-30, writer merge absent — 9.8–9.9 |
| §4 MLflow schema | 0/8 PASS — experiment/run naming, tags, spec params, per-round metrics, artifacts all absent or renamed — 9.10 |
| §3 matrix runner | PARTIAL — generic `scripts/run` (single/group/-n) exists; no matrix configs, no seeds 0–11 policy — 9.11 |
| §5 aggregation script | absent — user writes it per §5/§6 (unchanged by this worklist) |
| Run data | current `mlflow.db`: 0 runs; all legacy runs (`mlruns/group/*`, `/home/will/Modelos/pldp-bo/*`) pre-spec (K=2500, T=500, no method/partition/dataset tags); `paper/experiments/` does not exist |
| FEMNIST data | **not available yet** (user decision 2026-08-14) — pipeline implemented now (9.9), counts verified at extraction |

Work items are ordered by dependency. Each gives the exact change; acceptance = the stated check + the §5.1 verification checks.

### 9.1 Locked-config layer (`src/config/loader.py`, `config/experiments/*.yaml`)
- **Requirement:** §2 constants fixed and asserted.
- **Current:** defaults differ from the spec and shipped configs deviate further — T=15–50, K=8, ρ=0.5–0.6, E=1–4, momentum=0.0, C=5.0, η_s=0.5, total_budget=120–200, λ_aq=0.3, grid_points=100, min_warmup=5; per-client budget derived by equal division of `total_budget` or data-proportional personalization (`src/server_app.py:53–168`).
- **Required change:** startup assertion (fail fast on mismatch) for: T=200, K=100, ρ=0.1, E=5, B=64, lr=0.01, momentum=0.9, clip_norm C=1.0, α₀=10.0, **per-client B_RDP=10.0 flat** (personalization/data-proportional budgets disabled), [R_min, R_max]=[0.01, 2.0], λ_aq=0.1, G=50, kernel=matern52; drop `weight_decay`/`gradient_clip_norm` extras from the matrix configs.
- **Acceptance:** any deviation fails at startup; §5.1 budget-match check (utilization 0.98–1.02).

### 9.2 Per-round accounting + σ calibration (per_example mode: `src/privacy/per_update_dp.py`, `src/client/per_example_dp_client.py`, `src/client_app.py`)
- **Requirement (§2, M11):** 1 communication round = 1 Gaussian release; σ_t = √(α₀·q²/(2·R_t)) with q = B/n_i.
- **Current:** per_example mode calibrates per-step — R_t/num_steps with num_steps = E×len(trainloader) (`client_app.py:488–495`), accountant steps `num_steps` releases (`per_example_dp_client.py:284–290`) ⇒ σ carries an extra √(num_steps) factor; per_update mode steps once but uses σ = √(α₀·C²/(2R)) (C², not q²).
- **Required change:** in the per_example (record-level) mode — σ_t = √(α₀·q²/(2·R_t)); `accountant.step(num_steps=1)` at the round cost R_t; remove per-step composition.
- **Acceptance:** per-round parity `acct_cost` ≡ `r_t_final` (relative error ≤ 1e-6); §5.1 warm-up sum 1.33–1.47.

### 9.3 Log-spaced warm-up grid (`src/privacy/bo_scheduler.py:344`)
- **Requirement:** 10 log-spaced points over [0.01, 0.5], ratio 50^(1/9) ≈ 1.5444: 0.01, 0.0154, 0.0239, 0.0368, 0.0569, 0.0879, 0.1357, 0.2095, 0.3236, 0.4998 (sum ≈ 1.3995).
- **Current:** `np.linspace(rdp_min, rdp_max, warmup_rounds)` — linear over the full range (sum ≈ 10.05 with defaults); `min_warmup` = 5.
- **Required change:** hardcode the 10-point grid constant; warm-up count = 10; log `warmup_points` param. (If the per-update scheduler path is kept, mirror it at `epsilon_scheduler.py:123–125`.)
- **Acceptance:** §5.1 warm-up sum mean 1.33–1.47 over clients × runs.

### 9.4 Momentum DP-SGD + FedProx proximal term (`src/client/per_example_dp_client.py:124–136`, `src/client/base_client.py:68–73`)
- **Requirement (§2):** SGD momentum 0.9; FedProx baseline μ = 0.01; record-level per-example DP (C1).
- **Current:** `ValueError` guards reject momentum ≠ 0 and `proximal_mu > 0` in per_example mode; proximal term is (μ/2)·Σ‖w−w_global‖₂ (L2, not squared).
- **Required change (user decision 2026-08-14: implement momentum DP-SGD):** remove both guards; apply momentum Opacus-style to the per-step **averaged clipped gradient (pre-noise)** — DP-safe, no per-sample momentum buffers; proximal term → (μ/2)·‖w−w_global‖².
- **Acceptance:** momentum 0.9 + μ=0.01 config runs; smoke convergence; momentum applied post-clip/pre-noise (comment the invariant in code).

### 9.5 Fixed baselines + aggregation routing (`src/privacy/epsilon_scheduler.py:116–133`, `src/server_app.py:244–281`, `src/server/strategy.py`)
- **Requirement (§2 Aggregation):** `dpfedavg_fixed` and `fedprox_fixed` with R = B_RDP/(ρ·T) = 0.5/round; median attenuation for **all** private methods; plain averaging **only** for `nonprivate`.
- **Current:** `FixedRDPScheduler` exists but is never constructed (only restored from state, `client_app.py:169`); fallback divides by T (⇒ 0.05); attenuation gated on `strategy == "pldp_bo"` only.
- **Required change:** instantiate `FixedRDPScheduler(R=0.5)` for the two baselines; add explicit `aggregation: attenuation | plain` config key (server selection independent of strategy name); route variants + both fixed baselines → `MedianRobustAggregation`, `nonprivate` → `SafeFedAvg`.
- **Acceptance:** `aggregation` param logged per run; 4-cell strategy smoke test.

### 9.6 Per-client validation subset + clean pre-DP pass (`src/data/__init__.py:28–75`, `src/client/per_example_dp_client.py:217–220`)
- **Requirement (§2, methodology sentence 2026-08-14):** each client holds out a **fixed 10% of its own** train data (never trained on); reference variants (Retention, Efficiency, PerRemaining, SNR, Agreement) compute the clean reference by **training the local model without DP noise** (same seed) and evaluating both models on the subset; NUN/Utility use only the privatized model.
- **Current:** global 10% hold-out of the full train set shared by all clients; clean stats computed on the **pre-round global model** in per_example mode — wrong semantics (L_clean must come from the locally-trained clean model).
- **Required change:** per-client fixed hold-out drawn from each client's partition (seeded, fixed across rounds); clean local training pass (same E, B, lr, momentum, seed; no clipping/noise) after the DP pass for the reference variants.
- **Acceptance:** `validation_frac` = 0.1 param; L_clean varies per client; accounting unaffected.

### 9.7 Test-set evaluation + macro-F1 (`src/data/__init__.py:21–31`, `src/server_app.py:200–204, 225`)
- **Requirement (§4.3):** per-round `acc_test`/`f1_test` on the **official test set** (MNIST/CIFAR-100 `train=False`; FEMNIST test split); FEMNIST per-client test accuracy artifact.
- **Current:** `accuracy`/`server_loss` on the 10% train hold-out; no test loader; no F1 anywhere.
- **Required change:** load official test splits; evaluate the global model each round → `acc_test`, macro-F1; rename the hold-out evaluation to val accuracy (or drop); write `client_test_acc.json` (FEMNIST, final step).
- **Acceptance:** `acc_test` logged per round; MNIST sanity check in the expected FedAvg range.

### 9.8 Partitions (`src/data/partitioner.py:110–119`)
- **Requirement (§3):** Dirichlet α ∈ {1.0, 0.5, 0.1}; pathological — 2 classes/client (MNIST 20 clients/class, CIFAR-100 2 clients/class); min 30 samples/client; FEMNIST natural writer partition + <10-writer merge.
- **Current:** only `iid`/`noniid` (α=0.5); no pathological branch; no min-30 enforcement.
- **Required change:** add `pathological` partition type (2 non-overlapping classes per client); min-30 enforcement (drop/merge undersized clients — define once, log `partition_kwargs`); writer partition for FEMNIST (9.9).
- **Acceptance:** `partition_kwargs` logged; per-cell client counts documented.

### 9.9 FEMNIST pipeline — blocked on data (implement now, verify at extraction)
- **Requirement (§2/§3):** LEAF 80/20 **sample** split (`./preprocess.sh -s niid --sf 1.0 -k 0 -t sample`) → ≈654,281/≈163,570/3,598 writers; 62 classes; CNN 28×28 (FC input 3136); writer partition on the **train** split.
- **Current:** absent — registry is cifar10/cifar100/mnist; `_MODEL_DATA_COMPAT` rejects `femnist`; no `_INPUT_CHANNELS_MAP` entry.
- **Required change:** loader for `femnist_train.pt` / `femnist_test.pt` / `femnist_user_keys.pt`; registry/compat/channels entries; writer keys → clients; merge writers with <10 samples; CNN input 28×28.
- **Acceptance:** exact counts re-read from the `.pt` files at extraction and reported in §5.1 check 4; `dataset_root`/`data_hash` params logged.

### 9.10 MLflow logging schema (`src/tracking/tracker.py`, `src/server_app.py`, `src/server/strategy.py`, `src/client_app.py`)
- **Requirement: §4 in full.**
- **Current:** experiment always `pldp-bo`; only `group` (+`flower_series_id`) tags; params under renamed keys (`data.*`, `federated.*`, …); per-round metrics = val accuracy/loss + client stats (`epsilon_mean`, `rdp_cost_mean`, …); no timing/bytes instrumentation; `log_artifact` never called.
- **Required change:**
  - Experiment = `<dataset>_<partition>`; run = `<method>_seed<NN>`; tags: `dataset`, `partition`, `method`, `seed`, `config_version`, `code_git_hash`, `group`.
  - Params under the §4.2 names: `T`, `K`, `rho`, `E`, `B`, `eta_server`, `local_opt`, `clip_norm`, `alpha0`, `B_RDP`, `R_min`, `R_max`, `warmup_points`, `warmup_sum_nominal`, `lambda_aq`, `kernel`, `G`, `N`, `mu_fedprox`, `model`, `dataset_sizes`, `partition_kwargs`, `seeds`, `validation_frac`, `aggregation`, `enforce_budget`, `dataset_root`, `data_hash`.
  - Per-round metrics (step = round): `acc_test`, `f1_test`, `n_participants`, `mean_r_t`, `mean_cum_rdp`, `budget_utilization` (final), `bytes_round`, `bo_time_round`, `acct_time_round`, `bo_overhead_pct` (final) — add `perf_counter` timing around GP fit + acquisition (BO time) and budget check + σ calibration (accounting time); byte accounting at server send/receive.
  - Artifact `client_state.json`: per-client arrays `r_t_candidate`, `r_t_final`, `cum_rdp`, `remaining_rdp`, `phase` (add `exhausted`), `warmup_rounds`, `dropout_round`, `observed_m`, `acct_cost`, `enforcement_count`, + variant components (`L_clean`, `L_noisy`, `update_norm_noisy`, `update_norm_clean`, `sigma`, `agreement`); §4.4 N/A rules for nonprivate/fixed baselines.
  - Artifact `client_test_acc.json` (FEMNIST, final step).
- **Acceptance:** a single smoke run logs every §4 key; §5.1 checks runnable against it.

### 9.11 Matrix runner + run policy (`scripts/run`)
- **Requirement (§3):** 10 methods × 10 partition cells × seeds 0–11; FINISHED-only; same-seed rerun overwrite.
- **Current:** generic launcher (`-n` runs, seed = base + idx); no method/partition matrix; configs seed 42; no FINISHED filter; `start_run` creates duplicates on rerun.
- **Required change:** config template per (dataset, partition) + method override (CLI or 100 generated YAMLs); pin seeds 0–11; count only status FINISHED; rerun = terminate/archive the existing (experiment, run name) then rerun with the same seed.
- **Acceptance:** §6.1 inventory shows all 1,200 cells with n=12 or explicit failures.

### 9.12 Formula-fidelity fixes (SRC checks)
- **SNR:** code uses the **clipped** norm (`per_update_dp_client.py:164–166`); the paper's m_snr = ‖Δ‖₂²/σ² uses the clean (unclipped) update norm — switch to the raw clean norm.
- **Agreement:** code `logit_disagreement = 1 − cos_sim` is the complement of the paper's m_agr (minimization-equivalent) — keep, document the mapping in `meta.display_names`.
- **PerRemaining:** code falls back to R_t when the server sends no remaining budget — wire `remaining_rdp` server-side (9.10) and remove the fallback (paper defines R_remaining).

---

## 10. Status log

- 2026-08-14: file rewritten as the full experiment spec + paper contract (user decisions: full matrix, md+json+figures+curves, user writes the aggregation script).
- 2026-08-14 (implementation audit + gap-closure worklist, user-approved): audited `/home/will/Codigo/Pesquisa/pldp-bo` vs §2/§3/§4 — 1/13 config items, 0/8 logging items PASS; zero spec-conformant runs exist; no aggregation script; FEMNIST data not yet available (pipeline specified in 9.9). §9 worklist (12 items) added as the implementation spec: locked-config assertion, per-round accounting, log-grid warm-up, momentum DP-SGD (user decision), fixed baselines + aggregation routing, per-client validation subset + clean pass, official test-set eval + F1, partitions, FEMNIST, full §4 logging schema, runner policy, formula fidelity. §8 done criteria extended (item 6). Next step: user implements §9 in the repo, runs the §3 matrix, writes the aggregation script per §5/§6.
- 2026-08-14 (audit round, user-approved): validation subset 10% hold-out + clean pre-DP pass for variants C–G (§2); concrete seeds 0–11; aggregation assignment made explicit (attenuation for all private methods); `r_t_candidate`/`r_t_final`/`enforcement_count` logging; N/A rules for nonprivate/fixed baselines; deterministic `client_test_acc.json`; RDP hypothesis tests dropped (descriptive only); rounds-to-target per (dataset, partition); aggregations += mean_participations, phase-wise R_t, enforcement, FEMNIST per-client acc; fig-heterogeneity two panels (MNIST + CIFAR-100); provenance versions + display names; run policy and compute note (§3).
- Prior handoff items folded in: warm-up grid values (§2), warm-up sum verification (§5.1), drop-out logging (`dropout_round` in §4.4), FEMNIST counts (§5.1).
