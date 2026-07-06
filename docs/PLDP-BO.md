# PLDP-BO - Algorithm Description

**Version:** 0.1 (Experimental Development Version)

## Table of Contents

1. [Overview](#1-overview)
2. [Notation](#2-notation)
3. [Federated Learning Workflow](#3-federated-learning-workflow)
4. [Local Differential Privacy](#4-local-differential-privacy)
5. [Privacy Accounting](#5-privacy-accounting)
6. [Bayesian Optimization](#6-bayesian-optimization)
    - [6.1 Warm-up Phase](#61-warm-up-phase)
    - [6.2 Gaussian Process Model](#62-gaussian-process-model)
    - [6.3 Continuous Bayesian Optimization](#63-continuous-bayesian-optimization)
7. [Optimization Metric](#7-optimization-metric)
    - [Variant A: Optimization Stability Metric](#variant-a-optimization-stability-metric)
    - [Variant B: Model Utility Metric](#variant-b-model-utility-metric)
8. [Server Aggregation](#8-server-aggregation)
9. [Complete Client Algorithm](#9-complete-client-algorithm)
10. [Experimental Variants](#10-experimental-variants)

## 1. Overview

PLDP-BO is a Federated Learning (FL) algorithm that continuously personalizes the Local Differential Privacy (LDP) parameter $\varepsilon$ of each participating client throughout the training process. Each client independently runs a Bayesian Optimization (BO) process, preceded by a systematic warm-up exploration phase, to search for the most suitable privacy parameter according to a locally computed objective. The optimization is performed under a hard Rényi Differential Privacy (RDP) budget, ensuring that the cumulative privacy loss never exceeds a predefined limit.

The framework is metric-agnostic: BO only requires a scalar objective to optimize. Consequently, different optimization goals can be adopted without changing the remainder of the algorithm. The experimental evaluation will consider two objective variants:

1. Optimization Stability (FedProx-inspired)
2. Model Utility (Local validation loss)

The remainder of the algorithm is identical for both variants.

## 2. Notation

| Symbol | Description |
|---|---|
| $\varepsilon$ | Privacy parameter (Local DP) |
| $\varepsilon_{\min}, \varepsilon_{\max}$ | Bounds of the privacy search interval |
| $\varepsilon_{\text{budget}}$ | Total Rényi DP budget per client |
| $\varepsilon_{\text{candidate}}$ | Candidate $\varepsilon$ before budget verification |
| $\delta$ | Target delta parameter for DP |
| $C$ | L2 clipping norm |
| $L$ | Number of warm-up exploration rounds |
| $\lambda_{\text{aq}}$ | Privacy penalty coefficient in the acquisition function |
| $E$ | Number of local training epochs |
| $T$ | Total number of communication rounds |
| $\eta$ | Server learning rate |
| $w_g$ | Global model parameters |
| $w_{\text{local}}$ | Local model parameters after training |
| $w_{\text{DP}}$ | Noisy local model after DP perturbation |
| $\tilde{\Delta}_t$ | Noisy DP update at round $t$ |
| $\tilde{\Delta}_{\text{agg}}$ | Aggregated noisy update at the server |
| $\Delta_t$ | Local update before clipping and noise |
| $\hat{\Delta}_t$ | Clipped local update |
| $z_t$ | Gaussian noise vector at round $t$ |
| $\sigma_t^2$ | Noise variance at round $t$ |
| $\sigma$ | Noise standard deviation (DP mechanism) |
| $\sigma_n^2$ | Observation noise variance (GP model) |
| $m_{\text{stab}}$ | Stability optimization metric |
| $m_{\text{utility}}$ | Utility optimization metric |
| $b$ | Median of all received update norms (server) |
| $r_i$ | L2 norm of client $i$'s noisy update |
| $w_i$ | Attenuation weight for client $i$'s update |
| $\mathcal{GP}(\mu, k)$ | Gaussian Process with mean function $\mu$ and kernel $k$ |
| $\text{EI}(\varepsilon)$ | Expected Improvement acquisition function |
| $\alpha(\varepsilon)$ | Acquisition function with privacy penalty |

## 3. Federated Learning Workflow

Each communication round proceeds as follows.

### Client

1. Receive the current global model $w_g^{(t)}$.
2. Select the privacy parameter $\varepsilon_t$ according to the current phase:
   - Warm-up phase: use the predefined deterministic exploration strategy.
   - BO phase: maximize the acquisition function (see Section 6).
3. Verify that using $\varepsilon_t$ does not exceed the client's remaining RDP privacy budget.
   If the budget is violated, reduce $\varepsilon_t$ to the largest feasible value (for both phases, see Section 5).
4. Train locally for $E$ epochs.
5. Compute the local update $\Delta_t = w_{\text{local}}^{(t)} - w_g^{(t)}$.
6. Clip the update $\hat{\Delta}_t = \Delta_t \cdot \min\left(1, \frac{C}{\|\Delta_t\|_2}\right)$.
7. Generate Gaussian noise according to the selected privacy level $z_t \sim \mathcal{N}(0, \sigma_t^2 I)$.
8. Produce the DP update $\tilde{\Delta}_t = \hat{\Delta}_t + z_t$.
9. Send only the noisy update $\tilde{\Delta}_t$ to the server.
10. Compute the chosen optimization metric $m_t$ (Stability or Utility).
11. Store the observation $(\varepsilon_t, m_t)$.
    - If the client is in the BO phase, update the Gaussian Process with the new observation.
    - During warm-up, observations are only collected; the GP is fitted once after the warm-up period.

### Server

For every communication round:

1. Collect all noisy client updates.
2. Compute attenuation weights.
3. Aggregate updates.
4. Update the global model.
5. Broadcast the new global model.

## 4. Local Differential Privacy

The algorithm uses the Gaussian mechanism. Each client clips its update to an L2 sensitivity bounded by $C$.

Noise is sampled as

$$
z \sim \mathcal{N}(0, \sigma^2 I),
$$

where

$$
\sigma = f(\varepsilon, \delta, C)
$$

is determined using the standard Gaussian mechanism with RDP accounting.

The noisy update satisfies one-round Local Differential Privacy.

## 5. Privacy Accounting

Each client maintains an independent Rényi Differential Privacy accountant.

For each communication round:

1. Compute the RDP cost associated with the candidate $\varepsilon$.
2. Add the cost to the accumulated RDP budget.
3. Convert accumulated RDP into $(\varepsilon, \delta)$-DP (Mironov, 2017).
4. Verify $\varepsilon_{\text{total}} \le \varepsilon_{\text{budget}}$.

Budget-constraint enforcement:
If the candidate $\varepsilon$ would exceed the remaining budget, a smaller value is used:

- During the warm-up phase, the exploration strategy selects the largest $\varepsilon \le \varepsilon_{\text{candidate}}$ that satisfies the budget, or clamps to the maximum feasible value.
- During the BO phase, if the acquisition-function maximizer violates the budget, $\varepsilon$ is reduced until the constraint is satisfied (see Section 6.3).

If no feasible positive $\varepsilon$ remains, the client stops participating.

## 6. Bayesian Optimization

Each client maintains an independent Bayesian Optimization (BO) process throughout federated training.

The admissible privacy range is defined as

$$
\varepsilon \in [\varepsilon_{\min}, \varepsilon_{\max}],
$$

where different clients may use different bounds.

The optimization consists of two phases:

- **Warm-up (initial exploration):** the client deliberately evaluates several different $\varepsilon$ values to collect an initial dataset for the Gaussian Process.
- **Continuous optimization:** after warm-up, the client uses Bayesian Optimization to select $\varepsilon$ for each communication round while continuously updating the Gaussian Process with new observations.

Bayesian Optimization does not terminate after warm-up; it continues adapting the client's privacy level during the entire training process.

### 6.1 Warm-up Phase

During the first $L$ communication rounds, the client explores the privacy search space.

The exploration strategy may use:

- a predefined grid,
- evenly spaced $\varepsilon$ values,
- Latin Hypercube sampling,
- or another deterministic sampling strategy.

The objective is to collect representative observations covering the interval $[\varepsilon_{\min}, \varepsilon_{\max}]$.

For each warm-up round:

1. Select the next exploration $\varepsilon$ according to the chosen strategy.
2. Verify the privacy budget. If the selected $\varepsilon$ exceeds the remaining budget, reduce it to the largest admissible value that does not violate the budget (clamp to the feasible maximum).
3. Train locally (Section 3, steps 4-9).
4. Compute the selected optimization metric $m$.
5. Store the observation $(\varepsilon, m)$ in the observation history.

After $L$ observations have been collected, the initial Gaussian Process is fitted. This provides BO with enough information to make informed decisions instead of beginning from an almost uninformative prior.

### 6.2 Gaussian Process Model

The unknown objective function is

$$
f(\varepsilon).
$$

The client models it as

$$
f \sim \mathcal{GP}(\mu, k).
$$

A Matérn 5/2 kernel or RBF kernel may be used.

The GP explicitly models noisy observations:

$$
m_i = f(\varepsilon_i) + \eta_i,
$$

where

$$
\eta_i \sim \mathcal{N}(0, \sigma_n^2).
$$

Observation noise captures uncertainty caused by:

- stochastic SGD,
- minibatch sampling,
- non-IID data,
- Gaussian DP noise.

The observation noise variance $\sigma_n^2$ may be learned during GP fitting.

### 6.3 Continuous Bayesian Optimization

After warm-up, the client enters the optimization phase, which runs for every subsequent communication round until training ends.

**Acquisition function.** To select the next $\varepsilon$, the client maximizes an acquisition function that balances exploration and exploitation while incorporating a preference for stronger privacy:

$$
\alpha(\varepsilon) = \text{EI}(\varepsilon) - \lambda_{\text{aq}} \cdot \frac{\varepsilon - \varepsilon_{\min}}{\varepsilon_{\max} - \varepsilon_{\min}},
$$

where:

- $\text{EI}(\varepsilon)$ is the Expected Improvement over the current best observed value,
- $\lambda_{\text{aq}} \ge 0$ controls the strength of the penalty that favors smaller $\varepsilon$ (stronger privacy),
- the normalization by $\varepsilon_{\max} - \varepsilon_{\min}$ ensures consistent behavior across different search intervals.

**BO cycle.** For each round after warm-up:

1. Use the current Gaussian Process to model $f(\varepsilon)$.
2. Maximize the acquisition function $\alpha(\varepsilon)$ over $[\varepsilon_{\min}, \varepsilon_{\max}]$ to obtain a candidate $\varepsilon^*$.
3. Verify the RDP budget: if $\varepsilon^*$ would exceed the remaining budget, reduce it step-wise (e.g., by binary search or clamping) until the constraint is satisfied.
4. Perform local training using the (possibly reduced) $\varepsilon$.
5. Compute the optimization metric $m$.
6. Append $(\varepsilon, m)$ to the observation history.
7. Update the Gaussian Process with the augmented dataset (refit or incremental update).
8. The updated GP is used to select $\varepsilon$ for the next communication round.

Thus, each client learns an adaptive privacy schedule

$$
\varepsilon_1, \varepsilon_2, \ldots, \varepsilon_T,
$$

where the first $L$ values are determined by systematic exploration and the remaining $T-L$ values are selected by continuous Bayesian Optimization. The GP continually refines its estimate of the privacy-utility relationship as training progresses and the loss landscape evolves.

## 7. Optimization Metric

The BO framework is independent of the objective function. The experiments will evaluate two metric variants, both of which are scalar functions of $\varepsilon$ and the training outcome.

### Variant A: Optimization Stability Metric

**Motivation.** This metric aims to reduce client drift and improve optimization stability, inspired by the intuition behind FedProx. A small private update indicates that the local training step did not diverge excessively from the global model, promoting stable convergence.

**Metric.** After local training and the addition of Gaussian noise, the client constructs the private update $\tilde{\Delta}_t$. The stability metric is defined as

$$
m_{\text{stab}} = \| \tilde{\Delta}_t \|_2 .
$$

Equivalently, $m_{\text{stab}} = \| w_{\text{DP}} - w_g \|_2$, where $w_{\text{DP}} = w_g + \tilde{\Delta}_t$ is the noisy local model. Because the noise variance $\sigma^2$ is a function of $\varepsilon$, the magnitude of $\tilde{\Delta}_t$ directly reflects the chosen privacy level: stronger privacy (smaller $\varepsilon$) adds more noise, typically increasing the norm.

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; m_{\text{stab}} .
$$

**Interpretation.** Lower values indicate that the released update is close to zero, which can result from:

- local training that stays near the global model (small $\|\Delta_t\|$),
- or heavier noise that shrinks the effective update.

By minimizing this norm, the client implicitly balances the desire to contribute a meaningful update against the distortion introduced by privacy noise. This metric evaluates optimization stability under privacy constraints rather than raw predictive performance.

### Variant B: Model Utility Metric

**Motivation.** This metric directly measures predictive quality. Each client keeps a small validation subset that is never used during local training.

**Evaluation.** After generating $\tilde{\Delta}_t$, the client constructs the private local model

$$
w_{\text{DP}} = w_g + \tilde{\Delta}_t .
$$

The model is evaluated on the local validation set. The optimization metric is the validation loss:

$$
m_{\text{utility}} = \mathcal{L}_{\text{validation}}(w_{\text{DP}}).
$$

Cross-entropy loss will be used for classification experiments; alternative losses may be substituted depending on the task.

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; \mathcal{L}_{\text{validation}} .
$$

**Interpretation.** Lower validation loss indicates:

- better predictive accuracy,
- improved generalization,
- a better trade-off between the information carried by the update and the noise added for privacy.

Unlike the stability metric, this objective directly targets the model's performance on held-out local data.

## 8. Server Aggregation

The server performs weighted aggregation to mitigate the impact of potentially harmful or extremely large updates.

For every received noisy update $\tilde{\Delta}_i$:

1. Compute its L2 norm: $r_i = \| \tilde{\Delta}_i \|_2$.
2. Compute a robust baseline as the median of all received norms: $b = \operatorname{median}(\{r_i\})$.
3. Assign an attenuation weight: $w_i = \min\left(1, \frac{b}{r_i}\right)$.

The aggregated update is

$$
\tilde{\Delta}_{\text{agg}} = \frac{\sum_i w_i \tilde{\Delta}_i}{\sum_i w_i},
$$

and the global model is updated using

$$
w_g \leftarrow w_g + \eta \, \tilde{\Delta}_{\text{agg}},
$$

where $\eta$ is the server learning rate. This attenuation scheme is a form of median-based robust aggregation: updates with norm exceeding the median are down-weighted, reducing the influence of potential outliers.

## 9. Complete Client Algorithm

**Initialization (per client):**

- Privacy bounds: $\varepsilon_{\min}$, $\varepsilon_{\max}$, target $\delta$, total budget $\varepsilon_{\text{budget}}$.
- Clipping norm: $C$.
- Warm-up length: $L$ (number of exploration rounds).
- Acquisition penalty: $\lambda_{\text{aq}}$.
- Observation history: empty list.
- RDP accountant: initialized.
- Phase: Warm-up.

Training runs for a fixed number of $T$ communication rounds ($t = 1, 2, \dots, T$). The client may stop early if its privacy budget is exhausted. For each round:

1. Receive the current global model $w_g^{(t)}$.
2. If phase = Warm-up:
   - Select $\varepsilon_t$ from the predefined exploration sequence (grid, Latin hypercube, etc.).
   - Check budget: if $\varepsilon_t$ exceeds remaining budget, reduce it to the maximum allowed value (clamp).
   - Train locally for $E$ epochs $\rightarrow$ obtain $\Delta_t$.
   - Clip and noise $\Delta_t$ using $\varepsilon_t$ $\rightarrow$ $\tilde{\Delta}_t$.
   - Send $\tilde{\Delta}_t$ to the server.
   - Compute the chosen optimization metric $m_t$.
   - Store $(\varepsilon_t, m_t)$ in the observation history.
   - If the history now contains $L$ observations:
     - Fit the initial Gaussian Process (kernel, learn noise variance).
     - Set phase = BO.
3. Else (phase = BO):
   - Maximize the acquisition function $\alpha(\varepsilon)$ over $[\varepsilon_{\min}, \varepsilon_{\max}]$ using the current GP $\rightarrow$ candidate $\varepsilon^*$.
   - Verify budget: if $\varepsilon^*$ exceeds remaining budget, reduce (e.g., via binary search) to the largest feasible $\varepsilon \le \varepsilon^*$.
   - Set $\varepsilon_t$ to the (possibly reduced) value.
   - Train locally $\rightarrow$ $\Delta_t$.
   - Clip and noise using $\varepsilon_t$ $\rightarrow$ $\tilde{\Delta}_t$.
   - Send $\tilde{\Delta}_t$ to the server.
   - Compute the chosen optimization metric $m_t$.
   - Append $(\varepsilon_t, m_t)$ to the history.
   - Update the Gaussian Process with the extended history (refit or incremental update).
4. Update the RDP accountant with the cost of $\varepsilon_t$.
5. If the remaining budget cannot support any positive $\varepsilon$, cease participation.

The Bayesian Optimization phase continues for all subsequent rounds, providing an adaptive, personalized privacy schedule that reacts to the client's evolving data and optimization dynamics.

## 10. Experimental Variants

To isolate the effect of the optimization objective, the implementation should support the following configurations while keeping all other components (privacy mechanism, BO procedure, GP model, warm-up strategy, aggregation) identical:

| Variant | BO Objective | Purpose |
|---|---|---|
| PLDP-BO-Stability | $m_{\text{stab}} = \|\tilde{\Delta}\|_2$ | Minimize client drift and improve optimization stability under privacy noise. |
| PLDP-BO-Utility | $m_{\text{utility}} = \mathcal{L}_{\text{validation}}(w_{\text{DP}})$ | Maximize predictive performance for a given privacy budget. |

This design enables a direct comparison of how different optimization objectives influence the learned privacy schedules, convergence behavior, and final model performance, while demonstrating that PLDP-BO itself is a general framework whose optimization criterion can be changed without altering its privacy mechanism, Bayesian optimization procedure, or federated learning workflow.
