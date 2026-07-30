# PLDP-BO - Algorithm Description

**Version:** 0.3 (Implemented)

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
    - [Variant A: Noisy Update Norm (NUN)](#variant-a-noisy-update-norm-nun)
    - [Variant B: Model Utility Metric](#variant-b-model-utility-metric)
    - [Variant C: Utility Retention](#variant-c-utility-retention)
    - [Variant D: Utility Efficiency](#variant-d-utility-efficiency)
    - [Variant E: Utility per Remaining Budget](#variant-e-utility-per-remaining-budget)
    - [Variant F: Signal-to-Noise Ratio (SNR)](#variant-f-signal-to-noise-ratio-snr)
    - [Variant G: Logit Agreement](#variant-g-logit-agreement)
 8. [Server Aggregation](#8-server-aggregation)
9. [Complete Client Algorithm](#9-complete-client-algorithm)
10. [Experimental Variants](#10-experimental-variants)

## 1. Overview

PLDP-BO is a Federated Learning (FL) algorithm that continuously personalizes the Local Differential Privacy (LDP) parameter $\varepsilon$ of each participating client throughout the training process. Each client independently runs a Bayesian Optimization (BO) process, preceded by a systematic warm-up exploration phase, to search for the most suitable privacy parameter according to a locally computed objective. The optimization is performed under a hard Rényi Differential Privacy (RDP) budget, ensuring that the cumulative privacy loss never exceeds a predefined limit.

The framework is metric-agnostic: BO only requires a scalar objective to optimize. Consequently, different optimization goals can be adopted without changing the remainder of the algorithm. The implementation supports seven objective variants:

1. Noisy Update Norm (NUN)
2. Model Utility (Local validation loss)
3. Utility Retention
4. Utility Efficiency
5. Utility per Remaining Budget
6. Signal-to-Noise Ratio (SNR)
7. Logit Agreement

The remainder of the algorithm is identical for all variants.

## 2. Notation

| Symbol | Description |
|---|---|
| $\varepsilon$ | Privacy parameter (Local DP) |
| $\varepsilon_{\min}, \varepsilon_{\max}$ | Bounds of the privacy search interval |
| $\varepsilon_{\text{budget}}$ | Total $(\varepsilon,\delta)$-DP budget per client |
| $\varepsilon_{\text{candidate}}$ | Candidate $\varepsilon$ before budget verification |
| $\delta$ | Target delta parameter for DP |
| $\alpha$ | RDP order; evaluated over $\alpha \in \{2, 3, \dots, 100\}$ |
| $C$ | L2 clipping norm |
| $L$ | Number of warm-up exploration rounds |
| $G$ | Number of grid points for EI normalization |
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
| $m_{\text{nun}}$ | Noisy Update Norm (NUN) metric |
| $m_{\text{utility}}$ | Utility optimization metric |
| $b$ | Median of all received update norms (server) |
| $r_i$ | L2 norm of client $i$'s noisy update |
| $w_i$ | Attenuation weight for client $i$'s update |
| $\mathcal{GP}(\mu, k)$ | Gaussian Process with mean function $\mu$ and kernel $k$ |
| $\text{EI}(\varepsilon)$ | Expected Improvement acquisition function |
| $\text{EI}_{\text{norm}}(\varepsilon)$ | Normalized Expected Improvement |
| $\alpha(\varepsilon)$ | Acquisition function with privacy penalty |
| $\varepsilon_{\text{remaining}}$ | Remaining privacy budget at a given round |
| $L_{\text{clean}}$ | Validation loss of the clean (pre-DP) model |
| $L_{\text{noisy}}$ | Validation loss of the noisy (post-DP) model (same as $m_{\text{utility}}$) |
| $m_{\text{snr}}$ | Signal-to-Noise Ratio (SNR) metric |
| $m_{\text{ret}}$ | Utility Retention metric |
| $m_{\text{eff}}$ | Utility Efficiency metric |
| $m_{\text{rem}}$ | Utility per Remaining Budget metric |
| $m_{\text{agr}}$ | Logit Agreement metric |
| $\cos(\cdot, \cdot)$ | Cosine similarity |

## 3. Federated Learning Workflow

Each communication round proceeds as follows.

### Client

1. Receive the current global model $w_g^{(t)}$.
2. Select the privacy parameter $\varepsilon_t$ according to the current phase:
   - Warm-up phase: use the predefined deterministic exploration strategy.
   - BO phase: maximize the acquisition function (see Section 6).
3. Verify that using $\varepsilon_t$ does not exceed the client's remaining privacy budget.
   If the budget is violated, reduce $\varepsilon_t$ to the largest $\varepsilon \le \varepsilon_t$ that satisfies the budget (see Section 5).
4. Train locally for $E$ epochs.
5. Compute the local update $\Delta_t = w_{\text{local}}^{(t)} - w_g^{(t)}$.
6. Clip the update $\hat{\Delta}_t = \Delta_t \cdot \min\left(1, \frac{C}{\|\Delta_t\|_2}\right)$.
7. Generate Gaussian noise according to the selected privacy level $z_t \sim \mathcal{N}(0, \sigma_t^2 I)$.
8. Produce the DP update $\tilde{\Delta}_t = \hat{\Delta}_t + z_t$.
9. Send only the noisy update $\tilde{\Delta}_t$ to the server.
10. Compute the chosen optimization metric $m_t$ (see Section 7 for available variants).
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
\sigma = C \cdot \sqrt{\, 2 \log(1.25 / \delta) \,} \; / \; \varepsilon .
$$

The noisy update satisfies one-round Local Differential Privacy.

This is a **per-update** privacy mechanism: the client clips the model delta once and adds a single Gaussian noise vector per round. Unlike DP-SGD, which clips gradients and adds noise at every optimization step, PLDP-BO applies the Gaussian mechanism directly to the aggregated model update. This simplifies privacy accounting: each round corresponds to exactly one application of the Gaussian mechanism, with RDP cost $R_\alpha^{(t)} = \alpha C^2 / (2\sigma_t^2)$ (see Section 5).

## 5. Privacy Accounting

Each client maintains an independent Rényi Differential Privacy (RDP) accountant evaluated over a discrete set of RDP orders $\alpha \in \{2, 3, \dots, 100\}$.

The total privacy budget is specified as a target $(\varepsilon_{\text{budget}}, \delta)$-DP guarantee. The accountant tracks whether the cumulative privacy loss remains within this bound.

**Per-round RDP cost.** For a round with noise scale $\sigma_t$, the RDP cost at order $\alpha$ is

$$
R_\alpha^{(t)} = \frac{\alpha C^2}{2\sigma_t^2}.
$$

**Accumulation.** After $t$ rounds, the total RDP at order $\alpha$ is

$$
R_\alpha^{\text{(total)}} = \sum_{k=1}^{t} \frac{\alpha C^2}{2\sigma_k^2}.
$$

**Conversion to $(\varepsilon,\delta)$-DP.** The total privacy loss is (Mironov, 2017)

$$
\varepsilon_{\text{total}} = \min_{\alpha}
\left[ R_\alpha^{\text{(total)}} + \frac{\log(1/\delta)}{\alpha - 1} \right].
$$

**Budget enforcement (per round).**
For a candidate $\varepsilon_t$ (from warm-up or BO):

1. Compute $\sigma_t = C \cdot \sqrt{2 \log(1.25 / \delta)} \;/\; \varepsilon_t$.
2. Compute candidate RDP cost $R_\alpha^{(t)} = \alpha C^2 / (2\sigma_t^2)$.
3. Compute hypothetical total $R_\alpha^{\text{(new)}} = R_\alpha^{\text{(total)}} + R_\alpha^{(t)}$.
4. Convert to $\varepsilon_{\text{new}} = \min_\alpha [R_\alpha^{\text{(new)}} + \log(1/\delta)/(\alpha-1)]$.
5. If $\varepsilon_{\text{new}} \le \varepsilon_{\text{budget}}$, accept $\varepsilon_t$.
6. Otherwise, perform a binary search in $[\varepsilon_{\min}, \varepsilon_t]$ to find the largest $\varepsilon \le \varepsilon_t$ satisfying the budget (e.g., 30 iterations).

If the binary search reaches $\varepsilon_{\min}$ and it still violates the budget, the client's privacy budget is exhausted and it ceases participation.

**Accountant update.** After training with the (possibly reduced) $\varepsilon_t$ and its corresponding $\sigma_t$, the accountant records the cost:

$$
R_\alpha^{\text{(total)}} \leftarrow R_\alpha^{\text{(total)}} + \frac{\alpha C^2}{2\sigma_t^2}.
$$

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
2. Verify the privacy budget (see Section 5). If the selected $\varepsilon$ exceeds the remaining budget, reduce it via binary search in $[\varepsilon_{\min}, \varepsilon]$ to the largest feasible value.
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

The observation noise variance $\sigma_n^2$ is distinct from the DP mechanism noise variance $\sigma^2$ (Section 4); $\sigma_n^2$ is estimated during GP fitting and captures variability beyond the DP perturbation.

### 6.3 Continuous Bayesian Optimization

After warm-up, the client enters the optimization phase, which runs for every subsequent communication round until training ends.

**Acquisition function.** To select the next $\varepsilon$, the client maximizes an acquisition function that balances exploration and exploitation while incorporating a preference for stronger privacy:

$$
\alpha(\varepsilon) = \text{EI}_{\text{norm}}(\varepsilon) - \lambda_{\text{aq}} \cdot \frac{\varepsilon - \varepsilon_{\min}}{\varepsilon_{\max} - \varepsilon_{\min}},
$$

where:

- $\text{EI}_{\text{norm}}(\varepsilon)$ is the normalized Expected Improvement (see below),
- $\lambda_{\text{aq}} \ge 0$ controls the strength of the penalty that favors smaller $\varepsilon$ (stronger privacy),
- the normalization by $\varepsilon_{\max} - \varepsilon_{\min}$ ensures consistent behavior across different search intervals.

To decouple the penalty from the metric's scale, $\text{EI}(\varepsilon)$ is normalized to $[0, 1]$ by evaluating over a discrete grid $\{\varepsilon_1, \dots, \varepsilon_G\} \subset [\varepsilon_{\min}, \varepsilon_{\max}]$:

$$
\text{EI}_{\text{norm}}(\varepsilon) = \frac{\text{EI}(\varepsilon) - \text{EI}_{\min}}{\text{EI}_{\max} - \text{EI}_{\min}},
$$

where $\text{EI}_{\min} = \min_j \text{EI}(\varepsilon_j)$ and $\text{EI}_{\max} = \max_j \text{EI}(\varepsilon_j)$. The acquisition function uses this normalized form (as defined above).

Both terms now lie in $[0, 1]$, so $\lambda_{\text{aq}}$ directly controls their relative weight regardless of the objective metric's magnitude.

If $\text{EI}_{\max} = \text{EI}_{\min}$ (e.g., when the GP sees insufficient variation in the data, such as very early in training), $\text{EI}_{\text{norm}}$ is undefined. In this degenerate case the acquisition function reduces to $\alpha(\varepsilon) = -\lambda_{\text{aq}} \cdot (\varepsilon - \varepsilon_{\min})/(\varepsilon_{\max} - \varepsilon_{\min})$, which defaults to selecting $\varepsilon_{\min}$ (strongest privacy).

**BO cycle.** For each round after warm-up:

1. Use the current Gaussian Process to model $f(\varepsilon)$.
1a. Evaluate $\text{EI}(\varepsilon)$ over a fine grid in $[\varepsilon_{\min}, \varepsilon_{\max}]$ to compute the normalizing constants $\text{EI}_{\min}$ and $\text{EI}_{\max}$, then construct $\text{EI}_{\text{norm}}(\varepsilon)$ and $\alpha(\varepsilon)$.
2. Maximize the acquisition function $\alpha(\varepsilon)$ over $[\varepsilon_{\min}, \varepsilon_{\max}]$ to obtain a candidate $\varepsilon^*$.
3. Verify the RDP budget (see Section 5): if $\varepsilon^*$ would exceed the remaining budget, reduce it via binary search in $[\varepsilon_{\min}, \varepsilon^*]$ until the constraint is satisfied.
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

The BO framework is independent of the objective function. The implementation supports seven metric variants, all of which are scalar functions of $\varepsilon$ and the training outcome. Variants A and B require a single validation evaluation after DP perturbation. Variants C through G additionally require a clean (pre-DP) validation evaluation to compute reference loss and logits.

### Variant A: Noisy Update Norm (NUN)

**Motivation.** The Noisy Update Norm (NUN) measures the magnitude of the private update sent to the server. This metric captures the joint effect of the client's learning signal and the privacy noise. Weaker privacy (larger $\varepsilon$) adds less noise and preserves more signal, resulting in a smaller update norm. Stronger privacy (smaller $\varepsilon$) injects more noise, increasing the norm. By minimizing $m_{\text{nun}}$, the BO seeks a privacy level where the signal component dominates the noise.

**Metric.** After local training and the addition of Gaussian noise, the client constructs the private update $\tilde{\Delta}_t$. The NUN metric is defined as

$$
m_{\text{nun}} = \| \tilde{\Delta}_t \|_2 .
$$

Equivalently, $m_{\text{nun}} = \| w_{\text{DP}} - w_g \|_2$, where $w_{\text{DP}} = w_g + \tilde{\Delta}_t$ is the noisy local model.

**Expected norm.** For a model of dimension $d$, the expected squared norm is

$$
\mathbb{E}[\, m_{\text{nun}}^2 \,] = \| \hat{\Delta}_t \|_2^2 + d \cdot \sigma_t^2,
$$

where $\sigma_t = C \cdot \sqrt{2 \log(1.25 / \delta)} / \varepsilon_t$ is the noise scale. The noise contribution $d \cdot \sigma_t^2$ grows as $\varepsilon_t$ decreases, making the expected NUN larger under stronger privacy.

By Jensen's inequality (concavity of $\sqrt{\cdot}$), the expected norm satisfies

$$
\mathbb{E}[\, m_{\text{nun}} \,] \le \sqrt{\, \mathbb{E}[\, m_{\text{nun}}^2 \,] \,}
     = \sqrt{\, \| \hat{\Delta}_t \|_2^2 + d \cdot \sigma_t^2 \,},
$$

with strict inequality whenever the metric is non-deterministic. This bound is useful for interpreting the metric's typical magnitude.

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; m_{\text{nun}} .
$$

**Interpretation.** Lower values indicate that the released update is dominated by the learning signal rather than by privacy noise. This occurs when $\varepsilon_t$ is sufficiently large that $d \cdot \sigma_t^2$ is negligible relative to $\| \hat{\Delta}_t \|_2^2$.

The BO naturally drives $m_{\text{nun}}$ downward by favoring larger $\varepsilon$ (weaker privacy). The acquisition function's penalty term $\lambda_{\text{aq}}$ counterbalances this by penalizing large $\varepsilon$, creating a principled trade-off between update fidelity and privacy strength. Because $\text{EI}_{\text{norm}}$ and the penalty are both normalized to $[0, 1]$, $\lambda_{\text{aq}}$ directly represents the relative weight of privacy versus update fidelity and will be tuned via grid search.

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

Unlike the NUN metric, this objective directly targets the model's performance on held-out local data.

### Variant C: Utility Retention

**Motivation.** Utility Retention measures how much the DP perturbation degrades the model's validation performance relative to the clean (unperturbed) model. A value close to 1 indicates that privacy noise did not harm accuracy; larger values indicate degradation. This metric isolates the cost of privacy from the absolute loss, making it comparable across different loss scales.

**Metric.** The client computes the validation loss before and after adding DP noise. The utility retention is the ratio of noisy to clean loss:

$$
m_{\text{ret}} = \frac{L_{\text{noisy}}}{L_{\text{clean}}}.
$$

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; m_{\text{ret}} .
$$

**Interpretation.** Values near 1.0 indicate that the DP perturbation introduced negligible degradation. Values above 1.0 indicate increasing loss due to noise. By minimizing $m_{\text{ret}}$, BO seeks privacy levels where the noisy model retains the clean model's predictive performance. Unlike the NUN metric, this objective directly captures utility rather than update magnitude.

### Variant D: Utility Efficiency

**Motivation.** Utility Efficiency measures the fractional loss increase per unit of privacy budget spent. This captures how efficiently each unit of $\varepsilon$ is used: a small loss increase per $\varepsilon$ means the client obtains strong privacy at little cost to model quality. The denominator $\varepsilon$ intrinsically penalizes larger $\varepsilon$, encoding a privacy-utility trade-off without requiring the acquisition penalty $\lambda_{\text{aq}}$.

**Metric.** The client computes the loss degradation due to DP noise and normalizes it by the clean loss and the privacy parameter:

$$
m_{\text{eff}} = -\frac{\max(0, L_{\text{noisy}} - L_{\text{clean}})}{L_{\text{clean}} \cdot \varepsilon}.
$$

The numerator is zero when the noisy loss is not larger than the clean loss, making $m_{\text{eff}}$ non-positive.

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; m_{\text{eff}} .
$$

**Interpretation.** Because $m_{\text{eff}}$ is negative (or zero), minimizing it drives toward more negative values, corresponding to lower fractional degradation per unit $\varepsilon$. The BO naturally favors combinations of small $\varepsilon$ (strong privacy) and low degradation. The privacy preference arises intrinsically from the $\varepsilon$ denominator, unlike NUN and Utility which rely on the explicit penalty $\lambda_{\text{aq}}$.

### Variant E: Utility per Remaining Budget

**Motivation.** Utility per Remaining Budget extends the efficiency concept by normalizing degradation by the remaining privacy budget rather than the per-round epsilon. This makes the metric budget-aware: when little budget remains, the client is penalized more heavily for wasteful spending. The goal is to allocate limited budget across the remaining rounds as efficiently as possible.

**Metric.** The metric is identical to utility efficiency but uses the remaining budget instead of the per-round epsilon:

$$
m_{\text{rem}} = -\frac{\max(0, L_{\text{noisy}} - L_{\text{clean}})}{L_{\text{clean}} \cdot \varepsilon_{\text{remaining}}},
$$

where $\varepsilon_{\text{remaining}}$ is the client's remaining privacy budget for future rounds.

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; m_{\text{rem}} .
$$

**Interpretation.** As the remaining budget shrinks, the denominator shrinks, making the metric more sensitive to any loss degradation. This encourages the BO to select more conservative $\varepsilon$ values when budget is scarce, and to explore more freely when budget is abundant.

### Variant F: Signal-to-Noise Ratio (SNR)

**Motivation.** The Signal-to-Noise Ratio (SNR) measures the relative power of the learning signal to the DP noise before perturbation. A high SNR means the update is large relative to the noise scale, indicating that the signal is likely to survive the noise addition. A low SNR means the update will be dominated by noise. By minimizing SNR, the BO drives the system toward a noise-dominated regime, corresponding to stronger privacy.

**Metric.** The client computes the squared L2 norm of the local model update relative to the noise variance:

$$
m_{\text{snr}} = \frac{\|\Delta_t\|_2^2}{\sigma_t^2}.
$$

Since $\sigma_t \propto 1/\varepsilon_t$, SNR grows quadratically with $\varepsilon$: larger $\varepsilon$ yields less noise, hence higher SNR.

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; m_{\text{snr}} .
$$

**Interpretation.** Larger $\varepsilon$ (weaker privacy) produces less noise and therefore higher SNR. By minimizing SNR, the BO naturally favors smaller $\varepsilon$ (stronger privacy), providing an intrinsic privacy preference without relying on the acquisition penalty $\lambda_{\text{aq}}$.

### Variant G: Logit Agreement

**Motivation.** Logit Agreement measures how much the model's predictions (in logit space) change after DP perturbation. Small changes mean the noisy model produces similar class scores to the clean model, indicating that privacy noise did not distort the learned representations. This metric captures prediction-level stability rather than just loss magnitude.

**Metric.** The client computes the average cosine similarity between clean and noisy logit vectors across all validation samples, then defines agreement as the complement:

$$
m_{\text{agr}} = 1 - \frac{1}{N} \sum_{i=1}^N \cos\bigl(z_i^{\text{(clean)}}, z_i^{\text{(noisy)}}\bigr),
$$

where $z_i^{\text{(clean)}}$ and $z_i^{\text{(noisy)}}$ are the logit vectors for validation sample $i$ from the clean and noisy models respectively, and $\cos(\cdot, \cdot)$ is the cosine similarity. The range is $[0, 2]$, where 0 indicates identical logit directions and 2 indicates opposite directions.

**Optimization Objective.** BO minimizes

$$
\min_\varepsilon \; m_{\text{agr}} .
$$

**Interpretation.** Lower values indicate better agreement between the clean and noisy model predictions. A value near 0 means the DP perturbation did not meaningfully alter the model's output distribution. By minimizing agreement, BO seeks privacy levels where the model's predictions remain stable despite the added noise.

## 8. Server Aggregation

The server performs weighted aggregation to mitigate the impact of potentially harmful or extremely large updates.

For every received noisy update $\tilde{\Delta}_i$:

1. Compute its L2 norm: $r_i = \| \tilde{\Delta}_i \|_2$.
2. Compute a robust baseline as the median of all received norms: $b = \mathrm{median}(\{r_i\})$.
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

The client does **not** send its chosen $\varepsilon_t$ to the server. Attenuation weights are computed solely from the observed norms $r_i = \|\tilde{\Delta}_i\|_2$. Clients with stronger privacy (smaller $\varepsilon$) naturally produce noisier updates with larger norms, resulting in lower weights. This implicit mechanism avoids leaking the client's privacy preference.

## 9. Complete Client Algorithm

**Initialization (per client):**

- Privacy bounds: $\varepsilon_{\min}$, $\varepsilon_{\max}$, target $\delta$, total budget $\varepsilon_{\text{budget}}$.
- Clipping norm: $C$.
- Warm-up length: $L$ (number of exploration rounds).
- Acquisition penalty: $\lambda_{\text{aq}}$.
- Observation history: empty list.
- RDP accountant: initialized.
- Phase: Warm-up.

Training runs for a fixed number of $T$ communication rounds ($t = 1, 2, \dots, T$). The server initializes $w_g$ (e.g., randomly) before round 1. The client may stop early if its privacy budget is exhausted. For each round:

1. Receive the current global model $w_g^{(t)}$.
2. If phase = Warm-up:
   - Select $\varepsilon_t$ from the predefined exploration sequence (grid, Latin hypercube, etc.).
   - Check budget (see Section 5): if $\varepsilon_t$ exceeds remaining budget, reduce it via binary search in $[\varepsilon_{\min}, \varepsilon_t]$ to the largest feasible value.
   - Train locally for $E$ epochs $\rightarrow$ obtain $\Delta_t$.
   - Clip and noise $\Delta_t$ using $\varepsilon_t$ $\rightarrow$ $\tilde{\Delta}_t$.
   - Send $\tilde{\Delta}_t$ to the server.
   - Compute the chosen optimization metric $m_t$.
   - Store $(\varepsilon_t, m_t)$ in the observation history.
   - If the history now contains $L$ observations:
     - Fit the initial Gaussian Process (kernel, learn noise variance).
     - Set phase = BO.
3. Else (phase = BO):
   - Construct $\text{EI}_{\text{norm}}$ via grid evaluation (Section 6.3) and maximize $\alpha(\varepsilon)$ over $[\varepsilon_{\min}, \varepsilon_{\max}]$ $\rightarrow$ candidate $\varepsilon^*$.
   - Verify budget (see Section 5): if $\varepsilon^*$ exceeds remaining budget, reduce it via binary search in $[\varepsilon_{\min}, \varepsilon^*]$ to the largest feasible $\varepsilon \le \varepsilon^*$.
   - Set $\varepsilon_t$ to the (possibly reduced) value.
   - Train locally $\rightarrow$ $\Delta_t$.
   - Clip and noise using $\varepsilon_t$ $\rightarrow$ $\tilde{\Delta}_t$.
   - Send $\tilde{\Delta}_t$ to the server.
   - Compute the chosen optimization metric $m_t$.
   - Append $(\varepsilon_t, m_t)$ to the history.
   - Update the Gaussian Process with the extended history (refit or incremental update).
4. Update the RDP accountant (Section 5) with the cost of the $\varepsilon_t$ used: $R_\alpha^{\text{(total)}} \leftarrow R_\alpha^{\text{(total)}} + \alpha C^2 / (2\sigma_t^2)$.
5. If the remaining budget cannot support any positive $\varepsilon$, cease participation.

The Bayesian Optimization phase continues for all subsequent rounds, providing an adaptive, personalized privacy schedule that reacts to the client's evolving data and optimization dynamics.

## 10. Experimental Variants

To isolate the effect of the optimization objective, the implementation should support the following configurations while keeping all other components (privacy mechanism, BO procedure, GP model, warm-up strategy, aggregation) identical:

| Variant | BO Objective | Purpose |
|---|---|---|
| PLDP-BO-NUN | $m_{\text{nun}} = \lVert\tilde{\Delta}\rVert_2$ | Minimize the noisy update norm, balancing signal preservation against privacy noise. |
| PLDP-BO-Utility | $m_{\text{utility}} = \mathcal{L}_{\text{validation}}(w_{\text{DP}})$ | Maximize predictive performance for a given privacy budget. |
| PLDP-BO-UtilityRetention | $m_{\text{ret}} = L_{\text{noisy}} / L_{\text{clean}}$ | Minimize the noisy-to-clean loss ratio, preserving validation performance under DP. |
| PLDP-BO-UtilityEfficiency | $m_{\text{eff}}$ | Minimize fractional loss increase per unit $\varepsilon$, encoding an intrinsic privacy preference. |
| PLDP-BO-UtilityPerRemaining | $m_{\text{rem}}$ | Minimize fractional loss increase per unit remaining budget, adapting to budget scarcity. |
| PLDP-BO-SNR | $m_{\text{snr}} = \lVert\Delta_t\rVert_2^2 / \sigma_t^2$ | Minimize signal-to-noise ratio, driving toward a noise-dominated privacy regime. |
| PLDP-BO-Agreement | $m_{\text{agr}}$ | Minimize logit cosine dissimilarity, preserving prediction stability under DP. |

This design enables a direct comparison of how different optimization objectives influence the learned privacy schedules, convergence behavior, and final model performance, while demonstrating that PLDP-BO itself is a general framework whose optimization criterion can be changed without altering its privacy mechanism, Bayesian optimization procedure, or federated learning workflow.
