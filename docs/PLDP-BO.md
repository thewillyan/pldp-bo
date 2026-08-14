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

PLDP-BO is a Federated Learning (FL) algorithm that continuously personalizes the privacy budget of each participating client throughout the training process. Each client independently runs a Bayesian Optimization (BO) process, preceded by a systematic warm-up exploration phase, to search for the most suitable privacy parameter according to a locally computed objective. The optimization is performed under a hard Rényi Differential Privacy (RDP) budget, ensuring that the cumulative privacy loss never exceeds a predefined limit.

The privacy mechanism uses per-example differential privacy (DP-SGD style), where each client clips individual example gradients and adds calibrated Gaussian noise during local training. This is the most widely used approach for training deep learning models with differential privacy. Privacy accounting is performed directly in RDP space without conversion to $(\varepsilon,\delta)$-DP.

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
| $R$ | RDP cost (privacy parameter) |
| $R_{\min}, R_{\max}$ | Bounds of the RDP cost search interval |
| $B_{\text{RDP}}$ | Total RDP budget per client |
| $\alpha_0$ | Fixed RDP order (default 10.0) |
| $C$ | Per-example gradient clipping norm |
| $q$ | Subsampling rate ($\text{batch\_size} / \text{dataset\_size}$) |
| $\delta$ | Target delta parameter for sigma calibration |
| $L$ | Number of warm-up exploration rounds |
| $G$ | Number of grid points for EI normalization |
| $\lambda_{aq}$ | Privacy penalty coefficient in the acquisition function |
| $E$ | Number of local training epochs |
| $T$ | Total number of communication rounds |
| $\eta$ | Server learning rate |
| $w_g$ | Global model parameters |
| $w_{local}$ | Local model parameters after training |
| $\Delta_t$ | Local update at round $t$ |
| $\tilde{\Delta}_{agg}$ | Aggregated update at the server |
| $z_t$ | Gaussian noise vector at round $t$ |
| $\sigma_t^2$ | Noise variance at round $t$ |
| $\sigma$ | Noise standard deviation (DP mechanism) |
| $\sigma_n^2$ | Observation noise variance (GP model) |
| $m_{nun}$ | Noisy Update Norm (NUN) metric |
| $m_{utility}$ | Utility optimization metric |
| $b$ | Median of all received update norms (server) |
| $r_i$ | L2 norm of client $i$'s update |
| $w_i$ | Attenuation weight for client $i$'s update |
| $\mathcal{GP}(\mu, k)$ | Gaussian Process with mean function $\mu$ and kernel $k$ |
| $EI(R)$ | Expected Improvement acquisition function |
| $EI_{norm}(R)$ | Normalized Expected Improvement |
| $\alpha(R)$ | Acquisition function with privacy penalty |
| $R_{\text{remaining}}$ | Remaining RDP budget at a given round |
| $L_{clean}$ | Validation loss of the clean (pre-DP) model |
| $L_{noisy}$ | Validation loss of the noisy (post-DP) model (same as $m_{utility}$) |
| $m_{snr}$ | Signal-to-Noise Ratio (SNR) metric |
| $m_{ret}$ | Utility Retention metric |
| $m_{eff}$ | Utility Efficiency metric |
| $m_{rem}$ | Utility per Remaining Budget metric |
| $m_{agr}$ | Logit Agreement metric |
| $\cos(\cdot, \cdot)$ | Cosine similarity |

## 3. Federated Learning Workflow

Each communication round proceeds as follows.

### Client

1. Receive the current global model $w_g^{(t)}$.
2. Select the RDP cost $R_t$ according to the current phase:
   - Warm-up phase: use the predefined deterministic exploration strategy.
   - BO phase: maximize the acquisition function (see Section 6).
3. Verify that using $R_t$ does not exceed the client's remaining privacy budget.
   If the budget is violated, reduce $R_t$ to the largest value satisfying the budget (see Section 5).
4. Calibrate noise scale: $\sigma_t = \sqrt{\alpha_0 \cdot q^2 / (2 \cdot R_t)}$.
5. Train locally for $E$ epochs using DP-SGD:
   - For each mini-batch:
     a. Compute per-example gradients.
     b. Clip each gradient to L2 norm bounded by $C$.
     c. Average the clipped gradients.
     d. Add Gaussian noise $z \sim \mathcal{N}(0, \sigma_t^2 I)$ to the averaged gradients.
     e. Update model parameters using the noisy gradients.
6. Compute the local update $\Delta_t = w_{local}^{(t)} - w_g^{(t)}$.
7. Send the local update $\Delta_t$ to the server.
8. Compute the chosen optimization metric $m_t$ (see Section 7 for available variants).
9. Store the observation $(R_t, m_t)$.
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

The algorithm uses a per-example DP-SGD mechanism. For each mini-batch of training data, the client:

1. Computes gradients for each individual training example.
2. Clips each example's gradient to L2 norm bounded by $C$.
3. Averages the clipped gradients.
4. Adds calibrated Gaussian noise to the averaged gradients.

The noise is sampled as

$$
z \sim \mathcal{N}(0, \sigma^2 I),
$$

where $\sigma$ is calibrated to achieve the target RDP guarantee (see Section 5).

Each client uses Poisson subsampling with sampling rate $q = \text{batch\_size} / \text{dataset\_size}$. The per-example clipping ensures that each example contributes at most $C$ to the sensitivity, which is the standard approach in DP-SGD (Abadi et al., 2016). This mechanism satisfies RDP at order $\alpha$ with cost

$$
R_\alpha^{(t)} = \frac{\alpha \cdot q^2}{2 \sigma_t^2}.
$$

The per-example approach is the most widely used method for training deep learning models with differential privacy, as it provides tight accounting and strong empirical performance.

## 5. Privacy Accounting

Each client maintains an independent Rényi Differential Privacy (RDP) accountant. The privacy budget is specified directly in RDP space, without conversion to $(\varepsilon,\delta)$-DP.

**RDP order.** A fixed RDP order $\alpha_0$ is selected (configurable via `privacy.rdp_alpha`, default 10.0). All privacy accounting is performed at this order.

**Per-round RDP cost.** For a round with noise scale $\sigma_t$ and subsampling rate $q$, the RDP cost at order $\alpha_0$ is

$$
R_{\alpha_0}^{(t)} = \frac{\alpha_0 \cdot q^2}{2\sigma_t^2}.
$$

**Accumulation.** After $t$ rounds, the total RDP at order $\alpha_0$ is

$$
R_{\alpha_0}^{\text{(total)}} = \sum_{k=1}^{t} \frac{\alpha_0 \cdot q^2}{2\sigma_k^2}.
$$

**Sigma calibration.** For a target RDP cost $R_{\text{target}}$, the required noise scale is

$$
\sigma = \sqrt{\frac{\alpha_0 \cdot q^2}{2 \cdot R_{\text{target}}}}.
$$

**Budget enforcement (per round).**
For a candidate RDP cost $R_{\text{candidate}}$ (from warm-up or BO):

1. Compute hypothetical total $R_{\alpha_0}^{(new)} = R_{\alpha_0}^{(total)} + R_{\text{candidate}}$.
2. If $R_{\alpha_0}^{(new)} \le B_{\text{RDP}}$ (the total RDP budget), accept the candidate.
3. Otherwise, perform a binary search in $[R_{\min}, R_{\text{candidate}}]$ to find the largest RDP cost satisfying the budget (e.g., 30 iterations).
4. Convert the accepted RDP cost to noise scale: $\sigma_t = \sqrt{\alpha_0 \cdot q^2 / (2 \cdot R_{\text{accepted}})}$.

If the binary search reaches $R_{\min}$ and it still violates the budget, the client's privacy budget is exhausted and it ceases participation.

**Accountant update.** After training with the (possibly reduced) RDP cost $R_{\text{accepted}}$ and its corresponding $\sigma_t$, the accountant records the cost:

$$
R_{\alpha_0}^{\text{(total)}} \leftarrow R_{\alpha_0}^{\text{(total)}} + R_{\text{accepted}}.
$$

**Configuration:**

```yaml
privacy:
  enabled: true
  rdp_alpha: 10.0           # fixed RDP order

bo:
  enabled: true
  rdp_min: 0.01             # min RDP cost for BO search
  rdp_max: 2.0              # max RDP cost for BO search
  total_budget: 10.0        # total RDP(alpha_0) budget per client
```

**Advantages of RDP-native accounting:**

- No imprecise RDP-to-epsilon conversion.
- Direct sigma calibration.
- Well-suited for research comparing RDP-based privacy guarantees.

## 6. Bayesian Optimization

Each client maintains an independent Bayesian Optimization (BO) process throughout federated training.

The admissible RDP cost range is defined as

$$
R \in [R_{\min}, R_{\max}],
$$

where different clients may use different bounds.

The optimization consists of two phases:

- **Warm-up (initial exploration):** the client deliberately evaluates several different $R$ values to collect an initial dataset for the Gaussian Process.
- **Continuous optimization:** after warm-up, the client uses Bayesian Optimization to select $R$ for each communication round while continuously updating the Gaussian Process with new observations.

Bayesian Optimization does not terminate after warm-up; it continues adapting the client's privacy level during the entire training process.

### 6.1 Warm-up Phase

During the first $L$ communication rounds, the client explores the RDP cost space.

The exploration strategy may use:

- a predefined grid,
- evenly spaced $R$ values,
- Latin Hypercube sampling,
- or another deterministic sampling strategy.

The objective is to collect representative observations covering the interval $[R_{\min}, R_{\max}]$.

For each warm-up round:

1. Select the next exploration $R$ according to the chosen strategy.
2. Verify the privacy budget (see Section 5). If the selected $R$ exceeds the remaining budget, reduce it via binary search in $[R_{\min}, R]$ to the largest feasible value.
3. Train locally (Section 3, steps 4-7).
4. Compute the selected optimization metric $m$.
5. Store the observation $(R, m)$ in the observation history.

After $L$ observations have been collected, the initial Gaussian Process is fitted. This provides BO with enough information to make informed decisions instead of beginning from an almost uninformative prior.

### 6.2 Gaussian Process Model

The unknown objective function is

$$
f(R).
$$

The client models it as

$$
f \sim \mathcal{GP}(\mu, k).
$$

A Matérn 5/2 kernel or RBF kernel may be used.

The GP explicitly models noisy observations:

$$
m_i = f(R_i) + \eta_i,
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

**Acquisition function.** To select the next RDP cost $R$, the client maximizes an acquisition function that balances exploration and exploitation while incorporating a preference for stronger privacy:

$$
\alpha(R) = \text{EI}_{\text{norm}}(R) - \lambda_{\text{aq}} \cdot \frac{R - R_{\min}}{R_{\max} - R_{\min}},
$$

where:

- $EI_{norm}(R)$ is the normalized Expected Improvement (see below),
- $\lambda_{aq} \ge 0$ controls the strength of the penalty that favors smaller $R$ (stronger privacy),
- the normalization by $R_{\max} - R_{\min}$ ensures consistent behavior across different search intervals.

To decouple the penalty from the metric's scale, $EI(R)$ is normalized to $[0, 1]$ by evaluating over a discrete grid $\{R_1, \dots, R_G\} \subset [R_{\min}, R_{\max}]$:

$$
\text{EI}_{\text{norm}}(R) = \frac{\text{EI}(R) - \text{EI}_{\min}}{\text{EI}_{\max} - \text{EI}_{\min}},
$$

where $EI_{\min} = \min_j EI(R_j)$ and $EI_{\max} = \max_j EI(R_j)$. The acquisition function uses this normalized form (as defined above).

Both terms now lie in $[0, 1]$, so $\lambda_{aq}$ directly controls their relative weight regardless of the objective metric's magnitude.

If $EI_{\max} = EI_{\min}$ (e.g., when the GP sees insufficient variation in the data, such as very early in training), $EI_{norm}$ is undefined. In this degenerate case the acquisition function reduces to $\alpha(R) = -\lambda_{aq} \cdot (R - R_{\min})/(R_{\max} - R_{\min})$, which defaults to selecting $R_{\min}$ (strongest privacy).

**BO cycle.** For each round after warm-up:

1. Use the current Gaussian Process to model $f(R)$.
1a. Evaluate $EI(R)$ over a fine grid in $[R_{\min}, R_{\max}]$ to compute the normalizing constants $EI_{\min}$ and $EI_{\max}$, then construct $EI_{norm}(R)$ and $\alpha(R)$.
2. Maximize the acquisition function $\alpha(R)$ over $[R_{\min}, R_{\max}]$ to obtain a candidate $R^*$.
3. Verify the RDP budget (see Section 5): if $R^*$ would exceed the remaining budget, reduce it via binary search in $[R_{\min}, R^*]$ until the constraint is satisfied.
4. Perform local training using the (possibly reduced) RDP cost $R$.
5. Compute the optimization metric $m$.
6. Append $(R, m)$ to the observation history.
7. Update the Gaussian Process with the augmented dataset (refit or incremental update).
8. The updated GP is used to select $R$ for the next communication round.

Thus, each client learns an adaptive privacy schedule

$$
R_1, R_2, \ldots, R_T,
$$

where the first $L$ values are determined by systematic exploration and the remaining $T-L$ values are selected by continuous Bayesian Optimization. The GP continually refines its estimate of the privacy-utility relationship as training progresses and the loss landscape evolves.

## 7. Optimization Metric

The BO framework is independent of the objective function. The implementation supports seven metric variants, all of which are scalar functions of the RDP cost $R$ and the training outcome. Variants A and B require a single validation evaluation after DP perturbation. Variants C through G additionally require a clean (pre-DP) validation evaluation to compute reference loss and logits.

### Variant A: Noisy Update Norm (NUN)

**Motivation.** The Noisy Update Norm (NUN) measures the magnitude of the update sent to the server. This metric captures the joint effect of the client's learning signal and the privacy noise. Weaker privacy (larger $R$) adds less noise and preserves more signal, resulting in a smaller update norm. Stronger privacy (smaller $R$) injects more noise, increasing the norm. By minimizing $m_{nun}$, the BO seeks a privacy level where the signal component dominates the noise.

**Metric.** After local training with DP-SGD, the client computes the update $\Delta_t = w_{local}^{(t)} - w_g^{(t)}$. The NUN metric is defined as

$$
m_{\text{nun}} = \| \Delta_t \|_2 .
$$

**Expected norm.** For a model of dimension $d$, the expected squared norm depends on the noise scale $\sigma_t$ and the number of training steps. The noise contribution grows as $R_t$ decreases (smaller RDP cost means more noise), making the expected NUN larger under stronger privacy.

**Optimization Objective.** BO minimizes

$$
\min_R \; m_{\text{nun}} .
$$

**Interpretation.** Lower values indicate that the released update is dominated by the learning signal rather than by privacy noise. This occurs when $R_t$ is sufficiently large that the noise contribution is negligible relative to the signal.

The BO naturally drives $m_{nun}$ downward by favoring larger $R$ (weaker privacy). The acquisition function's penalty term $\lambda_{aq}$ counterbalances this by penalizing large $R$, creating a principled trade-off between update fidelity and privacy strength. Because $EI_{norm}$ and the penalty are both normalized to $[0, 1]$, $\lambda_{aq}$ directly represents the relative weight of privacy versus update fidelity and will be tuned via grid search.

### Variant B: Model Utility Metric

**Motivation.** This metric directly measures predictive quality. Each client keeps a small validation subset that is never used during local training.

**Evaluation.** After local training with DP-SGD, the client evaluates the trained model $w_{local}^{(t)}$ on the local validation set. The optimization metric is the validation loss:

$$
m_{\text{utility}} = \mathcal{L}_{\text{validation}}(w_{local}^{(t)}).
$$

Cross-entropy loss will be used for classification experiments; alternative losses may be substituted depending on the task.

**Optimization Objective.** BO minimizes

$$
\min_R \; \mathcal{L}_{\text{validation}} .
$$

**Interpretation.** Lower validation loss indicates:

- better predictive accuracy,
- improved generalization,
- a better trade-off between the information carried by the update and the noise added for privacy.

Unlike the NUN metric, this objective directly targets the model's performance on held-out local data.

### Variant C: Utility Retention

**Motivation.** Utility Retention measures how much the DP-SGD training degrades the model's validation performance relative to training without privacy noise. A value close to 1 indicates that privacy noise did not harm accuracy; larger values indicate degradation. This metric isolates the cost of privacy from the absolute loss, making it comparable across different loss scales.

**Metric.** The client computes the validation loss from training with and without DP noise. The utility retention is the ratio of noisy to clean loss:

$$
m_{\text{ret}} = \frac{L_{\text{noisy}}}{L_{\text{clean}}}.
$$

**Optimization Objective.** BO minimizes

$$
\min_R \; m_{\text{ret}} .
$$

**Interpretation.** Values near 1.0 indicate that the DP-SGD training introduced negligible degradation. Values above 1.0 indicate increasing loss due to noise. By minimizing $m_{ret}$, BO seeks privacy levels where the noisy model retains the clean model's predictive performance. Unlike the NUN metric, this objective directly captures utility rather than update magnitude.

### Variant D: Utility Efficiency

**Motivation.** Utility Efficiency measures the fractional loss increase per unit of RDP budget spent. This captures how efficiently each unit of RDP cost is used: a small loss increase per $R$ means the client obtains strong privacy at little cost to model quality. The denominator $R$ intrinsically penalizes larger RDP cost, encoding a privacy-utility trade-off without requiring the acquisition penalty $\lambda_{aq}$.

**Metric.** The client computes the loss degradation due to DP-SGD training and normalizes it by the clean loss and the RDP cost:

$$
m_{\text{eff}} = -\frac{\max(0, L_{\text{noisy}} - L_{\text{clean}})}{L_{\text{clean}} \cdot R}.
$$

The numerator is zero when the noisy loss is not larger than the clean loss, making $m_{eff}$ non-positive.

**Optimization Objective.** BO minimizes

$$
\min_R \; m_{\text{eff}} .
$$

**Interpretation.** Because $m_{eff}$ is negative (or zero), minimizing it drives toward more negative values, corresponding to lower fractional degradation per unit $R$. The BO naturally favors combinations of small $R$ (strong privacy) and low degradation. The privacy preference arises intrinsically from the $R$ denominator, unlike NUN and Utility which rely on the explicit penalty $\lambda_{aq}$.

### Variant E: Utility per Remaining Budget

**Motivation.** Utility per Remaining Budget extends the efficiency concept by normalizing degradation by the remaining RDP budget rather than the per-round RDP cost. This makes the metric budget-aware: when little budget remains, the client is penalized more heavily for wasteful spending. The goal is to allocate limited budget across the remaining rounds as efficiently as possible.

**Metric.** The metric is identical to utility efficiency but uses the remaining budget instead of the per-round RDP cost:

$$
m_{\text{rem}} = -\frac{\max(0, L_{\text{noisy}} - L_{\text{clean}})}{L_{\text{clean}} \cdot R_{\text{remaining}}},
$$

where $R_{\text{remaining}}$ is the client's remaining RDP budget for future rounds.

**Optimization Objective.** BO minimizes

$$
\min_R \; m_{\text{rem}} .
$$

**Interpretation.** As the remaining budget shrinks, the denominator shrinks, making the metric more sensitive to any loss degradation. This encourages the BO to select more conservative $R$ values when budget is scarce, and to explore more freely when budget is abundant.

### Variant F: Signal-to-Noise Ratio (SNR)

**Motivation.** The Signal-to-Noise Ratio (SNR) measures the relative power of the learning signal to the DP noise before perturbation. A high SNR means the update is large relative to the noise scale, indicating that the signal is likely to survive the noise addition. A low SNR means the update will be dominated by noise. By minimizing SNR, the BO drives the system toward a noise-dominated regime, corresponding to stronger privacy.

**Metric.** The client computes the squared L2 norm of the local model update relative to the noise variance:

$$
m_{\text{snr}} = \frac{\|\Delta_t\|_2^2}{\sigma_t^2}.
$$

Since $\sigma_t = \sqrt{\alpha_0 \cdot q^2 / (2 \cdot R_t)}$, SNR grows with $R_t$: larger $R_t$ yields less noise, hence higher SNR.

**Optimization Objective.** BO minimizes

$$
\min_R \; m_{\text{snr}} .
$$

**Interpretation.** Larger $R$ (weaker privacy) produces less noise and therefore higher SNR. By minimizing SNR, the BO naturally favors smaller $R$ (stronger privacy), providing an intrinsic privacy preference without relying on the acquisition penalty $\lambda_{aq}$.

### Variant G: Logit Agreement

**Motivation.** Logit Agreement measures how much the model's predictions (in logit space) change after DP-SGD training. Small changes mean the DP-trained model produces similar class scores to a model trained without privacy noise, indicating that privacy noise did not distort the learned representations. This metric captures prediction-level stability rather than just loss magnitude.

**Metric.** The client computes the average cosine similarity between clean and noisy logit vectors across all validation samples, then defines agreement as the complement:

$$
m_{\text{agr}} = 1 - \frac{1}{N} \sum_{i=1}^N \cos\bigl(z_i^{\text{(clean)}}, z_i^{\text{(noisy)}}\bigr),
$$

where $z_i^{(clean)}$ and $z_i^{(noisy)}$ are the logit vectors for validation sample $i$ from the clean and noisy models respectively, and $\cos(\cdot, \cdot)$ is the cosine similarity. The range is $[0, 2]$, where 0 indicates identical logit directions and 2 indicates opposite directions.

**Optimization Objective.** BO minimizes

$$
\min_R \; m_{\text{agr}} .
$$

**Interpretation.** Lower values indicate better agreement between the clean and noisy model predictions. A value near 0 means the DP-SGD training did not meaningfully alter the model's output distribution. By minimizing agreement, BO seeks privacy levels where the model's predictions remain stable despite the added noise.

## 8. Server Aggregation

The server performs weighted aggregation to mitigate the impact of potentially harmful or extremely large updates.

For every received noisy update $\tilde{\Delta}_i$:

1. Compute its L2 norm: $r_i = \| \tilde{\Delta}_i \|_2$.
2. Compute a robust baseline as the median of all received norms: $b = median(\{r_i\})$.
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

The client does **not** send its chosen $R_t$ to the server. Attenuation weights are computed solely from the observed norms $r_i = \|\Delta_i\|_2$. Clients with stronger privacy (smaller $R$) naturally produce noisier updates with larger norms, resulting in lower weights. This implicit mechanism avoids leaking the client's privacy preference.

## 9. Complete Client Algorithm

**Initialization (per client):**

- Privacy bounds: RDP cost bounds $R_{\min}$, $R_{\max}$, total RDP budget $B_{\text{RDP}}$.
- RDP order: $\alpha_0$ (default 10.0).
- Per-example clipping norm: $C$.
- Subsampling rate: $q = \text{batch\_size} / \text{dataset\_size}$.
- Warm-up length: $L$ (number of exploration rounds).
- Acquisition penalty: $\lambda_{aq}$.
- Observation history: empty list.
- RDP accountant: initialized.
- Phase: Warm-up.

Training runs for a fixed number of $T$ communication rounds ($t = 1, 2, \dots, T$). The server initializes $w_g$ (e.g., randomly) before round 1. The client may stop early if its privacy budget is exhausted. For each round:

1. Receive the current global model $w_g^{(t)}$.
2. If phase = Warm-up:
   - Select $R_t$ from the predefined exploration sequence (grid, Latin hypercube, etc.).
   - Check budget (see Section 5): if $R_t$ exceeds remaining budget, reduce it via binary search in $[R_{\min}, R_t]$ to the largest feasible value.
   - Calibrate noise scale: $\sigma_t = \sqrt{\alpha_0 \cdot q^2 / (2 \cdot R_t)}$.
   - Train locally for $E$ epochs using DP-SGD with noise scale $\sigma_t$ $\rightarrow$ obtain $w_{local}^{(t)}$.
   - Compute local update $\Delta_t = w_{local}^{(t)} - w_g^{(t)}$.
   - Send $\Delta_t$ to the server.
   - Compute the chosen optimization metric $m_t$.
   - Store $(R_t, m_t)$ in the observation history.
   - If the history now contains $L$ observations:
     - Fit the initial Gaussian Process (kernel, learn noise variance).
     - Set phase = BO.
3. Else (phase = BO):
   - Construct $EI_{norm}$ via grid evaluation (Section 6.3) and maximize $\alpha(R)$ over $[R_{\min}, R_{\max}]$ $\rightarrow$ candidate $R^*$.
   - Verify budget (see Section 5): if $R^*$ exceeds remaining budget, reduce it via binary search in $[R_{\min}, R^*]$ to the largest feasible $R \le R^*$.
   - Set $R_t$ to the (possibly reduced) value.
   - Calibrate noise scale: $\sigma_t = \sqrt{\alpha_0 \cdot q^2 / (2 \cdot R_t)}$.
   - Train locally for $E$ epochs using DP-SGD with noise scale $\sigma_t$ $\rightarrow$ obtain $w_{local}^{(t)}$.
   - Compute local update $\Delta_t = w_{local}^{(t)} - w_g^{(t)}$.
   - Send $\Delta_t$ to the server.
   - Compute the chosen optimization metric $m_t$.
   - Append $(R_t, m_t)$ to the history.
   - Update the Gaussian Process with the extended history (refit or incremental update).
4. Update the RDP accountant (Section 5) with the cost $R_t$ used: $R_{\alpha_0}^{(total)} \leftarrow R_{\alpha_0}^{(total)} + R_t$.
5. If the remaining budget cannot support any positive RDP cost, cease participation.

The Bayesian Optimization phase continues for all subsequent rounds, providing an adaptive, personalized privacy schedule that reacts to the client's evolving data and optimization dynamics.

## 10. Experimental Variants

To isolate the effect of the optimization objective, the implementation should support the following configurations while keeping all other components (privacy mechanism, BO procedure, GP model, warm-up strategy, aggregation) identical:

| Variant | BO Objective | Purpose |
|---|---|---|
| PLDP-BO-NUN | $m_{nun} = \lVert\tilde{\Delta}\rVert_2$ | Minimize the noisy update norm, balancing signal preservation against privacy noise. |
| PLDP-BO-Utility | $m_{utility} = \mathcal{L}_{validation}(w_{local})$ | Maximize predictive performance for a given privacy budget. |
| PLDP-BO-UtilityRetention | $m_{ret} = L_{noisy} / L_{clean}$ | Minimize the noisy-to-clean loss ratio, preserving validation performance under DP. |
| PLDP-BO-UtilityEfficiency | $m_{eff}$ | Minimize fractional loss increase per unit RDP cost, encoding an intrinsic privacy preference. |
| PLDP-BO-UtilityPerRemaining | $m_{rem}$ | Minimize fractional loss increase per unit remaining budget, adapting to budget scarcity. |
| PLDP-BO-SNR | $m_{snr} = \lVert\Delta_t\rVert_2^2 / \sigma_t^2$ | Minimize signal-to-noise ratio, driving toward a noise-dominated privacy regime. |
| PLDP-BO-Agreement | $m_{agr}$ | Minimize logit cosine dissimilarity, preserving prediction stability under DP. |

This design enables a direct comparison of how different optimization objectives influence the learned privacy schedules, convergence behavior, and final model performance, while demonstrating that PLDP-BO itself is a general framework whose optimization criterion can be changed without altering its privacy mechanism, Bayesian optimization procedure, or federated learning workflow.
