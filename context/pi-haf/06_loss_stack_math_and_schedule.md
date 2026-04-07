# Loss Stack: Math and Staged Schedule

Last reviewed: 2026-04-07
Scope: Loss contracts for updated PI-HAF layered-depth baseline.

## Section A: Current Losses (Reality Snapshot)

### Layered-depth active path (current repository state)

1. SILog loss (scale/log depth supervision).
2. Ordinal term (relative ordering regularization).
3. Edge-aware smoothness term.

### PI-HAF SR-side path (current repository state)

1. Flow/vector-field matching loss.
2. Mass conservation loss.
3. Focal frequency loss.

Important:
- These are current implementations in different paths and do not imply final layered-depth baseline equivalence.

## Section B: Target Baseline Loss Stack (User-Requested)

Staged schedule:

1. Stage 1 (epochs 0-5):
- Flow Loss + SSI Loss

2. Stage 2 (epochs 5-10):
- Flow Loss + SSI Loss + Wavelet Edge Loss + Ordinal Loss

## Mathematical Definitions

### 1) Flow Loss

Vector-field matching objective for denoising trajectory consistency:

$$
L_{flow} = \| v_\theta(x_t, t, c) - v^*(x_t, t, c) \|_2^2
$$

where $c$ denotes conditioning context (including layer-aware cues).

### 2) SSI Loss

Scale-and-shift invariant depth consistency:

$$
L_{ssi} = \min_{\alpha,\beta} \; \| \alpha \hat{d} + \beta - d \|_1
$$

(or equivalent SI/SSI formulation selected in implementation).

### 3) Wavelet Edge Loss

High-band consistency over selected wavelet family/level:

$$
L_{wav} = \sum_{\ell \in \mathcal{H}} w_\ell \; \| W_\ell(\hat{d}) - W_\ell(d) \|_p
$$

where $\mathcal{H}$ are detail sub-bands, and default wavelet is `sym4` at level 2.

### 4) Ordinal Loss

Physics ordering penalty:

$$
L_{ord} = \max(0, L_1 - L_2)
$$

where lower-index layer should remain in-front under the defined depth convention.

## Combined Objective by Stage

Stage 1:

$$
L = \lambda_f L_{flow} + \lambda_s L_{ssi}
$$

Stage 2:

$$
L = \lambda_f L_{flow} + \lambda_s L_{ssi} + \lambda_w L_{wav} + \lambda_o L_{ord}
$$

## Stability and Soundness Requirements

1. Every component loss must be logged each step/epoch.
2. Every component must pass finite-value checks.
3. Gradient norms must stay finite and non-zero.
4. Stage transition at epoch boundary must avoid abrupt instability.

## Required Monitoring

1. Per-loss component values over time.
2. Relative gradient scale by component (or proxy diagnostics).
3. Ordinal violation rate trend.
4. Wavelet-edge energy trend.

## Open Parameter Items

1. Exact component weights ($\lambda_f, \lambda_s, \lambda_w, \lambda_o$).
2. Exact wavelet coefficient weighting schedule across stages.
3. Exact SSI variant implementation details for final baseline.
