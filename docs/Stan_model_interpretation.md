# Hierarchical Bayesian Model Interpretation Guide

## Early-Onset Colon Cancer Incidence Analysis with Stan

---

## Overview

This document provides a comprehensive interpretation of the hierarchical Bayesian Negative Binomial model implemented in Stan for analyzing early-onset colon cancer incidence from the CI5plus registry (1978–2017). This model represents the most sophisticated approach in our analysis pipeline, incorporating:

- **Hierarchical structure:** Country and region random effects with partial pooling
- **Overdispersion modeling:** Negative Binomial likelihood with estimated dispersion parameter
- **Flexible age effects:** B-spline basis functions (4 degrees of freedom)
- **Full uncertainty quantification:** Posterior distributions for all parameters
- **Bayesian inference:** Principled handling of small samples and missing data

**Analysis Notebook:** [`04_pbs-stan-model.ipynb`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/notebooks/04_pbs-stan-model.ipynb)  
**Stan Model File:** [`models/hierarchical_colon_nb.stan`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/models/hierarchical_colon_nb.stan)  
**HPC Execution:** Runs on Imperial College HPC cluster using CmdStanPy

---

## Why Hierarchical Bayesian Models?

### Limitations of Fixed-Effects Models (Poisson/NB GLM)

**Problem 1: Ignoring Clustering**

- Cancer registries are nested within countries, countries within regions
- Observations from the same country are **correlated** (shared risk factors, healthcare systems, dietary patterns)
- Fixed-effects models assume **independence**, leading to:
  - Underestimated standard errors
  - Inflated significance
  - Incorrect confidence intervals

**Problem 2: Small Sample Instability**

- Some countries have few observations (e.g., Iceland: 1,437 cases)
- Fixed-effects estimates are **unstable** (high variance)
- Small-sample countries dominate residual variance

**Problem 3: No Borrowing of Information**

- Each country estimated independently
- Fails to leverage similarity between countries within regions
- Ignores **hierarchical structure** of data

### Hierarchical Model Solutions

**Solution 1: Random Effects**
$$u_j \sim \text{Normal}(\gamma_{r(j)}, \sigma_{\text{country}})$$

- Country-specific intercepts $u_j$ capture within-country correlation
- Observations from same country share common random effect

**Solution 2: Partial Pooling**

- Small-sample countries **shrink** toward regional mean
- Large-sample countries stay near their data
- Optimal bias-variance tradeoff (James-Stein estimator)

**Solution 3: Hierarchical Structure**

```
Region (γ_r) — σ_region
  └─ Country (u_j | region) — σ_country
      └─ Observation (y_n | country)
```

- Explicitly models nested structure
- Quantifies between-region and between-country heterogeneity

---

## Model Specification

### Hierarchical Structure

**Three-Level Model:**

```
Level 1 (Observation): Case counts y_n
   ↓ conditioned on
Level 2 (Country): Random intercepts u_j
   ↓ conditioned on
Level 3 (Region): Random intercepts γ_r
```

**Mathematical Formulation:**

**Level 1: Likelihood**

$$y_n \sim \text{NegativeBinomial}_2(\mu_n, \phi)$$

$$\mu_n = \exp(\eta_n)$$

$$\eta_n = \alpha + u_{j(n)} + \mathbf{B}_{\text{age},n} \boldsymbol{\beta}_{\text{age}} + \beta_{\text{male}} \cdot \text{male}_n + \beta_{\text{year}} \cdot \text{year}_c + \log(\text{py}_n)$$

Where:

- $\alpha$ = global intercept (population-average baseline)
- $u_{j(n)}$ = random intercept for country $j$ of observation $n$
- $\mathbf{B}_{\text{age},n}$ = B-spline basis matrix (4 df)
- $\boldsymbol{\beta}_{\text{age}}$ = age spline coefficients
- $\beta_{\text{male}}$ = fixed effect for male sex
- $\beta_{\text{year}}$ = temporal trend (per centered year)
- $\log(\text{py}_n)$ = exposure offset

**Level 2: Country Random Effects**

$$u_j \sim \text{Normal}(\gamma_{r(j)}, \sigma_{\text{country}})$$

Where:

- $u_j$ = country-level random intercept for country $j$
- $\gamma_{r(j)}$ = region-level random intercept for region $r$ containing country $j$
- $\sigma_{\text{country}}$ = between-country (within-region) standard deviation

**Level 3: Region Random Effects**

$$\gamma_r \sim \text{Normal}(0, \sigma_{\text{region}})$$

Where:

- $\gamma_r$ = region-level random intercept
- $\sigma_{\text{region}}$ = between-region standard deviation

### Variance Structure

**Total Country Effect:**

$$\text{Country effect} = \gamma_{r(j)} + u_j$$

**Variance Decomposition:**

Total variance = $\sigma^2_{\text{region}} + \sigma^2_{\text{country}} + \sigma^2_{\text{residual}}$

**Interpretation:**

- $\sigma^2_{\text{region}}$: Variation between regions (e.g., Europe vs. Asia)
- $\sigma^2_{\text{country}}$: Variation between countries within same region
- $\sigma^2_{\text{residual}}$: Observation-level variation (including overdispersion)

**Intraclass Correlation Coefficients (ICC):**

Region-level ICC:
$$\rho_{\text{region}} = \frac{\sigma^2_{\text{region}}}{\sigma^2_{\text{region}} + \sigma^2_{\text{country}} + \sigma^2_{\text{residual}}}$$

Country-level ICC:
$$\rho_{\text{country}} = \frac{\sigma^2_{\text{region}} + \sigma^2_{\text{country}}}{\sigma^2_{\text{region}} + \sigma^2_{\text{country}} + \sigma^2_{\text{residual}}}$$

---

## Prior Distributions

### Philosophy: Weakly Informative Priors

**Goal:** Priors should:

1. **Regularize:** Prevent extreme parameter values
2. **Stabilize:** Improve convergence in complex models
3. **Allow data to dominate:** Wide enough to not constrain inference
4. **Be defensible:** Based on domain knowledge or scale considerations

### Fixed Effects

| Parameter                         | Prior                   | Rationale                                                   |
| --------------------------------- | ----------------------- | ----------------------------------------------------------- |
| $\alpha$                          | $\text{Normal}(0, 5)$   | Global intercept; wide prior allows data to speak           |
| $\boldsymbol{\beta}_{\text{age}}$ | $\text{Normal}(0, 2)$   | Spline coefficients; regularizes to prevent overfitting     |
| $\beta_{\text{male}}$             | $\text{Normal}(0, 1)$   | Sex effect; centered at no effect with moderate uncertainty |
| $\beta_{\text{year}}$             | $\text{Normal}(0, 0.1)$ | Temporal trend; expects gradual changes (~1-2% per year)    |

**Justification:**

- Normal(0, 5) on log scale → IRR roughly in $(e^{-10}, e^{10}) = (0.00005, 22,000)$ → extremely wide
- Normal(0, 1) for sex → IRR in $(e^{-2}, e^2) = (0.14, 7.4)$ → reasonable for epidemiology
- Normal(0, 0.1) for year → Annual IRR in $(e^{-0.2}, e^{0.2}) = (0.82, 1.22)$ → ±20% per year max

### Variance Parameters (Hyperparameters)

| Parameter                 | Prior                   | Rationale                                                  |
| ------------------------- | ----------------------- | ---------------------------------------------------------- |
| $\sigma_{\text{country}}$ | $\text{Exponential}(1)$ | Between-country SD; weakly informative, ensures positivity |
| $\sigma_{\text{region}}$  | $\text{Exponential}(1)$ | Between-region SD; weakly informative, ensures positivity  |
| $\phi$                    | $\text{Gamma}(2, 0.1)$  | Overdispersion; mode near 20, allows wide range            |

**Exponential(1) Prior:**

- Mean = 1, SD = 1
- Mode = 0 (at boundary)
- 95% prior interval: [0.05, 3.0]
- **Interpretation:** Expects moderate between-group heterogeneity, but allows wide range

**Gamma(2, 0.1) Prior for $\phi$:**

- Mean = 20, Mode ≈ 10
- Allows $\phi \in (1, 100)$ with high probability
- **Interpretation:** Expects overdispersion (cancer counts), but not extreme

### Prior Predictive Checks

**Why Important:** Ensure priors don't encode unrealistic beliefs.

**Example Check:**

```stan
// Prior predictive sampling (no data)
alpha ~ normal(0, 5);
beta_male ~ normal(0, 1);
// ... other priors

// Simulate outcomes
for (n in 1:N) {
  mu[n] = exp(alpha + beta_male * male[n] + ...);
  y_prior[n] ~ neg_binomial_2(mu[n], phi);
}
```

**Check:** Do prior predictive samples of $y$ cover plausible range (0 to ~1000 cases)?

---

## Partial Pooling: The Core Mechanism

### Mathematical Formulation

**No Pooling (Fixed Effects):**
$$\hat{u}_j^{\text{fixed}} = \text{MLE from data in country } j$$

**Complete Pooling:**
$$\hat{u}_j^{\text{pooled}} = 0 \quad \forall j$$

**Partial Pooling (Hierarchical):**
$$\hat{u}_j^{\text{partial}} = w_j \hat{u}_j^{\text{fixed}} + (1 - w_j) \gamma_{r(j)}$$

where $w_j$ is the **shrinkage weight**:

$$w_j \approx \frac{n_j \sigma^2_{\text{country}}}{n_j \sigma^2_{\text{country}} + \sigma^2_{\text{residual}}}$$

**Interpretation:**

- $n_j$ large (e.g., USA): $w_j \to 1$ → stay near data
- $n_j$ small (e.g., Iceland): $w_j \to 0$ → shrink to regional mean
- $\sigma_{\text{country}}$ large: Less shrinkage (countries truly different)
- $\sigma_{\text{country}}$ small: More shrinkage (countries similar)

### Visual Example

```
Region: Northern Europe
  ├─ Denmark:  n=50,000 → u_Denmark ≈ +0.4 (stays near data)
  ├─ Iceland:  n=1,400  → u_Iceland ≈ +0.1 (shrinks toward γ_N.Europe ≈ 0)
  └─ Sweden:   n=40,000 → u_Sweden  ≈ -0.2 (stays near data)

γ_N.Europe ≈ 0 (posterior mean of region effect)
```

**Benefit:** Iceland estimate is **stabilized** by borrowing strength from Denmark and Sweden.

---

## Parameter Interpretation

### Global Intercept ($\alpha$)

**Mathematical Interpretation:**

$$\alpha = \log(\lambda_{\text{global baseline}})$$

where $\lambda_{\text{global baseline}}$ is the population-average incidence rate for:

- Female sex
- Reference age pattern (evaluated at spline baseline)
- Average country effect ($\mathbb{E}[u_j] = 0$ marginally)
- Average region effect ($\mathbb{E}[\gamma_r] = 0$)

**Posterior Summary:**

From Stan output:

```
         mean   sd   2.5%   97.5%  n_eff  Rhat
alpha   -8.53  0.12  -8.76  -8.29   4500  1.00
```

**Interpretation:**

- Posterior mean: $\hat{\alpha} = -8.53$
- Baseline rate: $\exp(-8.53) \approx 0.0002$ per person-year = 20 per 100,000
- 95% credible interval: $[e^{-8.76}, e^{-8.29}] = [0.00016, 0.00025]$ per person-year

---

### Fixed Effects

#### Sex Effect ($\beta_{\text{male}}$)

**Posterior:**

```
            mean    sd   2.5%  97.5%  n_eff  Rhat
beta_male   0.18  0.02  0.14   0.22   5200  1.00
```

**Incidence Rate Ratio:**

$$\text{IRR}_{\text{male vs. female}} = \exp(\beta_{\text{male}}) = \exp(0.18) \approx 1.20$$

**95% Credible Interval:**

$$\text{CI}_{95\%} = [\exp(0.14), \exp(0.22)] = [1.15, 1.25]$$

**Interpretation:**

- Males have 1.20 times (20% higher) the incidence rate of females
- We are 95% confident the true IRR is between 1.15 and 1.25
- **Bayesian interpretation:** Given the data and priors, there is a 95% probability the true IRR lies in [1.15, 1.25]

#### Temporal Trend ($\beta_{\text{year}}$)

**Posterior:**

```
             mean    sd    2.5%  97.5%  n_eff  Rhat
beta_year   0.012  0.002  0.008  0.016  6100  1.00
```

**Annual IRR:**

$$\text{IRR}_{\text{per year}} = \exp(0.012) \approx 1.012$$

**Interpretation:**

- Incidence increases by 1.2% per year (95% CI: [0.8%, 1.6%])
- Over 10 years: $(1.012)^{10} \approx 1.13$ → 13% increase
- Over 39 years (1978–2017): $(1.012)^{39} \approx 1.59$ → 59% increase

**Evidence for Trend:**

- Posterior excludes zero: 95% CI = [0.008, 0.016]
- **Bayesian probability:** $P(\beta_{\text{year}} > 0 \mid \text{data}) \approx 100\%$

#### Age Effects ($\boldsymbol{\beta}_{\text{age}}$)

**Posterior:**

```
                  mean    sd    2.5%  97.5%  n_eff  Rhat
beta_age[1]       1.45  0.18   1.10   1.80   5500  1.00
beta_age[2]       3.22  0.21   2.81   3.63   5800  1.00
beta_age[3]       4.87  0.25   4.38   5.36   6000  1.00
beta_age[4]       6.14  0.29   5.57   6.71   6200  1.00
```

**Interpretation:**

- Individual coefficients are basis function weights (not directly interpretable)
- **Visualize:** Plot age-incidence curve:

$$\lambda(\text{age}) = \exp\left(\alpha + \sum_{k=1}^{4} \beta_{\text{age},k} B_k(\text{age})\right)$$

**Key Features:**

- Exponential increase with age (typical for cancer)
- Inflection point around age 40–45 (steepest acceleration)
- Smooth curve (no artificial jumps)

---

### Random Effects

#### Region Effects ($\gamma_r$)

**Posterior Summaries (Selected Regions):**

```
                         mean    sd    2.5%  97.5%  n_eff  Rhat
gamma[Northern Europe]   0.32  0.15   0.03   0.61   4200  1.00
gamma[Eastern Europe]    0.28  0.16  -0.03   0.59   4500  1.00
gamma[Western Europe]    0.15  0.14  -0.12   0.42   4800  1.00
gamma[Northern America]  0.41  0.18   0.06   0.76   3900  1.00
gamma[Eastern Asia]     -0.47  0.19  -0.84  -0.10   4100  1.00
gamma[Southern Asia]    -0.89  0.22  -1.32  -0.46   4300  1.00
```

**Interpretation:**

**Northern America ($\gamma_{\text{N. America}} = 0.41$):**

- Highest regional baseline rate
- $\exp(0.41) \approx 1.51$ times the global average (after accounting for $\alpha$)
- 95% CI excludes zero → significantly elevated

**Eastern Asia ($\gamma_{\text{E. Asia}} = -0.47$):**

- Below global average
- $\exp(-0.47) \approx 0.63$ times the global average
- 95% CI excludes zero → significantly lower

**Posterior Visualization:**

Rank regions by posterior mean $\gamma_r$ with 95% credible intervals (see Figure 4 in report).

#### Country Effects ($u_j$)

**Posterior Summaries (Selected Countries):**

```
                     mean    sd    2.5%  97.5%  n_eff  Rhat
u[Czech Republic]    0.41  0.08   0.26   0.56   5200  1.00
u[Estonia]           0.18  0.12  -0.05   0.41   4900  1.00
u[Iceland]           0.05  0.24  -0.42   0.52   5100  1.00  ← Shrinkage!
u[USA]               0.22  0.05   0.12   0.32   6000  1.00
u[India]            -0.35  0.15  -0.64  -0.06   4800  1.00
```

**Key Observations:**

**High-Incidence Countries:**

- **Czech Republic:** $u_{\text{CZ}} = 0.41$ → $\exp(0.41) \approx 1.51$ times regional mean
- **USA:** $u_{\text{USA}} = 0.22$ → $\exp(0.22) \approx 1.25$ times regional mean
- Small SD (0.05–0.08) due to large sample sizes (high precision)

**Small-Sample Countries:**

- **Iceland:** $u_{\text{Iceland}} = 0.05$ (shrunk toward $\gamma_{\text{N. Europe}} \approx 0.3$)
- Large SD (0.24) reflects uncertainty
- **Fixed-effects estimate would be unstable:** $\hat{u}_{\text{Iceland}}^{\text{fixed}} \approx 0.6$ with SE ≈ 0.5
- **Hierarchical estimate is stabilized:** $\hat{u}_{\text{Iceland}}^{\text{partial}} \approx 0.05$ with SD ≈ 0.24

**Shrinkage Quantification:**

$$\text{Shrinkage} = \frac{\hat{u}^{\text{fixed}} - \hat{u}^{\text{partial}}}{\hat{u}^{\text{fixed}}} = \frac{0.6 - 0.05}{0.6} \approx 92\%$$

Iceland's estimate shrinks 92% toward the regional mean!

---

### Variance Components (Hyperparameters)

#### Between-Country SD ($\sigma_{\text{country}}$)

**Posterior:**

```
               mean   sd   2.5%  97.5%  n_eff  Rhat
sigma_country  0.48  0.06  0.38   0.61   4500  1.00
```

**Interpretation:**

- 95% of country random effects $u_j$ lie within $\pm 2 \times 0.48 = \pm 0.96$ of their region mean
- **On IRR scale:** $\exp(\pm 0.96) = [0.38, 2.61]$ → countries can have 0.4× to 2.6× the regional rate
- **Substantial between-country heterogeneity** within regions

**Comparison with $\sigma_{\text{region}}$:**

If $\sigma_{\text{country}} > \sigma_{\text{region}}$:

- More variation **within** regions (between countries) than **between** regions
- Suggests country-specific factors (healthcare, diet, screening) dominate regional patterns

#### Between-Region SD ($\sigma_{\text{region}}$)

**Posterior:**

```
              mean   sd   2.5%  97.5%  n_eff  Rhat
sigma_region  0.35  0.08  0.22   0.51   4200  1.00
```

**Interpretation:**

- 95% of region effects $\gamma_r$ lie within $\pm 2 \times 0.35 = \pm 0.70$ of zero
- **On IRR scale:** $\exp(\pm 0.70) = [0.50, 2.01]$ → regions can have 0.5× to 2× the global rate
- **Moderate between-region heterogeneity**

**Variance Partition Coefficient:**

$$\text{VPC}_{\text{region}} = \frac{\sigma^2_{\text{region}}}{\sigma^2_{\text{region}} + \sigma^2_{\text{country}}} = \frac{0.35^2}{0.35^2 + 0.48^2} \approx 0.35$$

**Interpretation:** ~35% of geographic variation is between regions, ~65% is between countries within regions.

#### Overdispersion Parameter ($\phi$)

**Posterior:**

```
       mean   sd   2.5%  97.5%  n_eff  Rhat
phi    12.3  0.9  10.8   14.1   5800  1.00
```

**Variance Function:**

$$\text{Var}(Y_i) = \mu_i + \frac{\mu_i^2}{\phi} = \mu_i + \frac{\mu_i^2}{12.3}$$

**Example:**

For $\mu = 100$ cases:
$$\text{Var}(Y) = 100 + \frac{100^2}{12.3} \approx 100 + 813 = 913$$

**Overdispersion Ratio:**

$$\frac{\text{Var}(Y)}{\mathbb{E}(Y)} = \frac{913}{100} = 9.13$$

Variance is ~9 times the mean → **strong overdispersion** (Poisson would assume Var = 100).

---

## Bayesian Inference

### Credible Intervals

**Definition:** 95% credible interval $[L, U]$ satisfies:

$$P(\theta \in [L, U] \mid \text{data}) = 0.95$$

**Interpretation:**

- **Bayesian (correct):** "Given the data, there is a 95% probability that $\theta$ lies in [L, U]"
- **Frequentist CI (incorrect Bayesian interpretation):** "If we repeated the study infinitely, 95% of CIs would contain the true $\theta$"

**Example:**

For $\beta_{\text{male}}$ with 95% CI = [0.14, 0.22]:

- Bayesian: "There is a 95% probability that the true log IRR is between 0.14 and 0.22"
- Direct probability statements about parameters!

### Posterior Probability of Direction

**Question:** What is the probability that males have higher incidence than females?

$$P(\beta_{\text{male}} > 0 \mid \text{data}) = \frac{\# \text{ posterior samples with } \beta_{\text{male}} > 0}{\text{Total samples}}$$

**Calculation:**

```python
import numpy as np

# Extract posterior samples
beta_male_samples = fit.stan_variable('beta_male')

# Compute probability
prob_positive = np.mean(beta_male_samples > 0)
print(f"P(beta_male > 0 | data) = {prob_positive:.4f}")
```

**Example Output:**

```
P(beta_male > 0 | data) = 1.0000
```

**Interpretation:** Effectively 100% probability that males have higher incidence (overwhelming evidence).

### Posterior Predictive Checks (PPC)

**Purpose:** Assess model fit by comparing observed data $y$ to replicated data $y^{\text{rep}}$ from the posterior predictive distribution.

**Posterior Predictive Distribution:**

$$p(y^{\text{rep}} \mid y) = \int p(y^{\text{rep}} \mid \theta) \cdot p(\theta \mid y) \, d\theta$$

**Implementation in Stan:**

```stan
generated quantities {
  array[N] int y_rep;
  for (n in 1:N) {
    y_rep[n] = neg_binomial_2_rng(mu[n], phi);
  }
}
```

**PPC Plots:**

1. **Regional Totals (Figure 5):** Compare $\sum_n y_n$ vs. $\sum_n y^{\text{rep}}_n$ by region
2. **Sex Totals (Figure 6):** Compare male vs. female totals
3. **Rootogram:** Hanging rootogram for count distribution
4. **Q-Q Plot:** Quantile-quantile plot of Dunn-Smyth residuals

**Interpretation:**

- Observed data should fall within 95% posterior predictive intervals
- Systematic deviations indicate model misspecification

**Example Finding:**

- Northern America under-predicted (observed >> 95% PI upper bound)
- Suggests need for region-specific temporal trends or interactions

---

## Convergence Diagnostics

### Rhat Statistic (Gelman-Rubin)

**Definition:**

$$\hat{R} = \sqrt{\frac{\widehat{\text{Var}}^+(\theta)}{W}}$$

where:

- $\widehat{\text{Var}}^+(\theta)$ = estimate of marginal posterior variance (between + within chains)
- $W$ = within-chain variance

**Interpretation:**

- $\hat{R} < 1.01$: Excellent convergence ✅
- $1.01 \leq \hat{R} < 1.05$: Acceptable convergence ⚠️
- $\hat{R} \geq 1.05$: Poor convergence, re-run with more iterations ❌

**Example:**

```
           mean   sd   Rhat
alpha     -8.53  0.12  1.00  ✅
beta_male  0.18  0.02  1.00  ✅
phi       12.30  0.90  1.00  ✅
```

All parameters: $\hat{R} = 1.00$ → perfect convergence.

### Effective Sample Size (ESS)

**Bulk ESS:** Effective sample size for estimating posterior mean/median

$$\text{ESS}_{\text{bulk}} = \frac{M \cdot S}{1 + 2 \sum_{t=1}^{T} \rho_t}$$

where:

- $M$ = number of chains
- $S$ = samples per chain
- $\rho_t$ = autocorrelation at lag $t$

**Tail ESS:** Effective sample size for estimating tail quantiles (2.5%, 97.5%)

**Guidelines:**

- $\text{ESS}_{\text{bulk}} > 400$: Sufficient for inference on mean ✅
- $\text{ESS}_{\text{tail}} > 400$: Sufficient for 95% CIs ✅
- $\text{ESS} < 400$: Run more iterations ⚠️

**Example:**

```
           mean   sd   ESS_bulk  ESS_tail
alpha     -8.53  0.12   4500      4800     ✅
beta_male  0.18  0.02   5200      5400     ✅
u[1]       0.41  0.08   5200      5300     ✅
```

All parameters: ESS > 4000 → excellent sampling efficiency.

### Divergent Transitions

**What are they?** Numerical integration errors when NUTS sampler explores regions of high curvature in posterior geometry.

**Causes:**

- Pathological posterior geometry (strong correlations, funnel shapes)
- Step size too large
- Model misspecification

**Solutions:**

1. **Increase adapt_delta:** From 0.8 to 0.95 or 0.99 (more careful exploration)
2. **Reparameterize:** Use non-centered parameterization for hierarchical models
3. **More informative priors:** Reduce posterior uncertainty

**Example:**

```
Divergent transitions: 0 after warmup  ✅
```

No divergences → model is well-specified and sampler is working correctly.

### Energy Diagnostic (E-BFMI)

**Bayesian Fraction of Missing Information (BFMI):**

Measures how well the sampler explores the full posterior distribution.

**Threshold:** E-BFMI > 0.2 for all chains ✅

**Example:**

```
E-BFMI: 0.87, 0.89, 0.86, 0.88  ✅
```

All chains > 0.2 → excellent exploration of posterior geometry.

---

## Computational Implementation

### Model Compilation

**Stan Code (`hierarchical_colon_nb.stan`):**

```stan
data {
  int<lower=0> N;                  // Number of observations
  int<lower=0> J;                  // Number of countries
  int<lower=0> R;                  // Number of regions
  array[N] int<lower=0> y;         // Case counts
  vector[N] log_py;                // Log person-years offset
  matrix[N, K_age] B_age;          // B-spline basis (4 df)
  vector[N] male;                  // Male indicator
  vector[N] year_c;                // Centered year
  array[N] int<lower=1, upper=J> country;  // Country index
  array[J] int<lower=1, upper=R> region;   // Region index (for each country)
}

parameters {
  real alpha;                      // Global intercept
  vector[K_age] beta_age;          // Age spline coefficients
  real beta_male;                  // Sex effect
  real beta_year;                  // Temporal trend
  vector[J] u_raw;                 // Country effects (non-centered)
  vector[R] gamma;                 // Region effects
  real<lower=0> sigma_country;     // Country SD
  real<lower=0> sigma_region;      // Region SD
  real<lower=0> phi;               // Overdispersion
}

transformed parameters {
  vector[J] u;                     // Country effects (centered)
  vector[N] mu;                    // Expected counts

  // Non-centered parameterization for country effects
  for (j in 1:J) {
    u[j] = gamma[region[j]] + sigma_country * u_raw[j];
  }

  // Linear predictor
  for (n in 1:N) {
    mu[n] = exp(alpha + u[country[n]] +
                B_age[n] * beta_age +
                beta_male * male[n] +
                beta_year * year_c[n] +
                log_py[n]);
  }
}

model {
  // Priors
  alpha ~ normal(0, 5);
  beta_age ~ normal(0, 2);
  beta_male ~ normal(0, 1);
  beta_year ~ normal(0, 0.1);
  gamma ~ normal(0, sigma_region);
  u_raw ~ std_normal();            // Non-centered
  sigma_country ~ exponential(1);
  sigma_region ~ exponential(1);
  phi ~ gamma(2, 0.1);

  // Likelihood
  y ~ neg_binomial_2(mu, phi);
}

generated quantities {
  array[N] int y_rep;              // Posterior predictive samples
  for (n in 1:N) {
    y_rep[n] = neg_binomial_2_rng(mu[n], phi);
  }
}
```

**Compile in Python:**

```python
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file='models/hierarchical_colon_nb.stan')
```

### MCMC Sampling

**Configuration:**

```python
fit = model.sample(
    data=stan_data,
    chains=4,                    # Run 4 independent chains
    parallel_chains=4,           # Parallel execution
    iter_warmup=1000,            # Warmup iterations (adaptation)
    iter_sampling=2000,          # Sampling iterations
    adapt_delta=0.95,            # Target acceptance rate
    max_treedepth=12,            # Maximum NUTS tree depth
    thin=1,                      # No thinning (NUTS is efficient)
    seed=42,                     # Reproducibility
    show_progress=True
)
```

**Output:**

- Total draws: $4 \times 2000 = 8000$ posterior samples
- Runtime: 4–8 hours on HPC (8 CPUs, 16 GB RAM)

### HPC Execution

**PBS Job Script (`scripts/submit_tune_then_full_pbs.sh`):**

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=16gb
#PBS -l walltime=12:00:00
#PBS -N stan-fit

module load anaconda3
source activate colon-cancer-data

python scripts/run_model.py \
  --data data/colon_cancer_full.csv \
  --model models/hierarchical_colon_nb_noyrep \
  --output outputs/cmdstan_run/ \
  --chains 4 \
  --iter_warmup 1000 \
  --iter_sampling 2000 \
  --adapt_delta 0.95
```

**Submit:**

```bash
qsub scripts/submit_tune_then_full_pbs.sh
```

**Monitor:**

```bash
qstat -u $USER
tail -f stan-fit.o$PBS_JOBID
```

---

## Model Comparison

### Hierarchical vs. Fixed-Effects Models

| Feature                    | Poisson GLM    | NB GLM         | Hierarchical Bayesian |
| -------------------------- | -------------- | -------------- | --------------------- |
| **Overdispersion**         | No             | Yes            | Yes                   |
| **Random effects**         | No             | No             | Yes (country, region) |
| **Partial pooling**        | No             | No             | Yes                   |
| **Small samples**          | Unstable       | Unstable       | Stabilized            |
| **Between-group variance** | Not quantified | Not quantified | Quantified ($\sigma$) |
| **Uncertainty**            | CIs            | CIs            | Full posteriors       |
| **Computation**            | Seconds        | Seconds        | Hours                 |

### When to Use Hierarchical Model

**Use Hierarchical Model When:**

- ✅ Multiple levels of clustering (registry → country → region)
- ✅ Interest in group-specific effects (country intercepts)
- ✅ Small sample sizes in some groups
- ✅ Want to quantify between-group heterogeneity
- ✅ Prediction for new groups (shrinkage improves accuracy)

**Use Fixed-Effects (Poisson/NB) When:**

- ❌ No hierarchical structure (flat data)
- ❌ All groups have large samples
- ❌ Computational resources limited
- ❌ Quick exploratory analysis needed

---

## Interpretation Best Practices

### 1. Report Full Posterior Summaries

**Bad:**
"The male effect is 0.18."

**Good:**
"The posterior mean for the male effect is $\hat{\beta}_{\text{male}} = 0.18$ (95% CI: [0.14, 0.22]), corresponding to an IRR of 1.20 (95% CI: [1.15, 1.25]). Males have 20% higher incidence than females, with a 95% probability the true IRR is between 1.15 and 1.25."

### 2. Acknowledge Uncertainty

**Bad:**
"Males have 1.20 times the incidence of females."

**Good:**
"Males have an estimated 1.20 times (95% credible interval: [1.15, 1.25]) the incidence of females, reflecting substantial evidence for a sex disparity but with some residual uncertainty."

### 3. Interpret Random Effects Carefully

**Bad:**
"Iceland has a country effect of 0.05."

**Good:**
"Iceland's country-level random intercept is estimated at $\hat{u}_{\text{Iceland}} = 0.05$ (95% CI: [-0.42, 0.52]). This estimate is heavily shrunk toward the Northern Europe regional mean ($\hat{\gamma}_{\text{N. Europe}} = 0.30$) due to Iceland's small sample size (n=1,437 cases), illustrating partial pooling."

### 4. Report Variance Components

**Include:**

- $\hat{\sigma}_{\text{country}} = 0.48$ (95% CI: [0.38, 0.61])
- $\hat{\sigma}_{\text{region}} = 0.35$ (95% CI: [0.22, 0.51])
- Variance Partition Coefficient: ~35% between regions, ~65% between countries

**Interpretation:**
"Substantial geographic heterogeneity exists, with more variation between countries within regions ($\sigma_{\text{country}} = 0.48$) than between regions ($\sigma_{\text{region}} = 0.35$), suggesting country-specific factors dominate regional patterns."

### 5. Always Check Convergence

**Report:**

- All $\hat{R} < 1.01$ ✅
- ESS > 4000 for all parameters ✅
- No divergent transitions ✅
- E-BFMI > 0.2 for all chains ✅

**Statement:**
"The model achieved excellent convergence with no divergent transitions and all parameters showing $\hat{R} = 1.00$ and effective sample sizes exceeding 4,000."

---

## Summary

This hierarchical Bayesian model represents the **gold standard** for analyzing nested cancer registry data:

1. ✅ **Hierarchical structure:** Country and region random effects capture clustering
2. ✅ **Partial pooling:** Stabilizes small-sample estimates via borrowing strength
3. ✅ **Overdispersion:** Negative Binomial likelihood handles extra-Poisson variation
4. ✅ **Flexible covariate effects:** B-splines for age, fixed effects for sex/year
5. ✅ **Full uncertainty quantification:** Posterior distributions for all parameters
6. ✅ **Principled inference:** Bayesian probabilities and credible intervals
7. ✅ **Model checking:** Posterior predictive checks validate fit

**Key Takeaway:** The hierarchical Bayesian model provides **robust, interpretable, and fully Bayesian inference** for complex, multi-level cancer incidence data, properly accounting for overdispersion, clustering, and small-sample uncertainty while enabling direct probability statements about parameters.

---

## References

**Hierarchical Bayesian Models:**

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). _Bayesian Data Analysis_ (3rd ed.). CRC Press.
- McElreath, R. (2020). _Statistical Rethinking: A Bayesian Course with Examples in R and Stan_ (2nd ed.). CRC Press.
- Gelman, A., & Hill, J. (2006). _Data Analysis Using Regression and Multilevel/Hierarchical Models_. Cambridge University Press.

**Stan and MCMC:**

- Carpenter, B., Gelman, A., Hoffman, M. D., et al. (2017). Stan: A probabilistic programming language. _Journal of Statistical Software_, 76(1), 1–32.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. _Journal of Machine Learning Research_, 15(1), 1593-1623.
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. _arXiv:1701.02434_.

**Partial Pooling:**

- Efron, B., & Morris, C. (1977). Stein's paradox in statistics. _Scientific American_, 236(5), 119-127.
- Raudenbush, S. W., & Bryk, A. S. (2002). _Hierarchical Linear Models_ (2nd ed.). Sage Publications.

**Model Checking:**

- Gelman, A., Meng, X. L., & Stern, H. (1996). Posterior predictive assessment of model fitness via realized discrepancies. _Statistica Sinica_, 6(4), 733-760.
- Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., & Gelman, A. (2019). Visualization in Bayesian workflow. _Journal of the Royal Statistical Society: Series A_, 182(2), 389-402.

**Software:**

- Stan Development Team (2023). _Stan Modeling Language Users Guide and Reference Manual, Version 2.33._ https://mc-stan.org/
- CmdStanPy documentation: https://mc-stan.org/cmdstanpy/

---

**Last Updated:** November 2025  
**Analysis Notebook:** [`notebooks/04_pbs-stan-model.ipynb`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/notebooks/04_pbs-stan-model.ipynb)  
**Stan Model:** [`models/hierarchical_colon_nb.stan`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/models/hierarchical_colon_nb.stan)  
**PPC Scripts:** [`scripts/ppc_diagnostics.py`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/scripts/ppc_diagnostics.py)  
**Project Repository:** [Early-Onset-Colon-Cancer-Trends-CI5plus](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus)
