# Negative Binomial Regression Model Interpretation Guide

## Early-Onset Colon Cancer Incidence Analysis

---

## Overview

This document provides a comprehensive interpretation of the Negative Binomial regression model applied to early-onset colon cancer incidence data from the CI5plus registry (1978–2017). The Negative Binomial model extends the Poisson framework by explicitly modeling overdispersion—a common feature of cancer count data where variance substantially exceeds the mean.

**Analysis Notebook:** [`03_poisson-NB-regression.ipynb`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/notebooks/03_poisson-NB-regression.ipynb)

**Key Advantage:** The Negative Binomial model provides more accurate standard errors and confidence intervals when data exhibit extra-Poisson variation, avoiding the false precision and inflated significance levels of Poisson models.

---

## Model Specification

### Mathematical Formulation

The Negative Binomial generalized linear model (GLM) with log link function models colon cancer case counts as:

$$\log(\mathbb{E}[Y_i]) = \mathbf{X}_i \boldsymbol{\beta} + \log(\text{py}_i)$$

**Likelihood:**

$$Y_i \sim \text{NegativeBinomial}_2(\mu_i, \phi)$$

where:

- $\mu_i = \text{py}_i \times \exp(\mathbf{X}_i \boldsymbol{\beta})$ is the mean count
- $\phi$ is the **overdispersion parameter** (also called precision or scale)

**Variance Structure:**

Unlike Poisson ($\text{Var}(Y) = \mu$), the Negative Binomial allows:

$$\text{Var}(Y_i) = \mu_i + \frac{\mu_i^2}{\phi}$$

**Interpretation of $\phi$:**

- **Large $\phi$ (e.g., $\phi > 100$):** Minimal overdispersion, approaches Poisson ($\text{Var} \approx \mu$)
- **Small $\phi$ (e.g., $\phi < 10$):** Substantial overdispersion, variance dominated by $\mu^2/\phi$ term
- **$\phi \to \infty$:** Negative Binomial converges to Poisson

**Alternative Parameterization (Alpha):**

Some software (e.g., `statsmodels.discrete.discrete_model.NegativeBinomial`) uses:

$$\text{Var}(Y_i) = \mu_i + \alpha \mu_i^2$$

where $\alpha = 1/\phi$. Larger $\alpha$ implies more overdispersion.

---

### Design Matrix Components

The design matrix $\mathbf{X}$ is identical to the Poisson model:

#### 1. Intercept (`const`)

- **Purpose:** Baseline log incidence rate
- **Reference Stratum:** Female sex, Australia/New Zealand region, baseline age pattern
- **Interpretation:** $\exp(\beta_0)$ is the incidence rate for the reference group

#### 2. Age Effect (B-spline Basis)

- **Functional Form:** Cubic B-splines with 4 degrees of freedom
- **Mathematical Representation:** $\sum_{k=1}^{4} \beta_{\text{age},k} \cdot B_k(\text{age}_i)$

**Interpretation:** Collectively define a smooth, non-linear log-incidence curve. Individual coefficients $\beta_{\text{age},k}$ are not interpretable in isolation—visualize by plotting the predicted rate curve.

#### 3. Sex Effect (Male Indicator)

- **Coding:** Binary dummy variable
  - Female (reference): $\mathbb{I}_{\text{male}} = 0$
  - Male: $\mathbb{I}_{\text{male}} = 1$

**Interpretation:**
$$\text{IRR}_{\text{male vs. female}} = \exp(\beta_{\text{male}})$$

#### 4. Regional Effects (Dummy Variables)

- **Reference Region:** Australia and New Zealand
- **Number of Regions:** 11 regions (UN M49 classification)

**Interpretation:**
$$\text{IRR}_{\text{region } r \text{ vs. AUS/NZ}} = \exp(\beta_r)$$

---

## Why Negative Binomial? The Overdispersion Problem

### Overdispersion in Cancer Registry Data

**Empirical Observation:**

For colon cancer incidence data, the observed variance is typically **much larger** than the mean:

$$\text{Var}(Y_{\text{observed}}) \gg \mathbb{E}(Y_{\text{observed}})$$

**Sources of Overdispersion:**

1. **Unobserved Heterogeneity:**

   - Within-stratum variation not captured by covariates
   - Individual-level risk factors (genetics, diet, lifestyle) not measured
   - Registry-specific differences in case ascertainment

2. **Clustering:**

   - Correlation within countries/registries
   - Spatial clustering of risk factors
   - Temporal correlation (serial dependence)

3. **Model Misspecification:**
   - Missing interactions (e.g., age × sex, region × year)
   - Non-linear effects not fully captured
   - Omitted confounders

**Consequence of Ignoring Overdispersion:**

When using Poisson models on overdispersed data:

- **Standard errors are underestimated** (too small)
- **Z-statistics are inflated** (too large)
- **P-values are too small** (false positives)
- **Confidence intervals are too narrow** (overconfident)

**Result:** Spurious significance, unreliable inference.

---

## Model Diagnostics: Detecting Overdispersion

### Deviance Test

Calculate the deviance-to-degrees-of-freedom ratio:

$$\text{Overdispersion Ratio} = \frac{\text{Deviance}}{\text{df}_{\text{residual}}}$$

**Interpretation:**

- **Ratio ≈ 1:** No overdispersion (Poisson adequate)
- **Ratio > 3:** Moderate overdispersion (consider Negative Binomial)
- **Ratio > 10:** Severe overdispersion (Negative Binomial strongly preferred)

**Example:**

If Poisson model has Deviance = 45,000 with df = 4,500:

$$\frac{45000}{4500} = 10$$

This indicates **severe overdispersion**—Negative Binomial is appropriate.

### Likelihood Ratio Test

Compare Poisson (nested within Negative Binomial when $\phi \to \infty$) using likelihood ratio test:

$$\text{LR} = 2[\log \mathcal{L}_{\text{NB}} - \log \mathcal{L}_{\text{Poisson}}]$$

Under $H_0: \phi = \infty$ (no overdispersion), $\text{LR} \sim \chi^2_1$ (one parameter: $\phi$).

**Interpretation:**

- Large LR (p < 0.001): Strong evidence for overdispersion
- Negative Binomial significantly improves fit

**Example:**

If $\text{LR} = 1500$ with $p < 0.001$:

- Overwhelming evidence against Poisson
- Negative Binomial is statistically necessary

### Information Criteria

Compare models using AIC/BIC:

$$\text{AIC} = -2 \log \mathcal{L} + 2p$$
$$\text{BIC} = -2 \log \mathcal{L} + p \log(n)$$

**Interpretation:**

- **Lower AIC/BIC:** Better model
- Difference > 10: Strong preference for lower model

**Example:**

| Model             | AIC     | BIC     |
| ----------------- | ------- | ------- |
| Poisson           | 250,000 | 250,100 |
| Negative Binomial | 248,500 | 248,610 |

Difference: $\Delta\text{AIC} = 1500$, $\Delta\text{BIC} = 1490$

**Conclusion:** Negative Binomial vastly preferred (difference >> 10).

---

## Coefficient Interpretation

### Intercept ($\beta_0$)

**Mathematical Interpretation:**

Identical to Poisson model—represents log incidence rate for reference stratum (Female, Australia/New Zealand, average age pattern).

$$\lambda_{\text{baseline}} = \exp(\beta_0)$$

**Example:**

If $\beta_0 = -8.5$:
$$\lambda_{\text{baseline}} = \exp(-8.5) \approx 0.000203 \text{ per person-year} = 20.3 \text{ per 100,000}$$

---

### Sex Coefficient ($\beta_{\text{male}}$)

**Incidence Rate Ratio (IRR):**

$$\text{IRR}_{\text{male vs. female}} = \exp(\beta_{\text{male}})$$

**Interpretation:** Same multiplicative interpretation as Poisson, but with **more accurate standard errors** accounting for overdispersion.

**Example:**

If $\beta_{\text{male}} = 0.18$ with **Negative Binomial SE** = 0.025 (vs. **Poisson SE** = 0.018):

**Negative Binomial:**
$$\text{CI}_{95\%}(\beta_{\text{male}}) = [0.18 - 1.96(0.025), 0.18 + 1.96(0.025)] = [0.131, 0.229]$$
$$\text{CI}_{95\%}(\text{IRR}) = [\exp(0.131), \exp(0.229)] = [1.14, 1.26]$$

**Poisson (for comparison):**
$$\text{CI}_{95\%}(\beta_{\text{male}}) = [0.145, 0.215]$$
$$\text{CI}_{95\%}(\text{IRR}) = [1.16, 1.24]$$

**Key Difference:** Negative Binomial CI is **wider** (more conservative), reflecting true uncertainty. Poisson CI is **falsely narrow** due to ignoring overdispersion.

---

### Region Coefficients ($\beta_r$)

**Incidence Rate Ratio:**

$$\text{IRR}_{\text{region } r} = \exp(\beta_r)$$

**Interpretation:** Same as Poisson, but with correctly inflated standard errors.

**Example:**

Eastern Europe: $\beta_{\text{E. Europe}} = 0.35$

**Negative Binomial:**

- SE = 0.040
- IRR = $\exp(0.35) = 1.42$
- 95% CI: $[1.31, 1.54]$ (wider, more honest)

**Poisson:**

- SE = 0.028 (underestimated)
- IRR = 1.42 (same point estimate)
- 95% CI: $[1.34, 1.50]$ (falsely narrow)

**Practical Impact:** Some regions significant under Poisson may become non-significant under Negative Binomial due to correctly adjusted SEs.

---

### Age (Spline) Coefficients ($\boldsymbol{\beta}_{\text{age}}$)

**Interpretation:** Identical to Poisson—visualize by plotting:

$$\lambda(\text{age}) = \exp\left(\beta_0 + \sum_{k=1}^{4} \beta_{\text{age},k} B_k(\text{age})\right)$$

The age-incidence curve shape is the same, but **confidence bands are wider** under Negative Binomial, reflecting overdispersion.

---

### Overdispersion Parameter ($\phi$ or $\alpha$)

**Critical Parameter:** Quantifies the degree of extra-Poisson variation.

**Interpretation ($\phi$ parameterization):**

From the variance equation:
$$\text{Var}(Y_i) = \mu_i + \frac{\mu_i^2}{\phi}$$

**Example:**

If $\hat{\phi} = 10$:

- For $\mu = 100$: $\text{Var}(Y) = 100 + \frac{100^2}{10} = 100 + 1000 = 1100$
- **Poisson would assume:** $\text{Var}(Y) = 100$ (11-fold underestimation!)

**Interpretation ($\alpha$ parameterization):**

If $\hat{\alpha} = 0.1$ (i.e., $\phi = 10$):

- Variance is $\mu + 0.1\mu^2$
- For large counts, quadratic term dominates

**Hypothesis Test:**

Test $H_0: \alpha = 0$ (or equivalently $\phi = \infty$):

- If p < 0.001: Strong evidence of overdispersion
- Negative Binomial is statistically necessary

---

## Model Implementation Details

### Using statsmodels

```python
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
import numpy as np
import pandas as pd

# 1. Prepare design matrix (identical to Poisson)
# [Same construction as Poisson: splines, sex dummy, region dummies, intercept]

# 2. Log-transform exposure offset
offset = np.log(colon_cancer_full['py'])

# 3. Fit Negative Binomial model (NB2 parameterization)
nb_model = dm.NegativeBinomial(y, X, offset=offset, loglike_method='nb2')
nb_results = nb_model.fit()

# 4. View results
print(nb_results.summary())

# 5. Extract overdispersion parameter
alpha = nb_results.params['alpha']  # If using alpha parameterization
phi = 1 / alpha  # Convert to phi parameterization
print(f"Overdispersion parameter: alpha = {alpha:.4f}, phi = {phi:.2f}")
```

**Key Differences from Poisson:**

1. **Model Class:** Use `discrete_model.NegativeBinomial` instead of `sm.GLM` with Poisson family
2. **Loglikelihood Method:** Specify `loglike_method='nb2'` for quadratic variance function
3. **Extra Parameter:** Model estimates $\alpha$ (or $\phi$) in addition to $\boldsymbol{\beta}$

### NB2 vs NB1 Parameterization

**NB2 (Quadratic):** $\text{Var}(Y) = \mu + \alpha \mu^2$ (most common)

**NB1 (Linear):** $\text{Var}(Y) = \mu + \alpha \mu$

**Recommendation:** Use NB2 for count data (cancers, cases). NB1 occasionally used for rates or when variance grows linearly with mean.

---

## Making Predictions

### Predictions on Training Data

```python
# Predict case counts for training data
predicted_counts_nb = nb_results.predict(X, offset=offset)
```

**Mathematical Operation:**

$$\hat{Y}_i = \exp(\mathbf{X}_i \hat{\boldsymbol{\beta}} + \log(\text{py}_i)) = \text{py}_i \times \exp(\mathbf{X}_i \hat{\boldsymbol{\beta}})$$

**Note:** Point predictions (means) are identical to Poisson. Differences appear in **prediction intervals** and **standard errors**.

### Prediction Intervals

**Negative Binomial:**

To generate prediction intervals accounting for overdispersion:

```python
# Generate predictions with standard errors
predictions = nb_results.get_prediction(X, offset=offset)
pred_summary = predictions.summary_frame(alpha=0.05)

# Extract prediction intervals
pred_mean = pred_summary['mean']
pred_lower = pred_summary['mean_ci_lower']
pred_upper = pred_summary['mean_ci_upper']
```

**Comparison:**

| Model             | Mean | 95% PI Lower | 95% PI Upper | Width |
| ----------------- | ---- | ------------ | ------------ | ----- |
| Poisson           | 150  | 126          | 174          | 48    |
| Negative Binomial | 150  | 105          | 195          | 90    |

**Negative Binomial intervals are wider**, reflecting true uncertainty.

### Generating Age-Incidence Curves

```python
# Same procedure as Poisson, but using nb_results
predicted_rate_nb = nb_results.predict(X_pred, offset=mean_log_py)
incidence_rate_per_100k = predicted_rate_nb / np.exp(mean_log_py) * 1e5

# Confidence bands (accounting for overdispersion)
predictions = nb_results.get_prediction(X_pred, offset=mean_log_py)
pred_summary = predictions.summary_frame(alpha=0.05)

# Plot with confidence bands
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(age_grid, incidence_rate_per_100k, linewidth=2, color='steelblue', label='NB Mean')
plt.fill_between(age_grid,
                 pred_summary['mean_ci_lower'] / np.exp(mean_log_py) * 1e5,
                 pred_summary['mean_ci_upper'] / np.exp(mean_log_py) * 1e5,
                 alpha=0.3, color='steelblue', label='95% CI')
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Incidence Rate (per 100,000)', fontsize=12)
plt.title('Negative Binomial: Age-Incidence Curve with Uncertainty', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Statistical Inference

### Hypothesis Testing

Same interpretation as Poisson:

- **Z-statistics:** $Z = \hat{\beta} / \text{SE}(\hat{\beta})$
- **P-values:** Test $H_0: \beta = 0$

**Critical Difference:** Standard errors are **larger** (more realistic), so:

- Z-statistics are **smaller**
- P-values are **larger**
- Fewer spurious "significant" results

**Example:**

| Parameter                 | Estimate | Poisson SE | Poisson p-value | NB SE | NB p-value |
| ------------------------- | -------- | ---------- | --------------- | ----- | ---------- |
| $\beta_{\text{region X}}$ | 0.10     | 0.035      | 0.004\*\*       | 0.052 | 0.054 (ns) |

**Interpretation:** Region X appears significant under Poisson (p=0.004) but is actually non-significant when overdispersion is properly accounted for (NB p=0.054).

### Confidence Intervals

**95% CI for IRRs:**

$$\text{CI}_{95\%}(\text{IRR}) = \left[\exp(\hat{\beta} - 1.96 \times \text{SE}_{\text{NB}}), \exp(\hat{\beta} + 1.96 \times \text{SE}_{\text{NB}})\right]$$

**Key Point:** Use $\text{SE}_{\text{NB}}$ (larger) instead of $\text{SE}_{\text{Poisson}}$ (falsely small).

---

## Model Comparison: Poisson vs. Negative Binomial

### Visual Comparison: Observed vs. Predicted

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Poisson
axes[0].scatter(poisson_results.fittedvalues, y, alpha=0.3, s=10, color='blue')
axes[0].plot([0, y.max()], [0, y.max()], 'k--', lw=2)
axes[0].set_xlabel('Poisson Predicted')
axes[0].set_ylabel('Observed')
axes[0].set_title('Poisson Model')
axes[0].grid(True, alpha=0.3)

# Negative Binomial
axes[1].scatter(nb_results.fittedvalues, y, alpha=0.3, s=10, color='orange')
axes[1].plot([0, y.max()], [0, y.max()], 'k--', lw=2)
axes[1].set_xlabel('NB Predicted')
axes[1].set_ylabel('Observed')
axes[1].set_title('Negative Binomial Model')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Expected Pattern:**

- Both models have similar point predictions (means)
- Scatter around identity line reflects **overdispersion**
- Negative Binomial explicitly models this scatter via $\phi$

### Summary Statistics Comparison

| Metric             | Poisson  | Negative Binomial | Interpretation             |
| ------------------ | -------- | ----------------- | -------------------------- |
| **Log-Likelihood** | -125,000 | -124,250          | NB fits better             |
| **AIC**            | 250,034  | 248,535           | NB preferred (lower)       |
| **BIC**            | 250,204  | 248,713           | NB preferred (lower)       |
| **Deviance/df**    | 10.2     | 1.1               | NB corrects overdispersion |
| **Pearson χ²/df**  | 11.5     | 1.0               | NB well-calibrated         |

**Interpretation:** Negative Binomial dramatically improves fit and resolves overdispersion.

---

## Model Diagnostics and Limitations

### Residual Diagnostics

**Pearson Residuals:**

$$r_i^P = \frac{Y_i - \hat{\mu}_i}{\sqrt{\hat{\mu}_i + \hat{\mu}_i^2/\hat{\phi}}}$$

Denominator includes overdispersion adjustment (vs. Poisson $\sqrt{\hat{\mu}_i}$).

**Check:**

- Plot $r_i^P$ vs. fitted values: Should show no pattern, centered around 0
- Q-Q plot: Should approximate normality (for large samples)
- Outliers: $|r_i^P| > 3$ warrant investigation

### Goodness-of-Fit

**Pearson Chi-Square Test:**

$$\chi^2 = \sum_i \frac{(Y_i - \hat{\mu}_i)^2}{\text{Var}(Y_i)}$$

where $\text{Var}(Y_i) = \hat{\mu}_i + \hat{\mu}_i^2/\hat{\phi}$ for Negative Binomial.

**Expected:** $\chi^2 / \text{df} \approx 1$ if model fits well.

### Model Assumptions

**Critical Assumptions:**

1. **Correct mean structure:** Log-linear relationship between covariates and rates
2. **Correct variance function:** Quadratic variance-mean relationship ($\text{Var} = \mu + \mu^2/\phi$)
3. **Independence:** After accounting for overdispersion, observations are independent
4. **No zero-inflation:** If excess zeros present, consider Zero-Inflated Negative Binomial (ZINB)

**Sensitivity Checks:**

- Compare NB1 vs. NB2 parameterizations
- Test for zero-inflation using Vuong test
- Assess impact of influential observations

---

## When to Use Negative Binomial

### Decision Tree

```
START: Count data with varying exposure
  │
  ├─ Fit Poisson model
  │   └─ Check Deviance/df ratio
  │        │
  │        ├─ Ratio ≈ 1: Use Poisson ✓
  │        │
  │        └─ Ratio > 3: Overdispersion detected
  │             │
  │             ├─ Fit Negative Binomial
  │             │   └─ Check AIC/BIC improvement
  │             │        │
  │             │        ├─ ΔAIC > 10: Use Negative Binomial ✓
  │             │        └─ ΔAIC < 10: Either model acceptable
  │             │
  │             └─ Alternative: Hierarchical model with random effects
```

### Practical Guidelines

**Use Negative Binomial when:**

- Deviance/df > 3 (moderate overdispersion)
- Pearson χ²/df > 3
- Poisson CIs seem implausibly narrow
- AIC/BIC strongly favor NB (Δ > 10)
- Likelihood ratio test rejects Poisson (p < 0.001)

**Stick with Poisson when:**

- Deviance/df ≈ 1 (no overdispersion)
- Computational simplicity desired
- Quick exploratory analysis

**Consider alternatives when:**

- Hierarchical structure present (use random effects models)
- Excess zeros observed (use ZINB or hurdle models)
- Temporal/spatial correlation (use GEE or mixed models)

---

## Comparison with Hierarchical Bayesian Models

| Feature             | Negative Binomial GLM | Hierarchical Bayesian (Stan)   |
| ------------------- | --------------------- | ------------------------------ |
| **Overdispersion**  | Yes (via $\phi$)      | Yes (via $\phi$)               |
| **Random effects**  | No                    | Yes (country/region)           |
| **Partial pooling** | No                    | Yes (shrinkage)                |
| **Uncertainty**     | Frequentist CIs       | Posterior distributions        |
| **Computation**     | Fast (seconds)        | Slow (hours)                   |
| **Small samples**   | May be unstable       | Stabilized via hierarchy       |
| **Interpretation**  | Marginal effects      | Conditional + marginal effects |

**When to Upgrade to Hierarchical Model:**

- Multiple levels of clustering (registry → country → region)
- Interest in country-specific effects
- Small sample sizes in some strata
- Want to quantify between-group heterogeneity ($\sigma_{\text{country}}$, $\sigma_{\text{region}}$)

**See:** [`04_pbs-stan-model.ipynb`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/notebooks/04_pbs-stan-model.ipynb) and [`docs/Stan_model_interpretation.md`](Stan_model_interpretation.md)

---

## Summary of Best Practices

The Negative Binomial regression implementation for colon cancer incidence follows best practices:

1. ✅ **Overdispersion modeling:** Explicitly accounts for extra-Poisson variation via $\phi$
2. ✅ **Accurate inference:** Standard errors and CIs reflect true uncertainty
3. ✅ **Model comparison:** Formal tests (LR, AIC, BIC) justify NB over Poisson
4. ✅ **Same interpretability:** IRRs and coefficient interpretations unchanged
5. ✅ **Diagnostic checks:** Residuals, goodness-of-fit tests validate model adequacy
6. ✅ **Exposure offset:** Properly models rates (cases per person-year)
7. ✅ **Flexible age effects:** B-splines capture non-linear age-incidence curves

**Key Takeaway:** The Negative Binomial model provides **robust, reliable inference** for overdispersed cancer count data, avoiding the false precision and spurious significance of Poisson models while maintaining straightforward interpretation.

---

## References

**Overdispersion and Negative Binomial:**

- Cameron, A. C., & Trivedi, P. K. (2013). _Regression Analysis of Count Data_ (2nd ed.). Cambridge University Press.
- Hilbe, J. M. (2011). _Negative Binomial Regression_ (2nd ed.). Cambridge University Press.
- Ver Hoef, J. M., & Boveng, P. L. (2007). Quasi-Poisson vs. negative binomial regression: How should we model overdispersed count data? _Ecology_, 88(11), 2766-2772.

**Statistical Theory:**

- McCullagh, P., & Nelder, J. A. (1989). _Generalized Linear Models_ (2nd ed.). Chapman and Hall/CRC.
- Agresti, A. (2015). _Foundations of Linear and Generalized Linear Models_. Wiley.

**Epidemiologic Applications:**

- Breslow, N. E. (1984). Extra-Poisson variation in log-linear models. _Applied Statistics_, 33(1), 38-44.
- Lawless, J. F. (1987). Negative binomial and mixed Poisson regression. _Canadian Journal of Statistics_, 15(3), 209-225.

**Software Documentation:**

- Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. _Proceedings of the 9th Python in Science Conference_.
- statsmodels NegativeBinomial documentation: https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html

---

**Last Updated:** November 2025  
**Analysis Notebook:** [`notebooks/03_poisson-NB-regression.ipynb`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/notebooks/03_poisson-NB-regression.ipynb)  
**Validation Tests:** [`tests/test_negativebinomial_model.py`](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus/blob/main/tests/test_negativebinomial_model.py)  
**Project Repository:** [Early-Onset-Colon-Cancer-Trends-CI5plus](https://github.com/ogeohia/Early-Onset-Colon-Cancer-Trends-CI5plus)
