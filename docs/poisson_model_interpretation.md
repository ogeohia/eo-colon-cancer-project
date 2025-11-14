# Poisson Regression Model Interpretation Guide

## Overview
This document explains the interpretation of the Poisson regression model coefficients in the updated `03_poisson-regression.ipynb` notebook, which now includes an explicit intercept term.

## Model Specification

### Mathematical Form
The Poisson generalized linear model (GLM) with log link is specified as:

```
log(E[Y_i]) = X_i β + log(py_i)
```

Where:
- `Y_i` = observed case counts
- `X_i` = design matrix (spline basis, sex dummy, region dummies)
- `β` = coefficient vector
- `py_i` = person-years (exposure offset)

This is equivalent to modeling the rate:
```
E[Y_i] = py_i × exp(X_i β)
```

### Model Components

1. **Intercept (`const`)**: 
   - Added via `sm.add_constant()` 
   - Represents the baseline log incidence rate
   - For reference sex (Female) and reference region (Australia and New Zealand) at the baseline age pattern

2. **Age (B-spline basis)**: 
   - 4 degrees of freedom cubic splines
   - Models non-linear relationship between age and log incidence rate
   - Individual coefficients are not directly interpretable
   - Best visualized by plotting predicted rates across age range

3. **Sex (Male dummy)**: 
   - Reference category: Female (coded as 0)
   - Coefficient represents log incidence rate ratio of Males vs Females
   - IRR = exp(β_male) gives the multiplicative effect

4. **Region (dummy variables)**: 
   - Reference category: Australia and New Zealand
   - Each coefficient represents log incidence rate ratio vs reference region
   - IRR = exp(β_region) gives the multiplicative effect

## Coefficient Interpretation

### Intercept
- **Interpretation**: The log incidence rate for the reference group (Female, Australia and New Zealand) at the baseline age pattern
- **Exponentiated**: exp(const) gives the baseline incidence rate (per person-year) when all other predictors are at their reference values

### Sex Coefficient (Male)
- **Interpretation**: The log incidence rate ratio comparing Males to Females
- **Example**: If β_male = 0.707
  - IRR = exp(0.707) ≈ 2.03
  - Males have approximately 2.03 times the incidence rate of Females, holding age and region constant

### Region Coefficients
- **Interpretation**: The log incidence rate ratio comparing each region to Australia and New Zealand
- **Example**: 
  - If β_eastern_europe = 0.450
    - IRR = exp(0.450) ≈ 1.57
    - Eastern Europe has 1.57 times the incidence rate (or 57% higher rate) compared to Australia and New Zealand
  - If β_eastern_asia = -0.425
    - IRR = exp(-0.425) ≈ 0.65
    - Eastern Asia has 0.65 times the incidence rate (or 35% lower rate) compared to Australia and New Zealand

### Age (Spline) Coefficients
- **Individual coefficients**: Not directly interpretable (they represent basis function weights)
- **Collective interpretation**: Together, they define a smooth, non-linear age-incidence relationship
- **Best practice**: Plot predicted incidence rates across the age range to visualize the age effect

## Key Changes from Previous Implementation

### Before (without explicit intercept)
```python
X = pd.concat([
    spline_basis,
    pd.get_dummies(colon_cancer_full['sex_label'], drop_first=True),
    pd.get_dummies(colon_cancer_full['region'], drop_first=True)
], axis=1).astype(float)
```
- **Problem**: No intercept term
- **Issue**: Spline and dummy coefficients had to collectively represent the baseline, making interpretation unclear

### After (with explicit intercept)
```python
X = pd.concat([
    spline_basis,
    pd.get_dummies(colon_cancer_full['sex_label'], drop_first=True),
    pd.get_dummies(colon_cancer_full['region'], drop_first=True)
], axis=1).astype(float)

# Remove patsy's default 'Intercept' column
if 'Intercept' in X.columns:
    X = X.drop(columns=['Intercept'])

# Add statsmodels intercept (creates 'const' column)
X = sm.add_constant(X, has_constant='add')
```
- **Benefit**: Clear baseline log incidence rate
- **Benefit**: Dummy coefficients are now proper log rate ratios relative to reference categories
- **Benefit**: Follows standard epidemiologic practice for rate models

## Prediction with the Model

### For Training Data
```python
# Predict on the original data
predicted_counts = poisson_results.predict(X, offset=offset)
```

### For New Data (Age Curve Visualization)
```python
# 1. Create age grid
age_grid = np.linspace(colon_cancer_full['age_cont'].min(), 
                       colon_cancer_full['age_cont'].max(), 100)

# 2. Generate spline basis for new ages
spline_basis_grid = dmatrix("bs(age_cont, df=4, include_intercept=False)", 
                            {"age_cont": age_grid}, return_type='dataframe')

# 3. Build prediction matrix (all zeros initially)
X_pred = pd.DataFrame(0, index=np.arange(len(age_grid)), columns=X.columns)

# 4. Set intercept to 1
X_pred['const'] = 1.0

# 5. Fill in spline values
for col in spline_basis_grid.columns:
    if col in X_pred.columns:
        X_pred[col] = spline_basis_grid[col].values

# 6. Set reference values for categorical variables
# (Male = 0 for Female, all region dummies = 0 for reference region)

# 7. Calculate mean offset for typical exposure
mean_log_py = np.log(colon_cancer_full['py']).replace(
    [-np.inf, np.inf], np.nan).dropna().mean()

# 8. Predict
predicted_rate = poisson_results.predict(X_pred, offset=mean_log_py)
```

## Standard Epidemiologic Practice

This implementation now follows standard epidemiologic practice for incidence rate models:

1. ✓ **Explicit intercept**: Provides interpretable baseline rate
2. ✓ **Reference categories**: Clear comparison groups for categorical variables
3. ✓ **Smooth age modeling**: Flexible non-parametric age relationship via splines
4. ✓ **Offset for exposure**: Properly accounts for varying person-years
5. ✓ **Exponentiated coefficients**: Convert to interpretable incidence rate ratios (IRRs)

## Statistical Significance

Check the p-values in the model summary:
- p < 0.05: Statistically significant at 5% level
- p < 0.001: Highly statistically significant
- Confidence intervals that don't include 0 (on log scale) or 1 (on IRR scale) indicate significance

## Further Reading

For overdispersion or more complex variance structures, consider:
- Negative Binomial regression (if variance > mean)
- Robust standard errors (sandwich estimators)
- Hierarchical models with random effects (for region/country variation)

See `04_stan-models.ipynb` for Bayesian hierarchical alternatives.
