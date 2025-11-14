# Tests

This directory contains validation tests for the analysis code.

## Test Files

### `test_poisson_model.py`
Validates the Poisson regression model implementation with explicit intercept.

**Purpose:**
- Verify that the model includes a single 'const' intercept column
- Confirm proper handling of reference categories
- Demonstrate interpretability of coefficients
- Test prediction functionality

**Run:**
```bash
python tests/test_poisson_model.py
```

**Expected Output:**
The test should pass all validation checks and display:
- Model fitting results
- Coefficient interpretations (IRRs)
- Prediction verification
- Validation status

## Testing Requirements

Install required packages:
```bash
pip install pandas numpy statsmodels patsy
```

Or using conda:
```bash
conda install -c conda-forge pandas numpy statsmodels patsy
```
