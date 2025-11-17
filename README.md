# Early-Onset Colon Cancer Trends: Global Analysis with CI5plus Data

A comprehensive Bayesian hierarchical modeling study of early-onset colon cancer incidence trends using population-based cancer registry data from **CI5plus** (Cancer Incidence in Five Continents).

---

## 📌 Project Overview

This project investigates **early-onset colon cancer** (EOCC, diagnosed before age 50) incidence trends across multiple countries and time periods using rigorous statistical methods. Recent epidemiological studies have documented concerning increases in EOCC rates in high-income countries, raising questions about whether these trends:

1. Represent a **true epidemiological shift** specific to younger populations
2. Mirror broader age-period-cohort effects across all age groups
3. Vary systematically by country, region, or socioeconomic development level

### Key Research Questions

- **Are EOCC trends increasing globally, or only in specific regions?**
- **How do incidence patterns differ by sex, age group, and calendar period?**
- **What role do country-level factors (e.g., HDI, healthcare access) play in observed trends?**
- **Can hierarchical Bayesian models capture geographic heterogeneity while borrowing strength across countries?**

### Analytical Approach

The analysis progresses through several stages:

1. **Data Preparation**: Clean, harmonize, and aggregate CI5plus registry data with population denominators
2. **Exploratory Analysis**: Visualize age-specific incidence rates, temporal trends, and geographic patterns
3. **Trend Analysis**: Quantify age-period effects using segmented regression and joinpoint analysis
4. **GLM Regression**: Fit Poisson and Negative Binomial models to account for overdispersion in count data
5. **Bayesian Hierarchical Modeling**: Implement multilevel models in Stan with:
   - Region and country random effects to capture geographic clustering
   - Age splines for flexible non-linear age effects
   - Year trends to estimate temporal changes
   - Negative Binomial likelihoods to handle overdispersion
   - Posterior predictive checks for model validation

### Outputs

- **Comprehensive Report**: PDF with methods, results, interpretation, and visualizations
- **Diagnostic Plots**: Posterior predictive checks, MCMC diagnostics, model convergence assessments
- **Reproducible Pipeline**: End-to-end workflow from raw data to publication-ready figures

---

## 🗂 Repository Structure

```
eo-colon-cancer-project/
├── data/                                          # Input data and documentation
│   ├── colon_cancer_full.csv                      # Main analysis dataset (CI5plus + HDI)
│   ├── country_aggregated_df2.csv                 # Country-level aggregates
│   ├── hdi_2023.csv                               # Human Development Index data
│   ├── HDR25_Statistical_Annex_HDI_Table.xlsx     # HDI source table
│   └── README.md                                  # Data documentation and citations
│
├── models/                            # Stan model files
│   ├── hierarchical_colon_nb.stan     # Full hierarchical NB model
│   ├── hierarchical_colon_nb_noyrep.stan  # Sampling variant (no y_rep)
│   ├── hierarchical_colon_nb          # Compiled executable (full)
│   └── hierarchical_colon_nb_noyrep   # Compiled executable (noyrep)
│
├── notebooks/                         # Analysis workflow
│   ├── 00_data-prep.ipynb             # Data cleaning and harmonization
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   ├── 02_trend-analysis.ipynb        # Temporal trend analysis
│   ├── 03_poisson-NB-regression.ipynb # GLM regression models
│   └── 04_pbs-stan-model.ipynb        # Bayesian hierarchical models
│
├── scripts/                           # Utility and execution scripts
│   ├── run_model.py                   # Stan model runner
│   ├── generate_yrep.py               # Generate posterior predictive draws
│   ├── ppc_diagnostics.py             # Posterior predictive checks
│   ├── ppc_bundle.py                  # Bundle PPC outputs
│   ├── export_figs.py                 # Export publication figures
│   ├── auto_gate.py / .sh             # Automated job monitoring
│   ├── submit_stan.pbs                # HPC job submission script
│   └── submit_and_tail.sh             # Submit + monitor workflow
│
├── outputs/                           # Model outputs and reports
│   ├── cmdstan_run/                   # Stan MCMC and diagnostics
│   │   └── gq_1664802/                # Successful GQ run outputs
│   │       ├── *.png                  # Posterior predictive check plots
│   │       ├── ppc_summary_*.csv      # PPC summary statistics
│   │       └── diagnose.txt           # MCMC diagnostics (all passed)
│   ├── figs/                          # Generated figures
│   ├── reports/                       # Final reports
│   │   └── eo_cc_report.pdf           # Main analysis report
│   └── stan_full_meta.json            # Metadata for full Stan run
│
├── tests/                             # Validation tests
│   ├── test_poisson_model.py          # Poisson GLM tests
│   ├── test_negativebinomial_model.py # NB GLM tests
│   └── README.md                      # Testing documentation
│
├── docs/                              # Documentation
│   ├── poisson_model_interpretation.md  # Poisson model guide
│   ├── NB_model_interpretation.md       # Negative Binomial guide
│   └── Stan_model_interpretation.md     # Stan model guide
│
├── environment.yml                    # Conda environment specification
├── .gitignore                         # Git exclusion patterns
└── README.md                          # This file
```

---

## 📊 Data Sources

### Primary Data

- **CI5plus (Cancer Incidence in Five Continents Plus)**  
  International Agency for Research on Cancer (IARC)  
  🔗 [https://ci5.iarc.fr/CI5plus/](https://ci5.iarc.fr/CI5plus/)

  Population-based cancer registry data covering 100+ registries across 60+ countries from 1953-2017. This project focuses on **colon cancer (ICD-10: C18)** incidence by age, sex, and calendar year.

- **Human Development Index (HDI)**  
  United Nations Development Programme (UNDP)  
  Socioeconomic development indicators at country level (2023)

See [`data/README.md`](data/README.md) for detailed data documentation, and citations.

> **Note**: Raw CI5plus microdata cannot be redistributed per IARC usage terms. Researchers should download directly from IARC. This repository contains aggregated/derived datasets only.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** with scientific computing stack
- **CmdStan 2.35+** for Bayesian models (optional for exploratory analysis)
- **HPC cluster access** (optional, for large-scale MCMC runs)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ogeohia/eo-colon-cancer-project.git
   cd eo-colon-cancer-project
   ```

2. **Set up the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate eo-colon-cancer
   ```

3. **Install CmdStan (for Bayesian models):**
   ```bash
   python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
   ```

### Usage

**Exploratory Analysis (no HPC required):**

```bash
jupyter notebook notebooks/
# Run: 00_data-prep.ipynb → 01_eda.ipynb → 02_trend-analysis.ipynb → 03_poisson-NB-regression.ipynb
```

**Bayesian Models (local):**

```bash
python scripts/run_model.py --model models/hierarchical_colon_nb.stan --chains 4 --iter 2000
```

**HPC Cluster (PBS):**

```bash
# Submit Stan model to PBS cluster
bash scripts/submit_and_tail.sh

# Monitor progress
tail -f stan-fit.o*
```

---

## 📚 Documentation

- **[Poisson Model Interpretation](docs/poisson_model_interpretation.md)**: Guide to Poisson GLM coefficients and IRR interpretation
- **[Negative Binomial Model Interpretation](docs/NB_model_interpretation.md)**: Handling overdispersion in count data
- **[Stan Model Interpretation](docs/Stan_model_interpretation.md)**: Bayesian hierarchical model specification and posterior inference
- **[Data README](data/README.md)**: Data sources, structure, and citation guidelines

---

## 🧪 Testing

Validate model implementations with unit tests:

```bash
# Test Poisson GLM
python tests/test_poisson_model.py

# Test Negative Binomial GLM
python tests/test_negativebinomial_model.py
```

All tests should pass with detailed output showing model fitting, coefficient interpretations, and prediction verification. See [`tests/README.md`](tests/README.md) for details.

---

## 📈 Key Results

The analysis revealed:

- **Geographic heterogeneity**: Significant variation in EOCC trends across countries and regions
- **Sex differences**: Distinct patterns by sex, with faster increases in males in some regions
- **Temporal trends**: Evidence of accelerating incidence in recent decades for certain age groups
- **Model performance**: Negative Binomial hierarchical models effectively captured overdispersion and geographic clustering
- **MCMC diagnostics**: All chains converged (R̂ < 1.01), ESS > 1000 for all parameters

See [`outputs/reports/eo_cc_report.pdf`](outputs/reports/eo_cc_report.pdf) for the full analysis report.

---

## 🛠 Technical Details

### Bayesian Hierarchical Model Specification

The final model (`hierarchical_colon_nb.stan`) implements:

- **Likelihood**: Negative Binomial with log-link to handle overdispersion
- **Hierarchical structure**:
  - Regional random effects (continent-level)
  - Country random effects nested within regions
- **Fixed effects**:
  - Age (B-spline basis with 5 knots)
  - Sex (male indicator)
  - Calendar year (centered, per-decade scaling)
- **Priors**:
  - Weakly informative priors on regression coefficients
  - Half-normal priors on variance components
  - Gamma prior on overdispersion parameter (φ)
- **Parallelization**: `reduce_sum()` for efficient within-chain parallelization
- **Validation**: Prior predictive checks, posterior predictive checks, MCMC diagnostics

### HPC Workflow

The successful production run (job ID 1664802):

- **Sampling**: 4 chains × 2500 iterations (1000 warmup + 1500 sampling)
- **Runtime**: ~7 hours on Imperial College HPC (PBS scheduler)
- **Generated Quantities**: Separate job for posterior predictive draws (y_rep)
- **Diagnostics**: All chains converged, no divergences, ESS > 1000, R̂ < 1.01

---

## 🔮 Future Directions

- **Multi-cancer analysis**: Extend to breast, lung, prostate cancers for comparative insights
- **Advanced age-period-cohort models**: Disentangle temporal effects with intrinsic estimator or hierarchical APC
- **Causal inference**: Incorporate screening policy changes, dietary trends, environmental exposures
- **Spatial models**: Add geographic correlation structure (e.g., CAR/SAR models)
- **Survival analysis**: Link incidence trends to survival outcomes where data available
- **Open science**: Publish pre-registered analysis protocol and share synthetic datasets for reproducibility
