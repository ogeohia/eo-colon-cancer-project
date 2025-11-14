# Posterior Predictive Bundle (Job 1664802)

This folder contains generated quantities and posterior predictive checks for the hierarchical NegBin model.

Contents

- hierarchical_colon_nb-\*.csv: Generated quantities CSV including y_rep for each posterior draw.
- ppc_summary_1664802.csv: Totals by grouping with predictive intervals, z-scores, and coverage (global, region, sex, age_band).
- ppc_global_total.png: Histogram of global total y_rep with observed overlay.
- ppc_region_totals.png: 90% predictive intervals vs observed totals per region.
- ppc_sex_totals.png: 90% predictive intervals vs observed totals by sex.
- ppc_ageband_totals.png: 90% predictive intervals vs observed totals by age band.

Provenance

- Fit draws: outputs/salvaged/1664802/hierarchical_colon_nb_noyrep-20251107023940.csv (single chain, 1000 warmup + 1500 sampling, diag_e, adapt_delta≈0.985, max_treedepth=13).
- Generated quantities model: models/hierarchical_colon_nb.stan
- Scripts: scripts/generate_yrep.py, scripts/ppc_bundle.py

How to load y_rep and compute quick checks (Python)

```python
import pandas as pd
import numpy as np

# Large CSV with y_rep columns
csv = 'outputs/cmdstan_run/gq_1664802/hierarchical_colon_nb-20251107155530.csv'

# Memory-heavy direct load (alternative is regenerating via CmdStanPy for in-memory access):
df = pd.read_csv(csv, comment='#')
y_rep_cols = [c for c in df.columns if c.startswith('y_rep[')]
y_rep = df[y_rep_cols].to_numpy(dtype=int)  # shape (S, N)
print('y_rep shape:', y_rep.shape)

summ = pd.read_csv('outputs/cmdstan_run/gq_1664802/ppc_summary_1664802.csv')
print(summ.head())
```

Interpretation notes

- Coverage flags (cover50/cover90) indicate calibration; repeated misses suggest mis-specification.
- Large |z_score| values highlight groups where observed totals deviate materially from predictive mean; prioritize those groups for refinement.

Regenerating artifacts

```bash
python scripts/generate_yrep.py --fitted outputs/salvaged/1664802/hierarchical_colon_nb_noyrep-20251107023940.csv
python scripts/ppc_bundle.py --fitted outputs/salvaged/1664802/hierarchical_colon_nb_noyrep-20251107023940.csv --outdir outputs/cmdstan_run/gq_1664802
```
