#!/usr/bin/env python3
"""Export key report figures from existing outputs without re-sampling.

Generates:
  outputs/cmdstan_run/gq_1664802/age_incidence_curve.png
  outputs/cmdstan_run/gq_1664802/irr_forest.png
  outputs/cmdstan_run/gq_1664802/ppc_stan.png (if y_rep CSV present)

Uses:
  data/colon_cancer_full.csv
  outputs/stan_summary_full.csv (CmdStan summary CSV)
  outputs/cmdstan_run/gq_*/ hierarchical_colon_nb-*.csv for y_rep (optional)

Safe to run repeatedly. Does not mutate model files.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'colon_cancer_full.csv'
SUMMARY = ROOT / 'outputs' / 'stan_summary_full.csv'
GQ_ROOT = ROOT / 'outputs' / 'cmdstan_run'
PLOTS = ROOT / 'outputs' / 'cmdstan_run' / 'gq_1664802'
PLOTS.mkdir(parents=True, exist_ok=True)

def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    df = pd.read_csv(path, index_col=0)
    # Normalize column names
    cols_lower = {c.lower(): c for c in df.columns}
    if 'mean' not in df.columns and 'Mean' in df.columns:
        df['mean'] = df['Mean']
    if 'sd' not in df.columns:
        if 'SD' in df.columns:
            df['sd'] = df['SD']
        elif 'StdDev' in df.columns:
            df['sd'] = df['StdDev']
        else:
            df['sd'] = np.nan
    return df

def export_age_curve(df: pd.DataFrame, summ: pd.DataFrame):
    alpha = float(summ.loc['alpha','mean']) if 'alpha' in summ.index else 0.0
    beta_age = []
    for idx in summ.index:
        if idx.startswith('beta_age['):
            i = int(idx.split('[')[1].split(']')[0])
            beta_age.append((i, float(summ.loc[idx,'mean'])))
    beta_age = [b for _,b in sorted(beta_age)]
    beta_age = np.array(beta_age) if beta_age else np.zeros(1)
    ages = np.linspace(df['age_cont'].min(), df['age_cont'].max(), 200)
    B = dmatrix('bs(age, df=4, include_intercept=False)', {'age': ages}, return_type='dataframe')
    B = np.asarray(B)
    if B.shape[1] != len(beta_age):
        m = min(B.shape[1], len(beta_age))
        B = B[:, :m]; beta_age = beta_age[:m]
    eta = alpha + B @ beta_age
    rate = np.exp(eta)
    plt.figure(figsize=(6,4))
    plt.plot(ages, rate, color='#1f77b4', lw=2)
    plt.xlabel('Age (years)')
    plt.ylabel('Relative incidence rate')
    plt.title('Predicted incidence vs age (reference strata)')
    plt.tight_layout()
    out = PLOTS / 'age_incidence_curve.png'
    plt.savefig(out, dpi=180); plt.close()
    print('Wrote', out)

def export_irr_forest(df: pd.DataFrame, summ: pd.DataFrame):
    rows = []
    if 'beta_male' in summ.index:
        mu, sd = float(summ.loc['beta_male','mean']), float(summ.loc['beta_male','sd'])
        rows.append(('Male vs Female', np.exp(mu), np.exp(mu-1.96*sd), np.exp(mu+1.96*sd)))
    region_names = sorted(df['region'].dropna().unique())
    for i, name in enumerate(region_names, start=1):
        key = f'region_eff[{i}]'
        if key in summ.index:
            mu, sd = float(summ.loc[key,'mean']), float(summ.loc[key,'sd'])
            rows.append((f'Region: {name}', np.exp(mu), np.exp(mu-1.96*sd), np.exp(mu+1.96*sd)))
    if not rows:
        print('No IRR rows found; skipping forest plot.')
        return
    plot_df = pd.DataFrame(rows, columns=['label','irr','low','high']).sort_values('irr')
    fig_h = max(2.5, 0.35*len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y = np.arange(len(plot_df))
    # Horizontal error bars with points
    ax.errorbar(
        plot_df['irr'], y,
        xerr=[plot_df['irr']-plot_df['low'], plot_df['high']-plot_df['irr']],
        fmt='o', color='#2ca02c', ecolor='#98df8a', capsize=3
    )
    # Use y-tick labels for names to avoid overlap with markers
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['label'], fontsize=9)
    ax.invert_yaxis()  # top item first
    ax.axvline(1.0, color='k', lw=1, ls='--')
    ax.set_xlabel('Incidence rate ratio (IRR)')
    ax.set_title('IRR: sex and region effects')
    ax.grid(axis='x', alpha=0.2)
    # Add left margin so long labels don't overlap the plot area
    fig.tight_layout()
    fig.subplots_adjust(left=0.35)
    out = PLOTS / 'irr_forest.png'
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print('Wrote', out)

def find_yrep_csv() -> Path | None:
    """Return the most recent generated-quantities CSV containing y_rep columns.

    Searches under outputs/cmdstan_run/gq_*/ for CSVs and prefers the latest
    file with any column starting with y_rep or y_rep_ppc.
    """
    if not GQ_ROOT.exists():
        return None
    candidates = []
    for d in GQ_ROOT.glob('gq_*'):
        for csv in d.glob('*.csv'):
            # Quick header scan for y_rep marker
            try:
                with open(csv, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        header = line.strip().split(',')
                        if any(h.startswith('y_rep') or h.startswith('y_rep_ppc') for h in header):
                            candidates.append(csv)
                        break
            except OSError:
                continue
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def export_ppc(df: pd.DataFrame):
    csv = find_yrep_csv()
    if csv is None:
        print('No generated quantities CSV with y_rep found; skipping PPC plot.')
        return
    # Read header to identify y_rep columns (skip cmdstan comment lines)
    header = None
    with open(csv, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('#'):
                header = line.strip().split(',')
                break
    if header is None:
        print('Could not parse header in GQ CSV; skipping PPC.')
        return
    ycols = [c for c in header if c.startswith('y_rep') or c.startswith('y_rep_ppc')]
    if not ycols:
        print('y_rep columns not found; skipping PPC.')
        return
    usecols = ycols[:150]
    df_y = pd.read_csv(csv, usecols=usecols, comment='#')
    obs = df['cases'].astype(int).to_numpy()
    yrep_vals = df_y.to_numpy().ravel()
    ymax = np.percentile(np.concatenate([obs, yrep_vals]), 99.5)
    bins = np.arange(0, int(ymax)+2)
    plt.figure(figsize=(6,4))
    plt.hist(yrep_vals, bins=bins, density=True, alpha=0.5, label='y_rep subset', color='#1f77b4')
    plt.hist(obs, bins=bins, density=True, alpha=0.5, label='observed', color='#ff7f0e')
    plt.xlabel('Counts'); plt.ylabel('Density'); plt.title('Posterior predictive check (marginal)')
    plt.legend(frameon=False); plt.tight_layout()
    out = PLOTS / 'ppc_stan.png'
    plt.savefig(out, dpi=180); plt.close()
    print('Wrote', out)

def main():
    if not DATA.exists():
        raise SystemExit(f'Missing dataset {DATA}')
    if not SUMMARY.exists():
        raise SystemExit(f'Missing summary {SUMMARY}. Run sampling or place summary CSV.')
    df = pd.read_csv(DATA)
    # Align with Stan run_model: restrict to under-50 cohort for plotting
    if "age_cont" in df.columns:
        df = df[df["age_cont"] < 50].reset_index(drop=True)
    summ = read_summary(SUMMARY)
    export_age_curve(df, summ)
    export_irr_forest(df, summ)
    export_ppc(df)

if __name__ == '__main__':
    # Keep BLAS single-threaded for reproducibility
    os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    main()
