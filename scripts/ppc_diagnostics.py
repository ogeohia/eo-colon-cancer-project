#!/usr/bin/env python3
"""
Additional PPC diagnostics:
- Distributional overlays (ECDF) and rootograms
- PIT and Dunn–Smyth residuals
- Year and sex-by-age interaction z-score heatmaps

Outputs saved alongside existing PPC bundle in the provided outdir.
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy as csp
from patsy import dmatrix

def load_data(proj: Path) -> pd.DataFrame:
    df = pd.read_csv(proj / 'data' / 'colon_cancer_full.csv')
    df = df[df['age_cont'] < 50].reset_index(drop=True)
    df['py'] = df['py'].clip(lower=1e-12)
    df['male_ind'] = (df['sex_label'] == 'Male').astype(int)
    unit_col = 'registry_code' if 'registry_code' in df.columns else 'country'
    df['_unit'] = df[unit_col]
    df['age_band'] = (np.floor(df['age_cont']/10)*10).astype(int).astype(str)+'s'
    df['year_bucket'] = (np.floor(df['year']/5)*5).astype(int).astype(str)  # 5-year bins
    return df

def build_stan_data(df: pd.DataFrame):
    age_spline = dmatrix("bs(age_cont, df=4, include_intercept=False)", data=df, return_type='dataframe')
    B_age = np.asarray(age_spline, dtype=float)
    _col_mean = B_age.mean(axis=0)
    _col_std = B_age.std(axis=0, ddof=0)
    _col_std[_col_std == 0] = 1.0
    B_age = (B_age - _col_mean) / _col_std
    K_age = B_age.shape[1]
    unit_col = '_unit'
    unit_codes = {c:i+1 for i,c in enumerate(sorted(df[unit_col].unique()))}
    region_codes  = {r:i+1 for i,r in enumerate(sorted(df['region'].unique()))}
    country_id = df[unit_col].map(unit_codes).astype(int).to_numpy()
    unit_to_region = df.groupby(unit_col)['region'].first().to_dict()
    region_id_country = np.array([region_codes[unit_to_region[c]] for c,_ in sorted(unit_codes.items(), key=lambda x: x[1])], dtype=int)
    return dict(
        N=len(df), y=df['cases'].astype(int).to_numpy(), py=df['py'].to_numpy(),
        J_country=len(unit_codes), R_region=len(region_codes), country_id=country_id,
        region_id_country=region_id_country,
        K_age=K_age, B_age=B_age,
        male=df['male_ind'].astype(int).to_numpy(),
        year_c=((df['year'] - df['year'].mean()) / 10.0).to_numpy(),
        grainsize=int(max(1000, len(df)//8)), prior_only=0,
    )

def ecdf_overlay(y: np.ndarray, yrep_draws: np.ndarray, outfile: Path):
    # y: [N], yrep_draws: [S, N]
    plt.figure(figsize=(6,4))
    x = np.sort(y)
    Fx = np.arange(1, len(x)+1)/len(x)
    plt.plot(x, Fx, color='black', label='Observed ECDF')
    # Few random draws for overlay
    idx = np.linspace(0, yrep_draws.shape[0]-1, num=min(50, yrep_draws.shape[0]), dtype=int)
    for i in idx:
        xr = np.sort(yrep_draws[i])
        Fr = np.arange(1, len(xr)+1)/len(xr)
        plt.plot(xr, Fr, color='C0', alpha=0.1)
    plt.xlabel('Counts'); plt.ylabel('ECDF'); plt.title('ECDF overlay: y vs y_rep')
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

def rootogram(y: np.ndarray, yrep_draws: np.ndarray, outfile: Path):
    # Frequency rootogram overall
    maxc = max(int(y.max()), int(np.percentile(yrep_draws, 99.5)))
    bins = np.arange(0, maxc+2)
    obs, _ = np.histogram(y, bins=bins)
    exp = np.histogram(yrep_draws.flatten(), bins=bins)[0] / yrep_draws.shape[0]
    plt.figure(figsize=(8,4))
    c = bins[:-1]
    plt.bar(c-0.2, np.sqrt(obs), width=0.4, color='black', alpha=0.7, label='sqrt(obs)')
    plt.bar(c+0.2, np.sqrt(exp), width=0.4, color='C0', alpha=0.6, label='sqrt(exp)')
    plt.xlabel('Count'); plt.ylabel('sqrt frequency'); plt.title('Rootogram')
    plt.legend(); plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

def pit_and_residuals(y: np.ndarray, yrep_draws: np.ndarray, outfile_hist: Path, outfile_qq: Path):
    # PIT: rank of y among draws per cell (randomized ties) / S
    S, N = yrep_draws.shape
    ranks = np.sum(yrep_draws < y[None, :], axis=0)
    ties = np.sum(yrep_draws == y[None, :], axis=0)
    u = (ranks + np.random.uniform(size=N)*np.maximum(1, ties)) / (S + 1)
    plt.figure(figsize=(6,4))
    sns.histplot(u, bins=30, stat='density', color='C0')
    plt.axhline(1.0, color='black', linestyle='--')
    plt.title('PIT histogram'); plt.tight_layout(); plt.savefig(outfile_hist, dpi=150); plt.close()
    # Dunn–Smyth residuals approx via PIT -> Normal
    from scipy.stats import norm
    r = norm.ppf(np.clip(u, 1e-6, 1-1e-6))
    # QQ plot
    q = np.sort(r); n = len(q); th = norm.ppf((np.arange(1, n+1)-0.5)/n)
    plt.figure(figsize=(5,5))
    plt.scatter(th, q, s=4, alpha=0.6)
    lim = [min(th.min(), q.min()), max(th.max(), q.max())]
    plt.plot(lim, lim, color='black', linestyle='--')
    plt.xlabel('Theoretical N(0,1) quantiles'); plt.ylabel('Residual quantiles')
    plt.title('Dunn–Smyth QQ'); plt.tight_layout(); plt.savefig(outfile_qq, dpi=150); plt.close()

def heatmaps(df: pd.DataFrame, yrep_draws: np.ndarray, outdir: Path):
    # Build per-group z-scores by year and sex-age bands
    obs = df['cases'].to_numpy()
    # Year buckets
    for key in ['year_bucket', 'sex_label', 'age_band']:
        if key not in df.columns:
            return
    def agg_z(group_cols, fname):
        g = df.groupby(group_cols).indices
        rows = []
        for k, idx in g.items():
            idx = np.array(list(idx))
            obs_tot = obs[idx].sum()
            pred = yrep_draws[:, idx].sum(axis=1)
            m, s = pred.mean(), pred.std(ddof=1)
            z = (obs_tot - m) / (s if s>0 else np.nan)
            rows.append((*((k,) if not isinstance(k, tuple) else k), obs_tot, m, s, z))
        cols = group_cols + ['obs_total','pred_mean','pred_sd','z']
        return pd.DataFrame(rows, columns=cols)
    # Year-only
    yr = agg_z(['year_bucket'], 'year')
    # Simple bar plot of year z-scores instead of pivot (single column)
    plt.figure(figsize=(10,4))
    yr_sorted = yr.sort_values('year_bucket')
    plt.bar(yr_sorted['year_bucket'], yr_sorted['z'], color='C0')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('z-score'); plt.xlabel('5-year bucket'); plt.title('Z-score by 5-year bucket')
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(outdir/'hm_year_z.png', dpi=150); plt.close()
    # Sex x age band
    sa = agg_z(['sex_label','age_band'], 'sex_age')
    pt2 = sa.pivot(index='age_band', columns='sex_label', values='z')
    plt.figure(figsize=(6,4))
    sns.heatmap(pt2.sort_index(), annot=True, fmt='.1f', cmap='RdBu_r', center=0)
    plt.title('Z-score by sex x age band'); plt.tight_layout(); plt.savefig(outdir/'hm_sex_age_z.png', dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fitted', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    proj = Path.cwd()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(proj)
    stan_data = build_stan_data(df)
    model = csp.CmdStanModel(stan_file=str(proj/'models'/'hierarchical_colon_nb.stan'), cpp_options={'STAN_THREADS':True}, force_compile=False)
    gq = model.generate_quantities(data=stan_data, mcmc_sample=[str(Path(args.fitted))])
    yrep = gq.stan_variable('y_rep')
    if yrep.ndim == 1:
        yrep = yrep[None, :]

    # Distributional overlays
    ecdf_overlay(df['cases'].to_numpy(), yrep, outdir/'ppc_ecdf_overlay.png')
    rootogram(df['cases'].to_numpy(), yrep, outdir/'ppc_rootogram.png')

    # PIT and residuals
    pit_and_residuals(df['cases'].to_numpy(), yrep, outdir/'ppc_pit_hist.png', outdir/'ppc_dunn_smyth_qq.png')

    # Year & sex-age heatmaps
    heatmaps(df, yrep, outdir)

    print('Diagnostics written to:', outdir)

if __name__ == '__main__':
    os.environ.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    main()
