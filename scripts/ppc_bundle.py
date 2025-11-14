#!/usr/bin/env python3
"""
Generate a compact PPC bundle from y_rep draws using the finished posterior.
Outputs written to outputs/cmdstan_run/gq_<jobid>/.

Artifacts:
- ppc_summary_1664802.csv (global, by region, sex, age_band)
- ppc_global_total.png (hist of total y_rep with observed line)
- ppc_region_totals.png (intervals + observed for each region)
- ppc_sex_totals.png (intervals + observed for sex)
- ppc_ageband_totals.png (intervals + observed for age bands)

Usage:
  python scripts/ppc_bundle.py \
    --fitted outputs/salvaged/1664802/hierarchical_colon_nb_noyrep-20251107023940.csv \
    --outdir outputs/cmdstan_run/gq_1664802
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns
import cmdstanpy as csp

def build_df_and_buckets(proj: Path) -> pd.DataFrame:
    df = pd.read_csv(proj / 'data' / 'colon_cancer_full.csv')
    if 'age_cont' in df.columns:
        df = df[df['age_cont'] < 50].reset_index(drop=True)
    if 'py' in df.columns:
        df['py'] = df['py'].clip(lower=1e-12)
    # Minimal columns
    df['male_ind'] = (df['sex_label'] == 'Male').astype(int)
    unit_col = 'registry_code' if 'registry_code' in df.columns else 'country'
    if 'region' not in df.columns:
        df['region'] = 'unknown'
    # Age bands by decade below 50
    if 'age_cont' in df.columns:
        ab = (np.floor(df['age_cont'] / 10) * 10).astype(int)
        df['age_band'] = ab.astype(str) + 's'
    else:
        df['age_band'] = 'all'
    # Keep references for grouping
    df['_unit'] = df[unit_col]
    return df

def build_stan_data_from_df(df: pd.DataFrame) -> dict:
    # Recreate the design pieces consistent with the modeling code
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
    stan_data = dict(
        N=len(df),
        y=df['cases'].astype(int).to_numpy(),
        py=df['py'].to_numpy(),
        J_country=len(unit_codes),
        R_region=len(region_codes),
        country_id=country_id,
        region_id_country=region_id_country,
        K_age=K_age,
        B_age=B_age,
        male=df['male_ind'].astype(int).to_numpy(),
        year_c=((df['year'] - df['year'].mean()) / 10.0).to_numpy(),
        grainsize=int(max(1000, len(df) // 8)),
        prior_only=0,
    )
    return stan_data

def compute_group_summary(draws_mat: np.ndarray, df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    # draws_mat: [S, N] integer counts for y_rep
    obs = df['cases'].to_numpy()
    # Build grouping membership matrix G (N x G)
    groups = pd.Categorical(df[group_key])
    labels = list(groups.categories)
    G = np.zeros((len(df), len(labels)), dtype=np.int8)
    G[np.arange(len(df)), groups.codes] = 1
    # Aggregate per-draw group totals: S x G
    agg = draws_mat @ G
    obs_tot = np.array([obs[groups.codes == j].sum() for j in range(len(labels))])
    # Summaries
    mean = agg.mean(axis=0)
    sd = agg.std(axis=0, ddof=1)
    q5, q50, q95 = np.percentile(agg, [5,50,95], axis=0)
    cover50_lo, cover50_hi = np.percentile(agg, [25,75], axis=0)
    cover90_lo, cover90_hi = q5, q95
    cover50 = ((obs_tot >= cover50_lo) & (obs_tot <= cover50_hi)).astype(float)
    cover90 = ((obs_tot >= cover90_lo) & (obs_tot <= cover90_hi)).astype(float)
    z = np.divide(obs_tot - mean, sd, out=np.full_like(mean, np.nan, dtype=float), where=sd>0)
    return pd.DataFrame({
        'grouping': group_key,
        'group': labels,
        'obs_total': obs_tot,
        'pred_mean': mean,
        'pred_sd': sd,
        'q05': q5,
        'q50': q50,
        'q95': q95,
        'z_score': z,
        'cover50': cover50,
        'cover90': cover90,
    })

def plot_totals_interval(df_summary: pd.DataFrame, title: str, outfile: Path):
    d = df_summary.sort_values('obs_total', ascending=False)
    h = max(2.0, 0.25 * len(d))
    plt.figure(figsize=(10, h))
    y = np.arange(len(d))
    plt.hlines(y, d['q05'], d['q95'], color='C0', alpha=0.5, linewidth=4, label='90% PI')
    plt.plot(d['q50'], y, 'o', color='C0', label='median')
    plt.plot(d['obs_total'], y, 'o', color='black', label='observed')
    plt.yticks(y, d['group'])
    plt.xlabel('Total counts')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_global_hist(agg_draws: np.ndarray, obs_total: int, title: str, outfile: Path):
    plt.figure(figsize=(8,4))
    sns.histplot(agg_draws, bins=40, color='C0', alpha=0.6, stat='density')
    plt.axvline(obs_total, color='black', linestyle='--', label=f'observed={obs_total}')
    plt.title(title)
    plt.xlabel('Total counts (global)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fitted', required=True, help='path to fitted_params CSV from sampling run')
    ap.add_argument('--outdir', required=True, help='output directory, e.g., outputs/cmdstan_run/gq_1664802')
    args = ap.parse_args()

    proj = Path.cwd()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Data
    df = build_df_and_buckets(proj)
    stan_data = build_stan_data_from_df(df)

    # Model (with GQ y_rep)
    stan_file = proj / 'models' / 'hierarchical_colon_nb.stan'
    model = csp.CmdStanModel(stan_file=str(stan_file), cpp_options={'STAN_THREADS': True}, force_compile=False)

    # Generate quantities in-memory from existing posterior draws
    gq = model.generate_quantities(data=stan_data, mcmc_sample=[str(Path(args.fitted))])
    y_rep = gq.stan_variable('y_rep')  # shape [S, N]
    if y_rep.ndim == 1:
        y_rep = y_rep[None, :]

    # Global summary
    obs_total = int(df['cases'].sum())
    global_draws = y_rep.sum(axis=1)

    # Subgroup summaries
    summaries = []
    summaries.append(pd.DataFrame({
        'grouping':'global','group':['all'],
        'obs_total':[obs_total],
        'pred_mean':[float(global_draws.mean())],
        'pred_sd':[float(global_draws.std(ddof=1))],
        'q05':[float(np.percentile(global_draws,5))],
        'q50':[float(np.percentile(global_draws,50))],
        'q95':[float(np.percentile(global_draws,95))],
        'z_score':[float((obs_total - global_draws.mean())/max(1e-9, global_draws.std(ddof=1)))],
        'cover50':[float(np.percentile(global_draws,25) <= obs_total <= np.percentile(global_draws,75))],
        'cover90':[float(np.percentile(global_draws,5) <= obs_total <= np.percentile(global_draws,95))],
    }))
    for key in ['region', 'sex_label', 'age_band']:
        summaries.append(compute_group_summary(y_rep, df, key))

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_csv = outdir / 'ppc_summary_1664802.csv'
    summary_df.to_csv(summary_csv, index=False)

    # Plots
    plot_global_hist(global_draws, obs_total, 'Posterior predictive: global total', outdir / 'ppc_global_total.png')
    plot_totals_interval(summary_df[summary_df['grouping']=='region'], 'Posterior predictive totals by region', outdir / 'ppc_region_totals.png')
    plot_totals_interval(summary_df[summary_df['grouping']=='sex_label'], 'Posterior predictive totals by sex', outdir / 'ppc_sex_totals.png')
    plot_totals_interval(summary_df[summary_df['grouping']=='age_band'], 'Posterior predictive totals by age band', outdir / 'ppc_ageband_totals.png')

    print('PPC bundle written to:', outdir)
    print(' -', summary_csv)
    print(' - ppc_global_total.png')
    print(' - ppc_region_totals.png')
    print(' - ppc_sex_totals.png')
    print(' - ppc_ageband_totals.png')

if __name__ == '__main__':
    os.environ.update({
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'
    })
    main()
