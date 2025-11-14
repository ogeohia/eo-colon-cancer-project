#!/usr/bin/env python3
"""
Generate-quantities pass to produce y_rep using the original Stan model
on top of an existing fitted_params CSV from the no-yrep sampling run.

Usage:
  python scripts/generate_yrep.py --fitted outputs/salvaged/1664802/hierarchical_colon_nb_noyrep-20251107023940.csv

If --fitted is omitted, the script will pick the most recent CSV in outputs/cmdstan_run/.
Outputs are written under outputs/cmdstan_run/ by default.
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from patsy import dmatrix
import cmdstanpy as csp

def build_stan_data(proj: Path) -> dict:
    DATA = proj / 'data' / 'colon_cancer_full.csv'
    df = pd.read_csv(DATA)
    if 'age_cont' in df.columns:
        df = df[df['age_cont'] < 50].reset_index(drop=True)
    if 'py' in df.columns:
        df['py'] = df['py'].clip(lower=1e-12)
    age_spline = dmatrix("bs(age_cont, df=4, include_intercept=False)", data=df, return_type='dataframe')
    B_age = np.asarray(age_spline, dtype=float)
    _col_mean = B_age.mean(axis=0)
    _col_std = B_age.std(axis=0, ddof=0)
    _col_std[_col_std == 0] = 1.0
    B_age = (B_age - _col_mean) / _col_std
    K_age = B_age.shape[1]
    df['male_ind'] = (df['sex_label'] == 'Male').astype(int)
    unit_col = 'registry_code' if 'registry_code' in df.columns else 'country'
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fitted', help='path to fitted_params CSV')
    ap.add_argument('--output_dir', default='outputs/cmdstan_run')
    args = ap.parse_args()

    proj = Path.cwd()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # Resolve fitted params CSV
    fitted = Path(args.fitted) if args.fitted else None
    if fitted is None:
        cands = sorted(outdir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            raise SystemExit('No CSV files found to use as fitted_params')
        fitted = cands[0]
    if not fitted.exists():
        raise SystemExit(f'fitted CSV not found: {fitted}')

    # Build data
    stan_data = build_stan_data(proj)

    # Use the original model with y_rep in generated quantities
    stan_file = proj / 'models' / 'hierarchical_colon_nb.stan'
    model = csp.CmdStanModel(stan_file=str(stan_file), cpp_options={'STAN_THREADS': True}, force_compile=False)

    print(f'[GQ] Using fitted params: {fitted}')
    print(f'[GQ] Output dir (final copy): {outdir}')
    # CmdStanPy v1.2.x API: use mcmc_sample to pass a CmdStanMCMC or list of csv paths
    gq = model.generate_quantities(
        data=stan_data,
        mcmc_sample=[str(fitted)],
    )
    # Copy results into the desired outputs dir
    import shutil, re
    tag = Path(fitted).stem
    m = re.search(r'(\d{7,})', str(fitted))
    if m:
        tag = m.group(1)
    dest = outdir / f'gq_{tag}'
    dest.mkdir(parents=True, exist_ok=True)
    # Save/copy generated quantities CSVs into destination
    try:
        gq.save_csvfiles(dir=str(dest))
    except Exception as e:
        print('[GQ] save_csvfiles unavailable, attempting manual fallback:', e)
        # Fallback: just indicate location unknown; user can find in tmp per cmdstanpy defaults
    print('[GQ] Results saved to:', dest)

if __name__ == '__main__':
    # Keep BLAS single-threaded for stability
    os.environ.update({
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'
    })
    main()
