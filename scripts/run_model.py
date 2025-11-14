import os, json, time
from pathlib import Path
import pandas as pd
import numpy as np
from patsy import dmatrix
import cmdstanpy as csp
# Enable short debug run with env flag
DEBUG = os.getenv("STAN_DEBUG", "0") == "1"

# Helper: run sampling with a config, write diagnostics/summary, and return (fit, ok)
from pathlib import Path as _P
def run_and_check(model, data_obj, cfg, prefix):
    # Determine available CPUs (prefer scheduler-provided counts)
    allowed = 0
    for key in ("PBS_NP", "NCPUS", "SLURM_CPUS_PER_TASK"):
        val = os.environ.get(key)
        if val and str(val).isdigit():
            allowed = int(val)
            break
    if not allowed:
        try:
            if hasattr(os, "sched_getaffinity"):
                allowed = len(os.sched_getaffinity(0))
        except Exception:
            allowed = 0
    if not allowed:
        allowed = os.cpu_count() or 1

    # Compute threads per chain: honor explicit override, else derive from allowed/chains
    target_chains = int(cfg.get("chains", 1))
    env_tpc = os.getenv("STAN_THREADS_PER_CHAIN")
    if env_tpc and str(env_tpc).isdigit():
        tpc = max(1, int(env_tpc))
    else:
        tpc = max(1, allowed // max(1, target_chains))
    os.environ["STAN_NUM_THREADS"] = str(tpc)
    # Ensure we don't oversubscribe CPUs: cap parallel_chains so tpc * parallel_chains <= allowed
    requested_parallel = int(cfg.get("parallel_chains", cfg.get("chains", 1)))
    max_parallel = max(1, allowed // max(1, tpc))
    effective_parallel = max(1, min(requested_parallel, max_parallel))
    # Build kwargs compatible across cmdstanpy versions (avoid deprecated 'init')
    sample_kwargs = dict(
        data=data_obj,
        chains=cfg.get("chains", 4),
        parallel_chains=effective_parallel,
        iter_warmup=cfg.get("iter_warmup", 1000),
        iter_sampling=cfg.get("iter_sampling", 1000),
        # adapt_engaged is set explicitly below depending on fixed_param
        adapt_delta=cfg.get("adapt_delta", 0.9),
        max_treedepth=cfg.get("max_treedepth", 10),
        step_size=cfg.get("step_size", None),
        seed=cfg.get("seed", 12345),
        show_progress=True,
        show_console=True,
        refresh=cfg.get("refresh", 50),
    )
    # Allow fixed_param for prior predictive
    if cfg.get("fixed_param", False):
        sample_kwargs["fixed_param"] = True
        # For fixed_param: no adaptation, no warmup, and no HMC-specific args
        sample_kwargs["adapt_engaged"] = False
        sample_kwargs["iter_warmup"] = 0
        # Ensure we don't pass any HMC/adaptation-specific arguments
        for k in (
            "adapt_delta",
            "max_treedepth",
            "step_size",
            "metric",
            "threads_per_chain",
            "save_warmup",
        ):
            sample_kwargs.pop(k, None)
    else:
        # Non-fixed runs: ensure adaptation is engaged unless explicitly disabled
        sample_kwargs["adapt_engaged"] = cfg.get("adapt_engaged", True)
        # Allow metric override (diag_e or dense_e) via STAN_METRIC
        metric = os.getenv("STAN_METRIC", "diag_e")
        sample_kwargs["metric"] = metric
    # Output all CmdStan files under outputs/cmdstan_run to avoid /var/tmp permissions
    outdir = _P("outputs") / "cmdstan_run"
    outdir.mkdir(parents=True, exist_ok=True)
    sample_kwargs["output_dir"] = str(outdir)
    # Optional threading
    # Always set threads_per_chain from computed tpc (or env override) for non-fixed runs
    if not cfg.get("fixed_param", False):
        sample_kwargs["threads_per_chain"] = tpc
    # Emit a small runtime note about the layout used
    print(f"[Stan] allowed_cpus={allowed} threads_per_chain={tpc} parallel_chains={effective_parallel}")
    fit = model.sample(**sample_kwargs)
    outdir = _P("outputs"); outdir.mkdir(exist_ok=True, parents=True)
    ok = True
    try:
        diag = fit.diagnose()
        (outdir / f"{prefix}_diagnose.txt").write_text(diag)
        ok = ok and ("No problems detected" in str(diag))
    except Exception as e:
        (outdir / f"{prefix}_diagnose.txt").write_text(f"diagnose failed: {e}")
        ok = False
    try:
        summ = fit.summary()
        summ.to_csv(outdir / f"{prefix}_summary.csv")
        if "R_hat" in summ.columns:
            rhat_thresh = float(os.getenv("STAN_RHAT_THRESH", "1.05"))
            ok = ok and bool((summ["R_hat"].fillna(0) < rhat_thresh).all())
    except Exception:
        pass
    with open(outdir / f"{prefix}_meta.json", "w") as f:
        json.dump({"config": cfg, "ok": ok}, f, indent=2)
    return fit, ok

# keep BLAS single-threaded
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

PROJ = Path.cwd()
DATA = PROJ / "data" / "colon_cancer_full.csv"
# Choose Stan model: default to no-yrep variant for faster sampling unless STAN_SAVE_YREP=1
STAN_ORIG = PROJ / "models" / "hierarchical_colon_nb.stan"
STAN_NOYREP = PROJ / "models" / "hierarchical_colon_nb_noyrep.stan"
SAVE_YREP = os.getenv("STAN_SAVE_YREP", "0") == "1"
STAN = STAN_ORIG if SAVE_YREP else STAN_NOYREP
OUT = PROJ / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ----- build stan_data -----
df = pd.read_csv(DATA)
# Restrict to under-50 for primary Stan analyses if column exists
if "age_cont" in df.columns:
    df = df[df["age_cont"] < 50].reset_index(drop=True)
# Guard exposure to avoid log(0) issues downstream
if "py" in df.columns:
    n_nonpos = int((df["py"] <= 0).sum())
    if n_nonpos:
        print(f"[prep] Found {n_nonpos} rows with non-positive py; clipping to 1e-12 to avoid log(0)")
    df["py"] = df["py"].clip(lower=1e-12)
age_spline = dmatrix("bs(age_cont, df=4, include_intercept=False)", data=df, return_type='dataframe')
B_age = np.asarray(age_spline, dtype=float)
# Standardize spline basis columns for better sampler geometry
_col_mean = B_age.mean(axis=0)
_col_std = B_age.std(axis=0, ddof=0)
_col_std[_col_std == 0] = 1.0
B_age = (B_age - _col_mean) / _col_std
K_age = B_age.shape[1]
df['male_ind'] = (df['sex_label'] == 'Male').astype(int)
# Use registry_code when available; else fallback to country as unit
unit_col = 'registry_code' if 'registry_code' in df.columns else 'country'
unit_codes = {c:i+1 for i,c in enumerate(sorted(df[unit_col].unique()))}
region_codes  = {r:i+1 for i,r in enumerate(sorted(df['region'].unique()))}
country_id = df[unit_col].map(unit_codes).astype(int).to_numpy()
unit_to_region = df.groupby(unit_col)['region'].first().to_dict()
region_id_country = np.array([region_codes[unit_to_region[c]] for c,_ in sorted(unit_codes.items(), key=lambda x: x[1])], dtype=int)

N_total = len(df)
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

N_total = stan_data["N"]

"""
Compute default chain/thread layout for environment context. Actual threads_per_chain
is (re)computed per run configuration inside run_and_check.
"""
allowed = os.cpu_count() or 1
chains = min(4, max(1, allowed))

# Compile: allow env toggle
force = os.getenv("STAN_FORCE_COMPILE", "0") == "1"
model = csp.CmdStanModel(
    stan_file=str(STAN),
    cpp_options={"STAN_THREADS": True},
    force_compile=force,
)

t0 = time.perf_counter()

# Config ladder for tuning → full (env can override)
configs = [
    {
        "name": "prior_pred",
        "chains": int(os.getenv("STAN_PRIOR_CHAINS", "1")),
        "parallel_chains": int(os.getenv("STAN_PRIOR_CHAINS", "1")),
        "iter_warmup": int(os.getenv("STAN_PRIOR_WARMUP", "250")),
        "iter_sampling": int(os.getenv("STAN_PRIOR_SAMPLING", "250")),
        "adapt_delta": float(os.getenv("STAN_PRIOR_ADAPT_DELTA", "0.9")),
        "max_treedepth": int(os.getenv("STAN_PRIOR_MAX_TREEDEPTH", "10")),
        "step_size": None if os.getenv("STAN_PRIOR_STEP_SIZE") is None else float(os.getenv("STAN_PRIOR_STEP_SIZE")),
        "refresh": int(os.getenv("STAN_PRIOR_REFRESH", "50")),
        "fixed_param": True,
    },
    {
        "name": "debug1",
        "chains": int(os.getenv("STAN_TUNE_CHAINS", "1")),
        "parallel_chains": int(os.getenv("STAN_TUNE_CHAINS", "1")),
        # threads_per_chain is computed in run_and_check based on available CPUs
        "iter_warmup": int(os.getenv("STAN_TUNE_WARMUP", "300")),
        "iter_sampling": int(os.getenv("STAN_TUNE_SAMPLING", "300")),
        "adapt_delta": float(os.getenv("STAN_TUNE_ADAPT_DELTA", "0.95")),
        "max_treedepth": int(os.getenv("STAN_TUNE_MAX_TREEDEPTH", "12")),
        # Let adaptation pick step size by default
        "step_size": None if os.getenv("STAN_TUNE_STEP_SIZE") is None else float(os.getenv("STAN_TUNE_STEP_SIZE")),
        "refresh": int(os.getenv("STAN_TUNE_REFRESH", "50")),
    },
    {
        "name": "stabilize",
        "chains": max(1, int(os.getenv("STAN_TUNE_CHAINS", "4"))),
        "parallel_chains": max(1, int(os.getenv("STAN_TUNE_CHAINS", "4"))),
        # threads_per_chain is computed in run_and_check based on available CPUs
        "iter_warmup": int(os.getenv("STAN_STAB_WARMUP", "1000")),
        "iter_sampling": int(os.getenv("STAN_STAB_SAMPLING", "1000")),
        "adapt_engaged": True,
        "adapt_delta": float(os.getenv("STAN_STAB_ADAPT_DELTA", "0.98")),
        "max_treedepth": int(os.getenv("STAN_STAB_MAX_TREEDEPTH", "15")),
        "step_size": None if os.getenv("STAN_STAB_STEP_SIZE") is None else float(os.getenv("STAN_STAB_STEP_SIZE")),
        "refresh": int(os.getenv("STAN_STAB_REFRESH", "50")),
    },
    {
        "name": "full",
        "chains": int(os.getenv("STAN_CHAINS", "4")),
        "parallel_chains": int(os.getenv("STAN_CHAINS", "4")),
        # threads_per_chain is computed in run_and_check based on available CPUs
        "iter_warmup": int(os.getenv("STAN_WARMUP", "1000")),
        "iter_sampling": int(os.getenv("STAN_SAMPLING", "1000")),
        "adapt_engaged": True,
        "adapt_delta": float(os.getenv("STAN_ADAPT_DELTA", "0.99")),
        "max_treedepth": int(os.getenv("STAN_MAX_TREEDEPTH", "15")),
        "step_size": None if os.getenv("STAN_STEP_SIZE") is None else float(os.getenv("STAN_STEP_SIZE")),
        "refresh": int(os.getenv("STAN_REFRESH", "50")),
    },
]

mode = os.getenv("STAN_MODE","tune")  # "tune" or "full"
# Default to multi-pass tuning so we escalate beyond the quick debug settings
single_pass = os.getenv("STAN_TUNE_SINGLE_PASS", "0") == "1"
Path("outputs").mkdir(exist_ok=True, parents=True)
status_path = Path("outputs/fit_status.txt")

fit = None
if mode == "full":
    # Optional prior predictive run
    if os.getenv("STAN_RUN_PRIOR_PRED", "1") == "1":
        prior_data = dict(stan_data)
        prior_data["prior_only"] = 1
        _fit_prior, _ = run_and_check(model, prior_data, configs[0], "stan_prior_pred")
    # Optional quick subset test before full
    subset_n = int(os.getenv("STAN_SUBSET_N", "0"))
    if subset_n > 0 and subset_n < N_total:
        rng = np.random.default_rng(202)
        idx = np.sort(rng.choice(N_total, size=subset_n, replace=False))
        stan_data_sub = dict(stan_data)
        stan_data_sub.update({
            "N": int(len(idx)),
            "y": stan_data["y"][idx],
            "py": stan_data["py"][idx],
            "country_id": stan_data["country_id"][idx],
            "B_age": stan_data["B_age"][idx, :],
            "male": stan_data["male"][idx],
            "year_c": stan_data["year_c"][idx],
            "prior_only": 0,
            # keep grainsize reasonably small for small N
            "grainsize": int(max(1, len(idx) // 4)),
        })
        _fit_sub, _ = run_and_check(model, stan_data_sub, {
            "name": "subset_quick",
            "chains": 1,
            "parallel_chains": 1,
            "iter_warmup": 200,
            "iter_sampling": 200,
            "adapt_delta": 0.9,
            "max_treedepth": 10,
            "refresh": 50,
        }, "stan_subset")
    # Now full run
    fit, ok = run_and_check(model, stan_data, configs[-1], "stan_full")
    status_path.write_text("OK\n" if ok else "NEEDS_TUNING\n")
else:
    # Prior predictive first (small, cheap)
    if os.getenv("STAN_RUN_PRIOR_PRED", "1") == "1":
        prior_data = dict(stan_data)
        prior_data["prior_only"] = 1
        _fit_prior, _ = run_and_check(model, prior_data, configs[0], "stan_prior_pred")
    # Small subset test for geometry/IO
    subset_n = int(os.getenv("STAN_SUBSET_N", os.getenv("STAN_TUNE_SUBSET_N", "5000")))
    if subset_n > 0 and subset_n < N_total:
        rng = np.random.default_rng(101)
        idx = np.sort(rng.choice(N_total, size=subset_n, replace=False))
        stan_data_sub = dict(stan_data)
        stan_data_sub.update({
            "N": int(len(idx)),
            "y": stan_data["y"][idx],
            "py": stan_data["py"][idx],
            "country_id": stan_data["country_id"][idx],
            "B_age": stan_data["B_age"][idx, :],
            "male": stan_data["male"][idx],
            "year_c": stan_data["year_c"][idx],
            "prior_only": 0,
            "grainsize": int(max(1, len(idx) // 4)),
        })
        _fit_sub, _ = run_and_check(model, stan_data_sub, {
            "name": "subset_debug",
            "chains": 1,
            "parallel_chains": 1,
            "iter_warmup": int(os.getenv("STAN_SUB_WARMUP", "200")),
            "iter_sampling": int(os.getenv("STAN_SUB_SAMPLING", "200")),
            "adapt_delta": float(os.getenv("STAN_SUB_ADAPT_DELTA", "0.9")),
            "max_treedepth": int(os.getenv("STAN_SUB_MAX_TREEDEPTH", "10")),
            "refresh": int(os.getenv("STAN_SUB_REFRESH", "50")),
        }, "stan_subset")
    # Then debug/stabilize ladder
    fit, ok = run_and_check(model, stan_data, configs[1], "stan_debug1")
    if not ok and not single_pass:
        fit, ok = run_and_check(model, stan_data, configs[2], "stan_stabilize")
        if not ok:
            cfg2 = dict(configs[2]); cfg2.update({"name":"stabilize2","adapt_delta":0.99,"max_treedepth":15,"step_size":0.02})
            fit, ok = run_and_check(model, stan_data, cfg2, "stan_stabilize2")
    status_path.write_text("OK\n" if ok else "NEEDS_TUNING\n")
elapsed = (time.perf_counter() - t0) / 60.0
print(f"Elapsed: {elapsed:.1f} min; CPUs={allowed}; chains={chains}")

fit.summary().to_csv(OUT / "stan_summary_full.csv")
with open(OUT / "diagnose.txt", "w") as f:
    f.write(fit.diagnose())
print("Saved outputs to", OUT)

# Record CmdStan and environment versions
def _write_run_metadata(path):
    meta = {}
    try:
        meta["cmdstanpy_version"] = getattr(csp, "__version__", "unknown")
    except Exception:
        meta["cmdstanpy_version"] = "unknown"
    try:
        cs_path = Path(csp.cmdstan_path())
        meta["cmdstan_path"] = str(cs_path)
        ver_file = cs_path / "VERSION"
        if ver_file.exists():
            meta["cmdstan_version"] = ver_file.read_text().strip()
    except Exception as e:
        meta["cmdstan_path_error"] = str(e)
    try:
        import platform, sys
        meta["python"] = sys.version
        meta["platform"] = platform.platform()
    except Exception:
        pass
    try:
        import numpy, pandas, scipy, arviz
        meta["packages"] = {
            "numpy": getattr(numpy, "__version__", None),
            "pandas": getattr(pandas, "__version__", None),
            "scipy": getattr(scipy, "__version__", None),
            "arviz": getattr(arviz, "__version__", None),
        }
    except Exception:
        pass
    # Include selected env controls
    meta["env"] = {k: os.getenv(k) for k in [
        "STAN_MODE","STAN_METRIC","STAN_NUM_THREADS","STAN_THREADS_PER_CHAIN",
        "STAN_CHAINS","STAN_WARMUP","STAN_SAMPLING",
        "STAN_ADAPT_DELTA","STAN_MAX_TREEDEPTH",
        "PBS_JOBID","PBS_QUEUE","PBS_NP","NCPUS","SLURM_CPUS_PER_TASK"
    ]}
    try:
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

_write_run_metadata(str(OUT / "run_metadata.json"))
