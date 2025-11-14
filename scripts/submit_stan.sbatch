#!/bin/bash
#SBATCH --job-name=stan-fit
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=you@imperial.ac.uk

set -euo pipefail

# activate environment (adjust if you use conda/mamba)
module load Mamba/23.11.0-0 2>/dev/null || true
# If module not available, source your profile: source ~/.bashrc

cd "$HOME/Early-Onset-Colon-Cancer-Trends-CI5plus"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the model script (adjust env name as needed)
mamba run -n colon-cancer-data --no-capture-output python scripts/run_model.py
