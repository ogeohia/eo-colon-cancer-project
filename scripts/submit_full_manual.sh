#!/usr/bin/env bash
# Wait for tuning to finish (outputs/fit_status.txt == OK), then submit full run to a long queue
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Defaults (can be overridden)
QUEUE=${QSUB_QUEUE_FULL:-${1:-v1_medium72}}
CPUS=${QSUB_CPUS_FULL:-${QSUB_CPUS:-8}}
MEM=${QSUB_MEM_FULL:-${QSUB_MEM:-16gb}}
WALLTIME=${QSUB_WALLTIME_FULL:-12:00:00}

echo "[full-manual] Target queue=${QUEUE} cpus=${CPUS} mem=${MEM} walltime=${WALLTIME}"

STATUS=outputs/fit_status.txt
echo "[full-manual] Waiting for ${STATUS} to report OK..."
start_ts=$(date +%s)
while true; do
  if [[ -f "${STATUS}" ]]; then
    if grep -q '^OK' "${STATUS}"; then
      echo "[full-manual] Tuning status OK. Proceeding to submit full run."
      break
    fi
  fi
  sleep 20
  now=$(date +%s)
  if (( now - start_ts > 28800 )); then # 8h safety
    echo "[full-manual] Timeout waiting for OK in ${STATUS}. Exiting." >&2
    exit 2
  fi
done

# Build -v env list: force STAN_MODE=full and set safe defaults
declare -a pass_vars
pass_vars=(STAN_MODE STAN_CHAINS STAN_WARMUP STAN_SAMPLING STAN_ADAPT_DELTA STAN_MAX_TREEDEPTH STAN_METRIC STAN_RUN_PRIOR_PRED STAN_SUBSET_N)
export STAN_MODE=full
export STAN_CHAINS=${STAN_CHAINS:-4}
export STAN_WARMUP=${STAN_WARMUP:-1000}
export STAN_SAMPLING=${STAN_SAMPLING:-2000}
export STAN_ADAPT_DELTA=${STAN_ADAPT_DELTA:-0.99}
export STAN_MAX_TREEDEPTH=${STAN_MAX_TREEDEPTH:-15}
export STAN_METRIC=${STAN_METRIC:-dense_e}
export STAN_RUN_PRIOR_PRED=${STAN_RUN_PRIOR_PRED:-1}
export STAN_SUBSET_N=${STAN_SUBSET_N:-0}

# Compose -v KEY=VALUE list
vlist=""
for key in "${pass_vars[@]}"; do
  val="${!key:-}"
  [[ -z "$val" ]] && continue
  val="${val//,/ _}"
  if [[ -z "$vlist" ]]; then vlist="${key}=${val}"; else vlist+=" ,${key}=${val}"; fi
done

# PBS resource args
args=( -l "select=1:ncpus=${CPUS}:mem=${MEM}:mpiprocs=1:ompthreads=${CPUS}" -l "walltime=${WALLTIME}" -q "${QUEUE}" )

echo "[full-manual] qsub ${args[*]} scripts/submit_stan.pbs"
jid=$(qsub -v "${vlist}" "${args[@]}" scripts/submit_stan.pbs)
echo "[full-manual] Submitted job: ${jid}"
echo "${jid}" > outputs/last_jobid_full

# Show initial log tail when it appears
jid_short="${jid%%.*}"
log1="${ROOT}/stan-fit.o${jid}"
log2="${ROOT}/stan-fit.o${jid_short}"
for i in {1..120}; do
  if [[ -s "${log1}" || -s "${log2}" ]]; then
    echo "==== First 60 lines of ${log1} or ${log2} ===="
    [[ -f "${log1}" ]] && head -n 60 "${log1}" || head -n 60 "${log2}" || true
    break
  fi
  sleep 5
done
exit 0
