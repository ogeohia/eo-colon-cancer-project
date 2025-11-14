#!/usr/bin/env bash
# Submit a tuning run, stream its log, then submit a full run if tuning status is OK
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

tail_pid_file=".tail_pid"

# Sensible defaults if not provided via environment
QSUB_CPUS=${QSUB_CPUS:-8}
QSUB_MEM=${QSUB_MEM:-16gb}
QSUB_WALLTIME_TUNE=${QSUB_WALLTIME_TUNE:-08:00:00}
QSUB_WALLTIME_FULL=${QSUB_WALLTIME_FULL:-12:00:00}

# Warn if requesting long walltime without specifying a long queue
if [[ -z "${QSUB_QUEUE_FULL:-}" && "${QSUB_WALLTIME_FULL}" > "08:00:00" ]]; then
	echo "[WARN] FULL walltime set to ${QSUB_WALLTIME_FULL} but QSUB_QUEUE_FULL not specified."
	echo "       Consider using a long-running queue (e.g., long/vlong) per your cluster policy."
fi

# Default Stan run controls (can be overridden by caller environment)
export STAN_CHAINS=${STAN_CHAINS:-4}
export STAN_WARMUP=${STAN_WARMUP:-1000}
export STAN_SAMPLING=${STAN_SAMPLING:-2000}
export STAN_ADAPT_DELTA=${STAN_ADAPT_DELTA:-0.99}
export STAN_MAX_TREEDEPTH=${STAN_MAX_TREEDEPTH:-15}
export STAN_METRIC=${STAN_METRIC:-dense_e}
export STAN_RUN_PRIOR_PRED=${STAN_RUN_PRIOR_PRED:-1}
export STAN_TUNE_SINGLE_PASS=${STAN_TUNE_SINGLE_PASS:-0}

# Wait for PBS qstat to respond successfully before submitting
wait_for_pbs() {
	local max_sec=${WAIT_PBS_MAX_SEC:-900} # default 15 minutes
	local start=$(date +%s)
	echo "Checking PBS availability..."
	while true; do
		if qstat 1>/dev/null 2>&1; then
			echo "PBS is available. Proceeding."
			return 0
		fi
		local now=$(date +%s)
		local elapsed=$(( now - start ))
		if (( elapsed >= max_sec )); then
			echo "PBS is still unavailable after ${elapsed}s. Exiting." >&2
			return 1
		fi
		printf "...waiting (%ds elapsed) for PBS to come back\r" "$elapsed"
		sleep 5
	done
}

submit_and_stream() {
	local mode="$1"; shift || true
	local qsub_extras=("$@")
	echo "Submitting ${mode} run..."

	mkdir -p outputs
	# Build list of STAN_* env vars to pass through explicitly in addition to STAN_MODE
	local pass_vars=(STAN_MODE)
	# Collect all env names starting with STAN_
	# shellcheck disable=SC2046
	for name in $(env | awk -F= '/^STAN_/ {print $1}'); do
		if [[ " ${pass_vars[*]} " != *" ${name} "* ]]; then
			pass_vars+=("${name}")
		fi
	done
	# Compose -v argument: KEY=VALUE comma-separated
	local vlist=""
	for key in "${pass_vars[@]}"; do
		# Default STAN_MODE to current mode, others use current env value if set
		local val
		if [[ "${key}" == "STAN_MODE" ]]; then
			val="${mode}"
		else
			val="${!key}"
		fi
		# Skip if empty/unset
		[[ -z "${val}" ]] && continue
		# PBS -v does not like commas; replace with underscores
		val="${val//,/ _}"
		if [[ -z "${vlist}" ]]; then
			vlist="${key}=${val}"
		else
			vlist+=" ,${key}=${val}"
		fi
	done

	local jid
	if [[ ${#qsub_extras[@]} -gt 0 ]]; then
		jid=$(qsub -v "${vlist}" "${qsub_extras[@]}" scripts/submit_stan.pbs)
	else
		jid=$(qsub -v "${vlist}" scripts/submit_stan.pbs)
	fi
	echo "Submitted job: ${jid}"
	echo "${jid}" > outputs/last_jobid

	local jid_short="${jid%%.*}"
	local log="${ROOT}/stan-fit.o${jid}"
	local alt="${ROOT}/stan-fit.o${jid_short}"

	# If held, show reason and stop
	local state
	state=$(qstat -f "${jid}" 2>/dev/null | awk '/job_state/ {print $3}') || true
	if [[ "${state:-}" == "H" ]]; then
		echo "Job is held. Details:"
		qstat -f "${jid}" | egrep -i 'job_state|hold|comment|queue|Resource_List|account|project' || true
		return 2
	fi

	# Wait for at least one log file to appear (up to ~45 minutes)
	for i in {1..540}; do
		if [[ -f "${log}" || -f "${alt}" ]]; then
			break
		fi
		sleep 5
	done

	# Prefer showing .pbs log if present and non-empty; otherwise show alt
	if [[ -f "${log}" && -s "${log}" ]]; then
		echo "==== First 80 lines of ${log} ===="; head -n 80 "${log}" || true
	elif [[ -f "${alt}" ]]; then
		echo "==== First 80 lines of ${alt} ===="; head -n 80 "${alt}" || true
	fi

	# Stream both potential log files to avoid missing content due to name differences
	echo "==== Now streaming ${log} and ${alt} ===="
	tail -n +1 -F "${log}" "${alt}" & echo $! > "${tail_pid_file}"

	# If neither exists yet, continue without tailing but still monitor job state
	if [[ ! -f "${log}" && ! -f "${alt}" ]]; then
		echo "No log yet for ${jid}. Will still monitor job state."
	fi

	# Monitor job until it leaves the queue
	local start_ts=$(date +%s)
	local last_print=0
	while true; do
		state=$(qstat -f "${jid}" 2>/dev/null | awk '/job_state/ {print $3}') || true
		if [[ -z "${state:-}" ]]; then
			echo "Job ${jid} finished (no longer in qstat)."
			break
		fi
		# Print a lightweight heartbeat to stderr every ~60s to avoid cluttering tail output
		now_ts=$(date +%s)
		if (( now_ts - last_print >= 60 )); then
			elapsed=$(( now_ts - start_ts ))
			>&2 echo "[${jid}] state=${state} elapsed=${elapsed}s"
			last_print=${now_ts}
		fi
		if [[ "${state}" == "H" ]]; then
			echo "Job held mid-run:"
			qstat -f "${jid}" | egrep -i 'job_state|hold|comment|queue|Resource_List|account|project' || true
			break
		fi
		sleep 10
	done

	# Stop tail if running
	if [[ -f "${tail_pid_file}" ]]; then
		kill "$(cat "${tail_pid_file}")" 2>/dev/null || true
		rm -f "${tail_pid_file}"
	fi

	return 0
}

# 1) Ensure PBS is up
wait_for_pbs || exit 1

# 2) Build PBS args for tune and full (allow separate overrides per phase)
# Common options:
#   QSUB_ARGS      - raw string of extra args, e.g. '-l select=1:ncpus=8:mem=16gb -l walltime=04:00:00'
#   QSUB_CPUS      - number of CPUs per node (e.g. 8)
#   QSUB_MEM       - memory per node (e.g. 16gb)
#   QSUB_SELECT    - number of nodes (default 1)
#   QSUB_WALLTIME  - walltime (e.g. 04:00:00)
#   QSUB_QUEUE     - queue name (e.g. v1_short8)
# Per-phase overrides (take precedence):
#   QSUB_ARGS_TUNE / QSUB_ARGS_FULL
#   QSUB_CPUS_TUNE / QSUB_CPUS_FULL
#   QSUB_MEM_TUNE  / QSUB_MEM_FULL
#   QSUB_SELECT_TUNE / QSUB_SELECT_FULL
#   QSUB_WALLTIME_TUNE / QSUB_WALLTIME_FULL
#   QSUB_QUEUE_TUNE / QSUB_QUEUE_FULL

qsub_args_common=()
if [[ -n "${QSUB_ARGS:-}" ]]; then
	# shellcheck disable=SC2206
	qsub_args_common=( ${QSUB_ARGS} )
fi

qsub_args_tune=( "${qsub_args_common[@]}" )
if [[ -n "${QSUB_ARGS_TUNE:-}" ]]; then
	# shellcheck disable=SC2206
	qsub_args_tune+=( ${QSUB_ARGS_TUNE} )
fi
sel=${QSUB_SELECT_TUNE:-${QSUB_SELECT:-}}
ncpus=${QSUB_CPUS_TUNE:-${QSUB_CPUS:-${QSUB_CPUS}}}
mem=${QSUB_MEM_TUNE:-${QSUB_MEM:-${QSUB_MEM}}}
if [[ -n "${sel}" || -n "${ncpus}" || -n "${mem}" ]]; then
	sel=${sel:-1}; ncpus=${ncpus:-2}; mem=${mem:-12gb}
	# Request a single multi-threaded process: mpiprocs=1 and ompthreads=ncpus
	qsub_args_tune+=( -l "select=${sel}:ncpus=${ncpus}:mem=${mem}:mpiprocs=1:ompthreads=${ncpus}" )
fi
wt=${QSUB_WALLTIME_TUNE:-${QSUB_WALLTIME:-${QSUB_WALLTIME_TUNE}}}
[[ -n "${wt}" ]] && qsub_args_tune+=( -l "walltime=${wt}" )
qq=${QSUB_QUEUE_TUNE:-${QSUB_QUEUE:-}}
[[ -n "${qq}" ]] && qsub_args_tune+=( -q "${qq}" )

qsub_args_full=( "${qsub_args_common[@]}" )
if [[ -n "${QSUB_ARGS_FULL:-}" ]]; then
	# shellcheck disable=SC2206
	qsub_args_full+=( ${QSUB_ARGS_FULL} )
fi
sel=${QSUB_SELECT_FULL:-${QSUB_SELECT:-}}
ncpus=${QSUB_CPUS_FULL:-${QSUB_CPUS:-${QSUB_CPUS}}}
mem=${QSUB_MEM_FULL:-${QSUB_MEM:-${QSUB_MEM}}}
if [[ -n "${sel}" || -n "${ncpus}" || -n "${mem}" ]]; then
	sel=${sel:-1}; ncpus=${ncpus:-2}; mem=${mem:-12gb}
	# Request a single multi-threaded process: mpiprocs=1 and ompthreads=ncpus
	qsub_args_full+=( -l "select=${sel}:ncpus=${ncpus}:mem=${mem}:mpiprocs=1:ompthreads=${ncpus}" )
fi
wt=${QSUB_WALLTIME_FULL:-${QSUB_WALLTIME:-${QSUB_WALLTIME_FULL}}}
[[ -n "${wt}" ]] && qsub_args_full+=( -l "walltime=${wt}" )
qq=${QSUB_QUEUE_FULL:-${QSUB_QUEUE:-}}
[[ -n "${qq}" ]] && qsub_args_full+=( -q "${qq}" )

if [[ ${#qsub_args_tune[@]} -gt 0 ]]; then
	echo "Tune PBS resource overrides: ${qsub_args_tune[*]}"
fi
if [[ ${#qsub_args_full[@]} -gt 0 ]]; then
	echo "Full PBS resource overrides: ${qsub_args_full[*]}"
fi

# Enable quick subset during tune by default
prev_subset=${STAN_SUBSET_N:-}
export STAN_SUBSET_N=${STAN_SUBSET_N:-5000}
submit_and_stream tune "${qsub_args_tune[@]}" || exit 0
if [[ -n "${prev_subset}" ]]; then export STAN_SUBSET_N="${prev_subset}"; else unset STAN_SUBSET_N; fi

# 3) Decide whether to launch full run
if [[ -f outputs/fit_status.txt ]] && grep -q '^OK' outputs/fit_status.txt; then
	echo "Tuning status OK. Submitting full run..."
	# No subset for full by default
	prev_subset=${STAN_SUBSET_N:-}
	export STAN_SUBSET_N=${STAN_SUBSET_N:-0}
	# Optionally override PBS resources for full runs by adding qsub args here, e.g.:
	# submit_and_stream full -l walltime=08:00:00 -l select=1:ncpus=4:mem=16gb
	submit_and_stream full "${qsub_args_full[@]}" || true
	if [[ -n "${prev_subset}" ]]; then export STAN_SUBSET_N="${prev_subset}"; else unset STAN_SUBSET_N; fi
else
	echo "Tuning did not report OK; skipping full run. See outputs/fit_status.txt"
fi

