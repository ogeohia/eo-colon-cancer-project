#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

jid=$(sbatch scripts/submit_stan.sbatch | awk '{print $NF}')
echo "Submitted job: $jid"
echo "$jid" > outputs/last_jobid

log="slurm-${jid}.out"
# wait for logfile or job end
while [[ ! -f "$log" ]]; do
  squeue -j "$jid" >/dev/null 2>&1 || break
  sleep 3
done

if [[ -f "$log" ]]; then
  echo "==== First 80 lines of ${log} ===="
  head -n 80 "$log" || true
  echo "==== Now streaming ${log} ===="
  tail -n +1 -f "$log"
else
  echo "No log file created; check squeue for job status."
  exit 1
fi
