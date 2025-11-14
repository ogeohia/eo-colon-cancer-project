#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Submit
jid=$(qsub scripts/submit_stan.pbs) || { echo "qsub failed"; qstat 2>/dev/null || true; exit 0; }
echo "Submitted job: $jid"
mkdir -p outputs
echo "$jid" > outputs/last_jobid

jid_short="${jid%%.*}"
log="stan-fit.o${jid}"
alt="stan-fit.o${jid_short}"

# If held, show reason and stop
state=$(qstat -f "$jid" 2>/dev/null | awk '/job_state/ {print $3}')
if [[ "$state" == "H" ]]; then
  echo "Job is held. Details:"
  qstat -f "$jid" | egrep -i 'job_state|hold|comment|queue|Resource_List|account|project'
  exit 0
fi

# Wait for log to appear
for i in {1..120}; do
  [[ -f "$log" ]] && break
  [[ -f "$alt" ]] && log="$alt" && break
  sleep 5
done

if [[ -f "$log" ]]; then
  echo "==== First 80 lines of ${log} ===="; head -n 80 "$log" || true
  echo "==== Now streaming ${log} ===="; tail -n +1 -f "$log"
else
  echo "No log yet. Try: qstat $jid && ls -lt stan-fit.o*"
fi
