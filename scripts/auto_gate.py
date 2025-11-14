#!/usr/bin/env python3
import argparse, os, re, subprocess, sys, time, shlex, datetime, pathlib

def run(cmd: str) -> tuple[int, str]:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    return p.returncode, out

def notify(subject: str, body: str):
    # Log-only notifications; PBS email is not supported on this cluster
    now = datetime.datetime.now().isoformat(timespec='seconds')
    print(f"[{now}] {subject}: {body}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-id', required=True)
    # email option removed; notifications are log-only
    ap.add_argument('--gate-hours', type=float, default=6.0)
    ap.add_argument('--workdir', default=str(pathlib.Path.cwd()))
    ap.add_argument('--select', default='1:ncpus=8:mem=16gb')
    ap.add_argument('--walltime', default='48:00:00')
    ap.add_argument('--contingency', type=int, default=0, help='0 initial, 1 first fallback, 2 second fallback')
    return ap.parse_args()

def get_job_info(jobid: str):
    code, out = run(f"qstat -f {shlex.quote(jobid)} | sed -n '1,200p'")
    if code != 0:
        # Return a stub indicating an error; caller can decide to retry
        return {'raw': out, 'state': 'UNK', 'error': True}
    info = {'raw': out}
    m = re.search(r"job_state\s*=\s*(\w)", out)
    if m: info['state'] = m.group(1)
    m = re.search(r"stime\s*=\s*(\w+\s+\w+\s+\d+\s+\d+:\d+:\d+\s+\d{4})", out)
    if m:
        try:
            info['stime'] = time.mktime(time.strptime(m.group(1), "%a %b %d %H:%M:%S %Y"))
        except Exception:
            pass
    return info

def latest_chain_csv(outdir: pathlib.Path) -> pathlib.Path | None:
    # pick the latest _1.csv from no-yrep runs
    cand = sorted(outdir.glob('hierarchical_colon_nb_noyrep-*_[1].csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if cand:
        return cand[0]
    # fallback: any model variant _1.csv
    cand = sorted(outdir.glob('*_[1].csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None

def has_adaptation_terminated(csv_path: pathlib.Path) -> bool:
    try:
        with csv_path.open('r', errors='ignore') as fh:
            for line in fh:
                if line.startswith('# Adaptation terminated'):
                    return True
    except Exception:
        pass
    return False

def submit_job(select: str, walltime: str, env: dict) -> str | None:
    env_kv = ",".join([f"{k}={v}" for k,v in env.items()])
    cmd = f"qsub -v {env_kv} -l select={select} -l walltime={walltime} scripts/submit_stan.pbs"
    code, out = run(cmd)
    if code == 0:
        return out.strip()
    return None

import shutil

if __name__ == '__main__':
    args = parse_args()
    workdir = pathlib.Path(args.workdir)
    os.chdir(workdir)
    outdir = workdir / 'outputs' / 'cmdstan_run'
    outdir.mkdir(parents=True, exist_ok=True)

    jobid = args.job_id
    sent_started = False
    adapted = False
    gate_seconds = int(args.gate_hours * 3600)
    gate_deadline = None

    while True:
        info = get_job_info(jobid)
        now = time.time()
        if not info:
            notify(f"Stan job {jobid} qstat failed", f"No info returned; will retry after sleep.")
            time.sleep(600)
            continue
        if info.get('error'):
            # Transient scheduler issue; do not exit
            raw = (info.get('raw') or '').strip().splitlines()[:2]
            notify(f"Stan job {jobid} qstat error", " | ".join(raw) or "unknown error")
            time.sleep(600)
            continue
        state = info.get('state', '')
        if state == 'R' and not sent_started:
            notify(f"Stan job {jobid} started", f"Job {jobid} is running. Gate={args.gate_hours}h.")
            sent_started = True
            stime = info.get('stime')
            if stime:
                gate_deadline = stime + gate_seconds
            else:
                gate_deadline = now + gate_seconds

        # Look for adaptation termination marker
        csv = latest_chain_csv(outdir)
        if csv and has_adaptation_terminated(csv):
            if not adapted:
                adapted = True
                notify(f"Stan job {jobid} adaptation terminated", f"Detected adaptation terminated in {csv.name}.")

        # Gate check: if not adapted and past deadline, trigger contingency
        if sent_started and not adapted and gate_deadline and now > gate_deadline:
            notify(f"Stan job {jobid} gate triggered", f"Gate {args.gate_hours}h reached without adaptation. Cancelling and submitting contingency.")
            run(f"qdel {jobid}")
            # Build contingency env
            level = args.contingency
            if level == 0:
                env = dict(
                    STAN_MODE='full', STAN_CHAINS='1', STAN_WARMUP='800', STAN_SAMPLING='1500',
                    STAN_ADAPT_DELTA='0.975', STAN_MAX_TREEDEPTH='11', STAN_METRIC='diag_e',
                    STAN_SAVE_YREP='0', STAN_FORCE_COMPILE='1'
                )
                next_level = 1
            else:
                env = dict(
                    STAN_MODE='full', STAN_CHAINS='1', STAN_WARMUP='800', STAN_SAMPLING='1200',
                    STAN_ADAPT_DELTA='0.975', STAN_MAX_TREEDEPTH='11', STAN_METRIC='diag_e',
                    STAN_SAVE_YREP='0', STAN_FORCE_COMPILE='1'
                )
                next_level = level + 1
            new_job = submit_job(args.select, args.walltime, env)
            if new_job:
                notify(f"Stan contingency submitted {new_job}", f"Submitted fallback level {level} from job {jobid}.")
                # Tail-call: exec a new watcher for the new job id
                os.execv(sys.executable, [sys.executable, __file__, '--job-id', new_job.replace('.pbs',''), '--gate-hours', str(args.gate_hours), '--workdir', str(workdir), '--select', args.select, '--walltime', args.walltime, '--contingency', str(next_level)])
            else:
                notify(f"Stan contingency submit failed", f"Failed to submit contingency from job {jobid}.")
                break

        # If job finished (E/F or missing), exit
        if state in ('E','F'):
            notify(f"Stan job {jobid} finished", f"Job {jobid} finished with state {state}.")
            break

        time.sleep(600)
