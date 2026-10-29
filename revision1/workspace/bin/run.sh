#!/usr/bin/env bash
# Start one long run in its own run directory: metadata, log, progress and a DONE or FAILED marker.
#
# Usage: run.sh --dir RUN_DIR [--name NAME] [--repo DIR] [--input PATH]... [--expect GLOB]... [--resume] -- COMMAND [ARG...]
#
#   --dir     new directory for this run (refused if it already holds a run, unless --resume)
#   --name    run name shown by runs-status (default: basename of RUN_DIR)
#   --repo    git checkout whose commit and diff are recorded (default: current directory)
#   --input   input file or directory to record (size, mtime, sha256 up to 1 GiB; files up to 1 MiB are copied)
#   --expect  output glob that must match at least one non-empty file, else the run is FAILED
#   --resume  rerun in RUN_DIR after a FAILED or lost run; the previous attempt moves to attempt-N/
#
# COMMAND is one program with its arguments. Wrap pipelines in a script that uses set -euo pipefail.
# Writes in RUN_DIR: run_metadata.json, log.txt, git.diff (if the checkout is dirty), environment.txt
# (if a conda env is active), inputs/, pid, runner.pid, and at the end DONE or FAILED plus issues.txt.
# Lines the run prints as "PROGRESS <done>/<total> <text>" are shown by runs-status.
# If the run writes units.tsv (unit<TAB>status<TAB>...; status ok, failed or skipped) in RUN_DIR,
# any unit not ok or skipped makes the run FAILED.
set -u

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
die() { echo "run.sh: $*" >&2; exit 2; }

run_dir='' name='' repo='' resume=0
inputs=() expects=()
while [ $# -gt 0 ]; do
  case $1 in
    --dir) [ $# -ge 2 ] || usage; run_dir=$2; shift 2 ;;
    --name) [ $# -ge 2 ] || usage; name=$2; shift 2 ;;
    --repo) [ $# -ge 2 ] || usage; repo=$2; shift 2 ;;
    --input) [ $# -ge 2 ] || usage; inputs+=("$2"); shift 2 ;;
    --expect) [ $# -ge 2 ] || usage; expects+=("$2"); shift 2 ;;
    --resume) resume=1; shift ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) die "unknown option '$1'; put the command after --" ;;
  esac
done
[ -n "$run_dir" ] && [ $# -gt 0 ] || usage
command -v python3 >/dev/null || die "python3 not found"

if [ -e "$run_dir/run_metadata.json" ]; then
  [ "$resume" -eq 1 ] || die "$run_dir already holds a run; use a new directory, or --resume after a failure"
  [ -e "$run_dir/DONE" ] && die "$run_dir finished with DONE; use a new directory"
  if [ -s "$run_dir/runner.pid" ] && kill -0 "$(cat "$run_dir/runner.pid")" 2>/dev/null; then
    die "the run in $run_dir is still running"
  fi
  n=1
  while [ -e "$run_dir/attempt-$n" ]; do n=$((n + 1)); done
  mkdir "$run_dir/attempt-$n" || exit 2
  for f in run_metadata.json log.txt FAILED issues.txt git.diff environment.txt pid runner.pid units.tsv; do
    if [ -e "$run_dir/$f" ]; then mv "$run_dir/$f" "$run_dir/attempt-$n/"; fi
  done
elif [ "$resume" -eq 1 ]; then
  die "nothing to resume in $run_dir"
fi

mkdir -p "$run_dir" || exit 2
run_dir=$(cd "$run_dir" && pwd)
name=${name:-$(basename "$run_dir")}
repo=${repo:-$(pwd)}
log="$run_dir/log.txt"
rc_file="$run_dir/.exit_code"
rm -f "$rc_file"
echo $$ > "$run_dir/runner.pid"

# Embedded helper: metadata at start, checks and marker at the end. Python 3 standard library only.
helper() {
  python3 - "$@" <<'PY'
import datetime, glob, hashlib, json, os, re, shutil, socket, subprocess, sys

mode, run_dir = sys.argv[1], sys.argv[2]
meta_path = os.path.join(run_dir, "run_metadata.json")

def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def out(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

if mode == "start":
    name, repo, started = sys.argv[3], sys.argv[4], sys.argv[5]
    n_in = int(sys.argv[6]); inputs = sys.argv[7:7 + n_in]
    rest = sys.argv[7 + n_in:]
    n_ex = int(rest[0]); expects = rest[1:1 + n_ex]; command = rest[1 + n_ex:]

    git = None
    if out(["git", "-C", repo, "rev-parse", "--git-dir"]) is not None:
        diff = out(["git", "-C", repo, "diff", "HEAD"]) or ""
        untracked = (out(["git", "-C", repo, "ls-files", "--others", "--exclude-standard"]) or "").split()
        git = {
            "repo": os.path.abspath(repo),
            "commit": (out(["git", "-C", repo, "rev-parse", "HEAD"]) or "").strip(),
            "branch": (out(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"]) or "").strip(),
            "dirty": bool(diff),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
            "untracked_files": len(untracked),
        }
        if diff:
            with open(os.path.join(run_dir, "git.diff"), "w") as f:
                f.write(diff)

    recorded = []
    for p in inputs:
        entry = {"path": os.path.abspath(p), "exists": os.path.exists(p)}
        if os.path.isfile(p):
            st = os.stat(p)
            entry.update(size=st.st_size, mtime=datetime.datetime.fromtimestamp(
                st.st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            entry["sha256"] = sha256(p) if st.st_size <= 1 << 30 else "skipped: over 1 GiB"
            if st.st_size <= 1 << 20:
                os.makedirs(os.path.join(run_dir, "inputs"), exist_ok=True)
                shutil.copy2(p, os.path.join(run_dir, "inputs", os.path.basename(p)))
                entry["copy"] = os.path.join("inputs", os.path.basename(p))
        elif os.path.isdir(p):
            entry["type"] = "directory"
        recorded.append(entry)

    env = {
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "python3": shutil.which("python3"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpus": os.cpu_count(),
        "gpus": [l for l in (out(["nvidia-smi", "-L"]) or "").splitlines() if l.strip()],
    }
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if env["conda_prefix"] and conda:
        listing = out([conda, "list", "--export", "-p", env["conda_prefix"]])
        if listing:
            with open(os.path.join(run_dir, "environment.txt"), "w") as f:
                f.write(listing)
            env["packages"] = "environment.txt"

    meta = {
        "name": name, "run_dir": run_dir, "host": socket.gethostname(),
        "user": os.environ.get("USER"), "cwd": os.getcwd(), "command": command,
        "started": started, "status": "running", "git": git, "inputs": recorded,
        "expect": expects, "environment": env,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

elif mode == "end":
    rc, signal_name, elapsed = int(sys.argv[3]), sys.argv[4], float(sys.argv[5])
    with open(meta_path) as f:
        meta = json.load(f)
    reasons = []
    if signal_name:
        reasons.append(f"stopped by signal {signal_name}")
    if rc != 0:
        reasons.append(f"exit code {rc}")

    os.chdir(meta["cwd"])
    for pattern in meta.get("expect", []):
        files = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
        if not any(os.path.getsize(p) > 0 for p in files):
            reasons.append(f"expected output missing or empty: {pattern}")

    failed_units, total_units = [], 0
    units_path = os.path.join(run_dir, "units.tsv")
    if os.path.exists(units_path):
        with open(units_path) as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2 or cols[1].lower() == "status":
                    continue
                total_units += 1
                if cols[1].lower() not in ("ok", "skipped"):
                    failed_units.append(cols[0])
        if failed_units:
            reasons.append(f"{len(failed_units)} of {total_units} units not ok in units.tsv")

    issue = re.compile(r"Traceback \(most recent call last\)|\w*(Error|Exception):|\bERROR\b|\bFATAL\b"
                       r"|Killed|Segmentation fault|[Oo]ut of memory|No space left on device"
                       r"|command not found|\bnan\b|\bNaN\b")
    issues, n_issues = [], 0
    log_path = os.path.join(run_dir, "log.txt")
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            for i, line in enumerate(f, 1):
                if not line.startswith("=== ") and issue.search(line):
                    n_issues += 1
                    if len(issues) < 50:
                        issues.append(f"{i}: {line.rstrip()[:300]}")
    if issues:
        with open(os.path.join(run_dir, "issues.txt"), "w") as f:
            f.write("\n".join(issues) + "\n")

    status = "FAILED" if reasons else "DONE"
    ended = now()
    marker = {
        "status": status, "exit_code": rc, "ended": ended, "seconds": int(elapsed),
        "reasons": "; ".join(reasons), "failed_units": len(failed_units),
        "units": total_units, "issue_lines": n_issues,
    }
    with open(os.path.join(run_dir, status), "w") as f:
        f.writelines(f"{k}={v}\n" for k, v in marker.items())
    meta.update(marker)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    lines = [f"=== run {meta['name']}: {status} after {duration(elapsed)} (exit code {rc})"]
    if reasons:
        lines.append("=== reasons: " + "; ".join(reasons))
    if failed_units:
        shown = ", ".join(failed_units[:20]) + (" ..." if len(failed_units) > 20 else "")
        lines.append(f"=== units not ok: {shown}")
    if n_issues:
        lines.append(f"=== log lines to check: {n_issues} (first: {issues[0]}); see issues.txt")
    lines.append(f"=== run directory: {run_dir}")
    text = "\n".join(lines)
    try:
        print(text, flush=True)
    except OSError:
        pass
    try:
        with open(log_path, "a") as f:
            f.write(text + "\n")
    except OSError:
        pass
    sys.exit(0 if status == "DONE" else 1)
PY
}

started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_s=$(date +%s)
helper start "$run_dir" "$name" "$repo" "$started" \
  "${#inputs[@]}" ${inputs[@]+"${inputs[@]}"} "${#expects[@]}" ${expects[@]+"${expects[@]}"} "$@" \
  || die "could not write $run_dir/run_metadata.json"

state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/long-runs"
mkdir -p "$state_dir" && printf '%s\t%s\t%s\t%s\n' "$started" "$(hostname)" "$run_dir" "$name" >> "$state_dir/index.tsv"

signal_name=''
trap 'signal_name=HUP' HUP
trap 'signal_name=INT' INT
trap 'signal_name=TERM' TERM

export PYTHONUNBUFFERED=1
{
  echo "=== run $name started $started on $(hostname) in $(pwd)"
  echo "=== command: $*"
  sh -c 'echo $$ > "$1"; shift; exec "$@"' run.sh "$run_dir/pid" "$@"
  echo $? > "$rc_file"
} 2>&1 | tee -a "$log"

# Finish even if the terminal is gone: ignore further signals and drop a dead terminal.
trap '' HUP INT TERM
if [ "$signal_name" = HUP ] || ! printf '\n' 2>/dev/null; then exec >/dev/null 2>&1; fi
rc=$(cat "$rc_file" 2>/dev/null)
rm -f "$rc_file"
case $signal_name in
  HUP) rc=${rc:-129} ;;
  INT) rc=${rc:-130} ;;
  TERM) rc=${rc:-143} ;;
esac
rc=${rc:-255}
helper end "$run_dir" "$rc" "$signal_name" "$(( $(date +%s) - start_s ))"
final=$?
rm -f "$run_dir/runner.pid"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$final"
