#!/usr/bin/env python3
import argparse
import subprocess
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--cpus", type=int, default=None)
args = parser.parse_args()

total = 0
with open(args.input) as f:
    for line in f:
        if line.startswith(">"):
            total += 1

cmd = ["RNAfold", "--noPS"]
if args.cpus:
    cmd.append(f"--jobs={args.cpus}")

def run_rnafold():
    with open(args.input) as infile, open(args.output, "w") as outfile:
        subprocess.check_call(cmd, stdin=infile, stdout=outfile)

def monitor_progress():
    while True:
        time.sleep(5)
        try:
            with open(args.output) as f:
                count = sum(1 for line in f if line.startswith(">"))
            pct = (count / total) * 100
            print(f"\rWindows processed: {count:,}/{total:,} ({pct:.1f}%)", end="", flush=True)
            if count >= total:
                break
        except:
            pass

t1 = threading.Thread(target=run_rnafold, daemon=True)
t2 = threading.Thread(target=monitor_progress, daemon=True)

t1.start()
t2.start()
t1.join()
t2.join()

print(f"\nFolded -> {args.output}")
