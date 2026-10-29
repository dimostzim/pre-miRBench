#!/bin/bash
# 23 Oct (user: "do whatever u want on node 5 and 6"): replace node 5's DeepMir genome queue (5 parallel chunks, four
# genomes; about 4.5 days) by one with 20 parallel chunks for the three fly genomes; chicken is scored on node 6.
# Waits until the five running dme chunks c000-c004 are done, so their work is kept, then stops the old queue
# (run_genome_inference.py, its evaluate.py processes, genome_queue.sh, and the DeepMir containers they started;
# each PID is checked against its command first), then runs the new queue in this terminal.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench-v3
OLD=$W/runs/20261022-2030-genome-deepmir
NEW=$W/runs/20261023-switch-genome-deepmir-flies
E=$W/genome_eval/deepmir_dme
done_chunks() { local n=0; for c in 0 1 2 3 4; do [ -s "$E/genome_dme_c00$c/predictions.csv" ] && n=$((n + 1)); done; echo $n; }
until [ "$(done_chunks)" = 5 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) dme chunks c000-c004 done"
q=$(cat "$OLD/pid")
ps -o args= -p "$q" | grep -q "genome_queue.sh" || { echo "pid $q is not the old genome queue"; exit 1; }
py=$(pgrep -P "$q" -f run_genome_inference.py || true)
ev=""; [ -n "$py" ] && ev=$(pgrep -P "$py" -f evaluate.py || true)
echo "stopping run_genome_inference.py $py, evaluate.py $ev, genome_queue.sh $q"
[ -n "$py" ] && kill -TERM $py
[ -n "$ev" ] && kill -TERM $ev
kill -TERM "$q"
sleep 5
containers=$(docker ps -q --filter ancestor=deepmir:latest)
echo "stopping DeepMir containers: $containers"
[ -n "$containers" ] && docker stop $containers > /dev/null
echo "stopped $(date -u +%FT%TZ) by $W/genome/switch_deepmir_flies.sh; replaced by $NEW (flies, 20 chunks) and node 6 runs/20261023-0630-genome-deepmir-gga-node6 (chicken)" > "$OLD/STOPPED_NOTE.txt"
cd "$W"
export RUN_DIR=$NEW SPECIES="dme dan dya"
exec bash bin/run.sh --dir "$NEW" --name v3-genome-deepmir-flies --repo "$W/repo" --input "$W/genome/genome_queue_species.sh" \
  --expect "$W/genome_metrics/deepmir_dya.csv" -- bash "$W/genome/genome_queue_species.sh" "$W" "$W/repo" deepmir 20 \
  /SCRATCH/dtzim01/conda-envs/premirbench-metrics/bin/python "$W/runs/20261022-1904-train-node5"
