#!/bin/bash
# Run on the Mac (24 Oct 2026): gather the v3 results inputs into node 6 /SCRATCH/dtzim01/premirbench-v3/results, as v2's
# gather.sh and gather_node5.sh did on node 4. Node 5 files go through the Mac (no node 5 <-> node 6 SSH); node 4 files are
# pushed by node 4 to node 6 directly. Each file is copied once, its md5 checked against the source and recorded in
# results/manifest.tsv; an existing destination must have the source's md5. A missing source (a run still going) is listed,
# so a rerun picks it up later.
set -euo pipefail
W=/SCRATCH/dtzim01/premirbench-v3; OUT=$W/results; V4=/SCRATCH/dtzim01/premirbench
KEY="ssh -i \$HOME/.ssh/id_ed25519_biogemt_internal -o BatchMode=yes"
opts=(-o BatchMode=yes -o ConnectTimeout=30)
stage=$(mktemp -d /tmp/pmb_v3_gather.XXXX)
missing=()
ssh "${opts[@]}" biogemt-6 "mkdir -p $OUT; [ -f $OUT/manifest.tsv ] || printf 'file\tsource\tmd5\n' > $OUT/manifest.tsv"

md5_on() { ssh "${opts[@]}" "$1" "test -f '$2' && md5sum < '$2'" | cut -d' ' -f1 || true; }

copy() { # <source host: 4, 5 or 6> <source path> <destination relative to OUT>
  local host=$1 src=$2 dst=$3 src_md5 dst_md5
  src_md5=$(md5_on "biogemt-$host" "$src")
  [ -n "$src_md5" ] || { missing+=("biogemt-$host:$src"); return 0; }
  dst_md5=$(md5_on biogemt-6 "$OUT/$dst")
  if [ -n "$dst_md5" ]; then
    [ "$dst_md5" = "$src_md5" ] || { echo "ERROR: $dst exists and differs from its source"; exit 1; }
    return 0
  fi
  ssh "${opts[@]}" biogemt-6 "mkdir -p '$(dirname "$OUT/$dst")'"
  case $host in
    6) ssh "${opts[@]}" biogemt-6 "cp -p '$src' '$OUT/$dst'" ;;
    5) rm -f "$stage/file"; scp -q -p "${opts[@]}" "biogemt-5:$src" "$stage/file"; scp -q -p "${opts[@]}" "$stage/file" "biogemt-6:$OUT/$dst" ;;
    4) ssh "${opts[@]}" biogemt-4 "rsync -a -e \"$KEY\" '$src' 'dtzim01@10.200.29.6:$OUT/$dst'" ;;
  esac
  dst_md5=$(md5_on biogemt-6 "$OUT/$dst")
  [ "$dst_md5" = "$src_md5" ] || { echo "ERROR: md5 mismatch after copying $src"; exit 1; }
  ssh "${opts[@]}" biogemt-6 "printf '%s\tbiogemt-%s:%s\t%s\n' '$dst' '$host' '$src' '$src_md5' >> $OUT/manifest.tsv"
  echo "copied $dst"
}

# Benchmark predictions (one model per method; the replicate name is the folder name).
for run in deepmir_v3_rep1 dnnpremir_v3_rep1 mirdnn_v3_rep1 mire2e_v3_rep1 premirbench_model_v2_v3_seed42 premirbench_model_v3_original; do
  copy 5 "$W/evaluation/$run/predictions.csv" "benchmark/predictions/$run/predictions.csv"
done
for run in mustard_v3_rep1 deepmirgene_v3_rep1; do
  copy 6 "$W/evaluation/$run/predictions.csv" "benchmark/predictions/$run/predictions.csv"
done
copy 6 "$W/mirmachine/benchmark_predictions.csv" "benchmark/predictions/mirmachine_retrained_genome_cutoffs/predictions.csv"
for model in baseline_combined_gb baseline_hairpin_filter_lr baseline_kmer_lr baseline_loop_centrality_lr baseline_mfe \
             baseline_mining_rf baseline_paired_profile_lr; do
  copy 4 "$V4/revision1_overlap_audit/results_v3/baseline_predictions/$model/predictions.csv" "benchmark/predictions/$model/predictions.csv"
done

# Genome metrics, comparator summaries and genome curves.
for sp in dme dan dya gga; do
  for tool in mirdnn dnnpremir mire2e premirbench_model_v2; do
    copy 5 "$W/genome_metrics/${tool}_$sp.csv" "genome/metrics/${tool}_$sp.csv"
  done
  copy 5 "$W/genome_metrics/premirbench_model_v3_original_$sp.csv" "genome/metrics/premirbench_model_v3_$sp.csv"
  for tool in mustard deepmirgene; do
    copy 6 "$W/genome_metrics/${tool}_$sp.csv" "genome/metrics/${tool}_$sp.csv"
  done
  copy 6 "$W/mirmachine/comparison/$sp.summary.csv" "genome/comparators/mirmachine_retrained_genome_cutoffs_$sp.summary.csv"
  for kind in pr_curves family_ap discovery_curves; do
    for tool in mirdnn dnnpremir mire2e premirbench_model_v2 premirbench_model_v3; do
      copy 5 "$W/results/genome/$kind/${tool}_$sp.csv" "genome/$kind/${tool}_$sp.csv"
    done
  done
done
for sp in dme dan dya; do
  copy 5 "$W/genome_metrics/deepmir_$sp.csv" "genome/metrics/deepmir_$sp.csv"
  for kind in pr_curves family_ap discovery_curves; do
    copy 5 "$W/results/genome/$kind/deepmir_$sp.csv" "genome/$kind/deepmir_$sp.csv"
  done
done
copy 6 "$W/genome_metrics/deepmir_gga.csv" "genome/metrics/deepmir_gga.csv"
copy 6 "$W/comparators/ncortho/ncortho_training_family_refs_gga.summary.csv" "genome/comparators/ncortho_training_family_refs_gga.summary.csv"

rm -r "$stage"
if [ ${#missing[@]} -gt 0 ]; then printf "missing (not yet produced): %s\n" "${missing[@]}"; fi
echo "$(date -u +%FT%TZ) gather done"
