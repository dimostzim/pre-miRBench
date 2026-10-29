"""Read-only check: loop-centred positives against the v2 genome-scan match of the same precursor."""
import csv
import sys

V2 = "/SCRATCH/dtzim01/premirbench/v2/genome_scale"
V = "/SCRATCH/dtzim01/premirbench/v3/validation"
for sp in sys.argv[1:]:
    stats = {r["metric"]: r["value"] for r in csv.DictReader(open(f"{V}/{sp}.positives_shift0.stats.csv"))}
    pos = {r["mirna_id"]: r for r in csv.DictReader(open(f"{V}/{sp}.positives_shift0.csv"))}
    scan = {}
    for r in csv.DictReader(open(f"{V2}/{sp}/candidates.csv")):
        if r["genome_label"] == "positive":
            scan[r["mirna_id"]] = r
    both = [m for m in pos if m in scan]
    same = sum(1 for m in both if (pos[m]["hairpin_loop_start"], pos[m]["hairpin_loop_end"]) == (scan[m]["hairpin_loop_start"], scan[m]["hairpin_loop_end"]))
    same_window = sum(1 for m in both if (pos[m]["start"], pos[m]["sequence"]) == (scan[m]["start"], scan[m]["sequence"]))
    only_scan = [m for m in scan if m not in pos]
    print(f"{sp}: BED {stats['bed_pre_entries']}, kept {stats['positives_written']}, no hairpin {stats['skipped_no_hairpin']}, "
          f"repeat {stats['skipped_repeat']}, boundary {stats['skipped_boundary']} | v2 scan matches {len(scan)}, "
          f"in both {len(both)}, same loop {same}, same window {same_window}, scan-only {len(only_scan)} {only_scan[:5]}")
