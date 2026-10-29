"""Read-only: why precursors matched by the v2 genome scan are not kept as loop-centred positives."""
import csv
import sys

sys.path.insert(0, "/SCRATCH/dtzim01/premirbench/v2/repo/pipeline/dataset")
from extract_hairpin_candidates import collapse_nested, count_masked, fails_repeat_rule, fold_chunk, load_fasta, window_bounds  # noqa: E402
from extract_positives import FOLD_FLANK, match_hairpin  # noqa: E402

V2 = "/SCRATCH/dtzim01/premirbench/v2/genome_scale"
V = "/SCRATCH/dtzim01/premirbench/v3/validation"
for sp in sys.argv[1:]:
    pos = {r["mirna_id"] for r in csv.DictReader(open(f"{V}/{sp}.positives_shift0.csv"))}
    scan = {r["mirna_id"]: r for r in csv.DictReader(open(f"{V2}/{sp}/candidates.csv")) if r["genome_label"] == "positive"}
    bed = {}
    for line in open(f"{V2}/inputs/{sp}.precursors.prefixed.bed"):
        p = line.split("\t")
        bed[p[3]] = (p[0], int(p[1]), int(p[2]), p[5].strip())
    genome = load_fasta(f"{V2}/inputs/{sp}.genome.prefixed.fa")
    for name in [m for m in scan if m not in pos]:
        chrom, start, end, strand = bed[name]
        c = (start + end) // 2
        rs = max(0, c - FOLD_FLANK)
        hp = match_hairpin((chrom, start, end, strand, rs, genome[chrom][rs:c + FOLD_FLANK], 100, 50, 100, -20.0, 15))
        r = scan[name]
        hs, he = int(r["hairpin_start"]), int(r["hairpin_end"])
        ws, we = window_bounds(int(r["hairpin_loop_start"]), int(r["hairpin_loop_end"]), 200)
        frac = count_masked(genome[chrom][hs:he]) / (he - hs)
        print(sp, name, "new match:", None if hp is None else hp[2:], "| v2 hairpin", (hs, he, r["hairpin_loop_start"], r["hairpin_loop_end"]),
              f"v2 hairpin masked {frac:.2f}", "fails rule:", fails_repeat_rule(genome[chrom], ws, we, hs, he, 0.1))
