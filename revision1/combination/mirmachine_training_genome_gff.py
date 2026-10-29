#!/usr/bin/env python3
"""Retrained MirMachine's hits on one training-species genome as GFF, from its own cutoff-fitting search (29 Oct 2026).

Reads the cmsearch table of `mirmachine_retrained.py genome-cutoffs` (genome_cutoffs/<sp>.tbl) and writes the same two
GFFs as `mirmachine_retrained.py genome` writes for a held-out genome: all included hits of at least 50 nt, and those at
or above the family's genome-fitted cutoff, with score = bitscore - cutoff (cutoffs read as in that script).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/SCRATCH/dtzim01/premirbench/v2/repo/pipeline/comparators")
from mirmachine_retrained import included_hits, load_cutoffs  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True)
    parser.add_argument("--work-dir", required=True, help="Directory with the genome-fitted trusted_cutoffs.tsv.")
    parser.add_argument("--all-gff", required=True)
    parser.add_argument("--filtered-gff", required=True)
    args = parser.parse_args()

    cutoff = load_cutoffs(Path(args.work_dir))
    hits = included_hits(args.table)
    called = 0
    with open(args.all_gff, "w") as all_out, open(args.filtered_gff, "w") as filtered_out:
        for hit in sorted(hits, key=lambda h: (h["target"], h["start"])):
            margin = hit["bitscore"] - cutoff[hit["family"]]
            line = "\t".join([hit["target"], "MirMachine_retrained", "microRNA", str(hit["start"] + 1), str(hit["end"]), f"{margin:.2f}",
                              hit["strand"], ".", f"gene_id={hit['family'].upper()}.PRE;bitscore={hit['bitscore']}"]) + "\n"
            all_out.write(line)
            if margin >= 0:
                filtered_out.write(line)
                called += 1
    print(f"{args.table}: {len(hits)} included hits, {called} at or above the cutoff")


if __name__ == "__main__":
    main()
