#!/usr/bin/env python3
import argparse

COMPLEMENT = str.maketrans('ACGUTRYSWKMBDHVN', 'UGCAAYRSWMKVHDBN')


def reverse_complement(seq):
    """Return reverse complement of RNA/DNA sequence."""
    return seq.upper().translate(COMPLEMENT)[::-1]


def count_masked(seq):
    """Count masked bases (lowercase for soft-masking, N for hard-masking)."""
    return sum(1 for c in seq if c.islower() or c == 'N')


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--window", type=int, default=200, help="Window size (default: 200)")
parser.add_argument("--step", type=int, default=50, help="Step size (default: 50)")
parser.add_argument("--output", required=True)
parser.add_argument("--chr", dest="chromosomes", default=None,
                    help="Comma-separated list of chromosomes to process (default: all)")
parser.add_argument("--dna", action="store_true", help="Keep DNA bases (default: convert T→U)")
parser.add_argument("--both_strands", action="store_true", help="Generate windows for both strands")
parser.add_argument("--max_repeat_frac", type=float, default=1.0,
                    help="Max fraction of repeat-masked (lowercase) bases (default: 1.0 = no filter)")
args = parser.parse_args()

# Parse chromosome filter
chr_filter = None
if args.chromosomes:
    chr_filter = set(c.strip() for c in args.chromosomes.split(','))

sequences = []
header = None
chunks = []
with open(args.input) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header:
                sequences.append((header, "".join(chunks)))
            header = line[1:].strip().split()[0]
            chunks = []
        else:
            chunks.append(line)
    if header:
        sequences.append((header, "".join(chunks)))

count = 0
skipped_repeats = 0
skipped_chr = 0
with open(args.output, "w") as out:
    for seq_id, seq in sequences:
        # Filter by chromosome if specified
        if chr_filter and seq_id not in chr_filter:
            skipped_chr += 1
            continue
        seq_clean = seq.replace(" ", "").replace("\t", "")

        strands = [('+', seq_clean)]
        if args.both_strands:
            strands.append(('-', reverse_complement(seq_clean)))

        for strand, strand_seq in strands:
            seq_upper = strand_seq.upper()
            if not args.dna:
                seq_upper = seq_upper.replace("T", "U")

            for start in range(0, len(seq_upper) - args.window + 1, args.step):
                end = start + args.window

                # Check repeat content using original case info
                if args.max_repeat_frac < 1.0:
                    orig_window = strand_seq[start:end]
                    repeat_frac = count_masked(orig_window) / args.window
                    if repeat_frac > args.max_repeat_frac:
                        skipped_repeats += 1
                        continue

                # window_id format: chr|start-end|strand
                window_id = f"{seq_id}|{start + 1}-{end}|{strand}"
                out.write(f">{window_id}\n{seq_upper[start:end]}\n")
                count += 1

print(f"{count} windows -> {args.output}")
if chr_filter:
    print(f"chromosomes: {', '.join(sorted(chr_filter))}")
if skipped_chr > 0:
    print(f"{skipped_chr} chromosomes skipped (not in --chr list)")
if skipped_repeats > 0:
    print(f"{skipped_repeats} windows skipped (repeat fraction > {args.max_repeat_frac})")
