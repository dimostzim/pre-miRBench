#!/usr/bin/env python3
"""
Extract positives from MirGeneDB BED + genome FASTA.

Each _pre entry is matched to a hairpin found the same way as the negative and
genome-scale candidates (extract_hairpin_candidates.py): RNALfold on the
precursor's strand around the precursor, single stem-loops of precursor-like
length, one hairpin per terminal loop. The matched hairpin is the one whose
terminal-loop centre is closest to the precursor midpoint, within --tolerance
nt (the labelling rule of prepare_genome_candidates.py); precursors without one
are dropped. The window is then placed, shifted and filtered exactly as a
candidate window (terminal-loop centre, seeded shift, repeat rule on the
hairpin, no N), folded with RNAfold and written as CSV.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_hairpin_candidates import (  # noqa: E402
    collapse_nested,
    fails_repeat_rule,
    fold_chunk,
    fold_windows,
    load_fasta,
    reverse_complement,
    window_bounds,
    window_shift,
)

FOLD_FLANK = 300  # nt folded on each side of the precursor midpoint; holds any 100-nt hairpin whose loop is within the tolerance


def match_hairpin(job):
    """The hairpin on the precursor's strand whose terminal-loop centre is closest to the precursor midpoint."""
    chrom, start, end, strand, region_start, region, span, min_length, max_length, max_mfe, tolerance = job
    hairpins = collapse_nested(fold_chunk(
        (chrom, region_start, region_start + len(region), strand, region, span, min_length, max_length, max_mfe)))
    target = (start + end) / 2
    hits = [row for row in hairpins if abs((row[5] + row[6]) / 2 - target) <= tolerance]
    return min(hits, key=lambda row: (abs((row[5] + row[6]) / 2 - target), row[5])) if hits else None


def parse_args():
    parser = argparse.ArgumentParser(description='Extract positives from MirGeneDB BED + genome FASTA')
    parser.add_argument('--bed', required=True, help='MirGeneDB BED (filters to _pre entries)')
    parser.add_argument('--genome', required=True, help='Genome FASTA (soft-masked)')
    parser.add_argument('--window', type=int, default=200, help='Window size (default: 200)')
    parser.add_argument('--max-repeat-frac', type=float, default=0.1)
    parser.add_argument('--span', type=int, default=100)
    parser.add_argument('--min-hairpin-length', type=int, default=50)
    parser.add_argument('--max-hairpin-length', type=int, default=100)
    parser.add_argument('--max-hairpin-mfe', type=float, default=-20.0)
    parser.add_argument('--tolerance', type=int, default=15)
    parser.add_argument('--max-shift', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stats-output', default=None)
    parser.add_argument('--precursor-fasta', default=None,
                        help='Also write every _pre interval found in the genome (before filters) as RNA FASTA')
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--chr', dest='chromosomes', default=None,
                        help='Comma-separated chromosomes to keep (default: all)')
    return parser.parse_args()


def main():
    args = parse_args()
    chromosome_filter = None
    if args.chromosomes:
        chromosome_filter = {chrom.strip() for chrom in args.chromosomes.split(',') if chrom.strip()}

    print('Loading genome...')
    genome = load_fasta(args.genome)
    print(f'{len(genome)} chromosomes loaded')

    mirnas = []
    with open(args.bed) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            if not parts[3].endswith('_pre'):
                continue
            chrom, start, end, name, strand = parts[0], int(parts[1]), int(parts[2]), parts[3], parts[5]
            if chromosome_filter and chrom not in chromosome_filter:
                continue
            mirnas.append((chrom, start, end, strand, name))
    print(f'{len(mirnas)} _pre entries in BED')

    if args.precursor_fasta:
        os.makedirs(os.path.dirname(os.path.abspath(args.precursor_fasta)), exist_ok=True)
        with open(args.precursor_fasta, 'w') as f:
            for chrom, start, end, strand, name in mirnas:
                if chrom not in genome:
                    continue
                seq = genome[chrom][start:end].upper().replace('T', 'U')
                if strand == '-':
                    seq = reverse_complement(seq)
                f.write(f'>{chrom}|{start}-{end}|{strand}|{name}\n{seq}\n')

    present = [mirna for mirna in mirnas if mirna[0] in genome]
    skipped_missing = len(mirnas) - len(present)
    jobs = []
    for chrom, start, end, strand, _name in present:
        region_start = max(0, (start + end) // 2 - FOLD_FLANK)
        region = genome[chrom][region_start:(start + end) // 2 + FOLD_FLANK]
        jobs.append((chrom, start, end, strand, region_start, region, args.span, args.min_hairpin_length,
                     args.max_hairpin_length, args.max_hairpin_mfe, args.tolerance))
    with ProcessPoolExecutor(max_workers=args.cpus) as executor:
        matches = list(executor.map(match_hairpin, jobs, chunksize=16))

    records = []
    skipped_no_hairpin = skipped_repeat = skipped_boundary = 0
    for (chrom, start, end, strand, name), hairpin in zip(present, matches):
        if hairpin is None:
            skipped_no_hairpin += 1
            continue
        _chrom, _strand, hairpin_start, hairpin_end, hairpin_mfe, loop_start, loop_end = hairpin
        chr_seq = genome[chrom]
        win_start, win_end = window_bounds(loop_start, loop_end, args.window)
        shift = window_shift(args.seed, chrom, strand, loop_start, loop_end, args.max_shift)
        win_start, win_end = win_start + shift, win_end + shift
        if win_start < 0 or win_end > len(chr_seq):
            skipped_boundary += 1
            continue
        if fails_repeat_rule(chr_seq, win_start, win_end, hairpin_start, hairpin_end, args.max_repeat_frac):
            skipped_repeat += 1
            continue
        seq = chr_seq[win_start:win_end].upper().replace('T', 'U')
        if strand == '-':
            seq = reverse_complement(seq)
        records.append({
            'window_id': f'{chrom}|{win_start + 1}-{win_end}|{strand}', 'chrom': chrom, 'start': win_start, 'end': win_end,
            'strand': strand, 'sequence': seq, 'mirna_id': name, 'target_start': start, 'target_end': end, 'label': 1,
            'hairpin_start': hairpin_start, 'hairpin_end': hairpin_end, 'hairpin_loop_start': loop_start,
            'hairpin_loop_end': loop_end, 'hairpin_mfe': f'{hairpin_mfe:g}',
        })

    print(f'{len(records)} positives kept  ({skipped_missing} chr missing, {skipped_no_hairpin} no matching hairpin, '
          f'{skipped_boundary} boundary, {skipped_repeat} repeat-filtered)')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    folds = fold_windows(records, args.cpus)
    fieldnames = ['window_id', 'chrom', 'start', 'end', 'strand', 'sequence', 'structure', 'mfe', 'mirna_id',
                  'target_start', 'target_end', 'label', 'hairpin_start', 'hairpin_end', 'hairpin_loop_start',
                  'hairpin_loop_end', 'hairpin_mfe']
    written = 0
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if record['window_id'] not in folds:
                continue
            structure, mfe = folds[record['window_id']]
            writer.writerow({**record, 'structure': structure, 'mfe': mfe})
            written += 1

    print(f'{written} positives -> {args.output}')

    if args.stats_output:
        with open(args.stats_output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['bed_pre_entries', len(mirnas)])
            writer.writerow(['positives_after_filters', len(records)])
            writer.writerow(['positives_written', written])
            writer.writerow(['skipped_chr_missing', skipped_missing])
            writer.writerow(['skipped_no_hairpin', skipped_no_hairpin])
            writer.writerow(['skipped_boundary', skipped_boundary])
            writer.writerow(['skipped_repeat', skipped_repeat])


if __name__ == '__main__':
    main()
