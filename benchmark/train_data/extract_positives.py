#!/usr/bin/env python3
"""
Extract positives from MirGeneDB BED + genome FASTA.

For each _pre entry: center a fixed-size window on the miRNA, extract the
strand-correct sequence, apply repeat filter, run RNAfold, write CSV.
"""
import argparse
import csv
import os
import subprocess
import tempfile

COMPLEMENT = str.maketrans('ACGUTRYSWKMBDHVN', 'UGCAAYRSWMKVHDBN')


def reverse_complement(seq):
    return seq.upper().translate(COMPLEMENT)[::-1]


def count_masked(seq):
    return sum(1 for c in seq if c.islower() or c == 'N')


def load_fasta(path):
    genome = {}
    header = None
    chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header is not None:
                    genome[header] = ''.join(chunks)
                header = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        genome[header] = ''.join(chunks)
    return genome


def parse_rnafold_output(fold_path):
    results = {}
    with open(fold_path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            f.readline()  # sequence (use our original, not RNAfold's echo)
            struct_line = f.readline().strip()
            window_id = header[1:].strip()
            left = struct_line.rfind('(')
            right = struct_line.rfind(')')
            if left == -1 or right == -1:
                continue
            try:
                mfe = float(struct_line[left + 1:right].strip())
            except ValueError:
                continue
            structure = struct_line[:left].strip()
            results[window_id] = (structure, mfe)
    return results


parser = argparse.ArgumentParser(description='Extract positives from MirGeneDB BED + genome FASTA')
parser.add_argument('--bed', required=True, help='MirGeneDB BED (filters to _pre entries)')
parser.add_argument('--genome', required=True, help='Genome FASTA (soft-masked)')
parser.add_argument('--window', type=int, default=200, help='Window size (default: 200)')
parser.add_argument('--max-repeat-frac', type=float, default=0.1)
parser.add_argument('--output', required=True)
parser.add_argument('--cpus', type=int, default=8)
parser.add_argument('--chr', dest='chromosomes', default=None,
                    help='Comma-separated chromosomes to keep (default: all)')
args = parser.parse_args()

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

half = args.window // 2
records = []
skipped_missing = skipped_repeat = skipped_boundary = 0

for chrom, start, end, strand, name in mirnas:
    if chrom not in genome:
        skipped_missing += 1
        continue
    chr_seq = genome[chrom]
    center = (start + end) // 2
    win_start = center - half
    win_end = win_start + args.window
    if win_start < 0:
        win_start = 0
        win_end = args.window
    if win_end > len(chr_seq):
        skipped_boundary += 1
        continue
    raw = chr_seq[win_start:win_end]
    if count_masked(raw) / args.window > args.max_repeat_frac:
        skipped_repeat += 1
        continue
    seq = raw.upper().replace('T', 'U')
    if strand == '-':
        seq = reverse_complement(seq)
    window_id = f'{chrom}|{win_start + 1}-{win_end}|{strand}'
    records.append((window_id, chrom, win_start, win_end, strand, seq, name))

print(f'{len(records)} positives kept  ({skipped_missing} chr missing, {skipped_boundary} boundary, {skipped_repeat} repeat-filtered)')

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    fa_path = os.path.join(tmp, 'positives.fa')
    fold_path = os.path.join(tmp, 'positives.fold')
    with open(fa_path, 'w') as f:
        for window_id, _chrom, _ws, _we, _strand, seq, _name in records:
            f.write(f'>{window_id}\n{seq}\n')
    cmd = ['RNAfold', '--noPS', f'--jobs={args.cpus}']
    with open(fa_path) as inf, open(fold_path, 'w') as outf:
        subprocess.check_call(cmd, stdin=inf, stdout=outf)
    fold_results = parse_rnafold_output(fold_path)

written = 0
with open(args.output, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['window_id', 'chrom', 'start', 'end', 'strand', 'sequence', 'structure', 'mfe', 'mirna_id', 'label'])
    for window_id, chrom, win_start, win_end, strand, seq, name in records:
        if window_id not in fold_results:
            continue
        structure, mfe = fold_results[window_id]
        writer.writerow([window_id, chrom, win_start, win_end, strand, seq, structure, mfe, name, 1])
        written += 1

print(f'{written} positives -> {args.output}')
