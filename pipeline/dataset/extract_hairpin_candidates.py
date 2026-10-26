#!/usr/bin/env python3
"""
Build hairpin-centred candidate windows from a genome FASTA.

RNALfold finds locally stable structures on both strands. Single stem-loop
hairpins of precursor-like length are kept. RNALfold reports the same stem-loop
several times with progressively extended stems, so hairpins sharing a terminal
loop are collapsed to the most stable (longest) one. A fixed-size window is
centred on the terminal loop of each remaining hairpin (the loop centre does not
move when a stem is extended) and moved by a seeded shift of at most --max-shift
nt; extract_positives.py places positive windows by the same rule. Windows
overlapping an excluded interval (annotated precursors plus a buffer) are
discarded, as are windows whose hairpin has more than --max-repeat-frac
soft-masked or N bases or whose window contains N. Every window is folded with
RNAfold.

Two scan modes:
  sample  fold randomly placed, non-overlapping genome chunks until a target
          number of bases is reached, and select windows greedily in order of
          hairpin MFE so that no two windows overlap on either strand
          (benchmark negative pools)
  full    fold every contig completely and keep one window per terminal loop,
          overlapping windows included (genome-scale evaluation)

Output rows share the dataset CSV schema used by build_dataset.py and add the
selected hairpin interval and MFE.
"""
import argparse
import csv
import math
import os
import random
import re
import subprocess
import tempfile
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

COMPLEMENT = str.maketrans("ACGUTNacgutn", "UGCAANugcaan")
RNALFOLD_LINE = re.compile(r"^([.()]+)\s+\(\s*(-?\d+(?:\.\d+)?)\)\s+(\d+)\s*$")
FIELDNAMES = [
    "window_id",
    "chrom",
    "start",
    "end",
    "strand",
    "sequence",
    "structure",
    "mfe",
    "mirna_id",
    "target_start",
    "target_end",
    "label",
    "hairpin_start",
    "hairpin_end",
    "hairpin_loop_start",
    "hairpin_loop_end",
    "hairpin_mfe",
]


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def count_masked(sequence):
    return sum(1 for char in sequence if char == "N" or char == "n" or char.islower())


def load_fasta(path):
    genome = {}
    name = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    genome[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        genome[name] = "".join(chunks)
    return genome


TERMINAL_LOOP = re.compile(r"\(\.+\)")


def parse_rnalfold(text, min_length, max_length, max_mfe):
    """Single stem-loop hairpins from RNALfold output.

    Returns (start, end, mfe, loop_start, loop_end) in 0-based half-open coordinates of the
    folded sequence; the loop interval covers the unpaired terminal-loop bases.
    """
    hairpins = []
    for line in text.splitlines():
        match = RNALFOLD_LINE.match(line.strip())
        if not match:
            continue
        structure, mfe, position = match.group(1), float(match.group(2)), int(match.group(3))
        core = structure.strip(".")
        if not core or ")(" in core.replace(".", ""):
            continue
        if not min_length <= len(core) <= max_length or mfe > max_mfe:
            continue
        local_start = position - 1 + (len(structure) - len(structure.lstrip(".")))
        loop = TERMINAL_LOOP.search(core)
        hairpins.append((local_start, local_start + len(core), mfe, local_start + loop.start() + 1, local_start + loop.end() - 1))
    return hairpins


def to_genomic(local_start, local_end, chunk_start, chunk_end, strand):
    if strand == "+":
        return chunk_start + local_start, chunk_start + local_end
    return chunk_end - local_end, chunk_end - local_start


def window_bounds(centre_start, centre_end, window):
    """Same centring rule as extract_positives.py: centre = (start + end) // 2 of the centring interval."""
    start = (centre_start + centre_end) // 2 - window // 2
    return start, start + window


def window_shift(seed, chrom, strand, loop_start, loop_end, max_shift):
    """Seeded shift of the window centre, a function of the hairpin alone, so every window gets it the same way."""
    if max_shift == 0:
        return 0
    return random.Random(f"{seed}|{chrom}|{strand}|{loop_start}|{loop_end}").randint(-max_shift, max_shift)


def fails_repeat_rule(sequence, start, end, hairpin_start, hairpin_end, max_masked_frac):
    """Masked bases are counted in the hairpin only, so repeats in the flanks do not remove a window; N is never allowed."""
    masked = count_masked(sequence[hairpin_start:hairpin_end]) / (hairpin_end - hairpin_start)
    return masked > max_masked_frac or "N" in sequence[start:end].upper()


def fold_chunk(job):
    chrom, chunk_start, chunk_end, strand, sequence, span, min_length, max_length, max_mfe = job
    folded = sequence.upper().replace("T", "U")
    if strand == "-":
        folded = reverse_complement(folded)
    output = subprocess.run(
        ["RNALfold", "-L", str(span), "--noconv"],
        input=f">chunk\n{folded}\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for local_start, local_end, mfe, loop_start, loop_end in parse_rnalfold(output, min_length, max_length, max_mfe):
        hairpin_start, hairpin_end = to_genomic(local_start, local_end, chunk_start, chunk_end, strand)
        loop_genomic_start, loop_genomic_end = to_genomic(loop_start, loop_end, chunk_start, chunk_end, strand)
        rows.append((chrom, strand, hairpin_start, hairpin_end, mfe, loop_genomic_start, loop_genomic_end))
    return rows


def collapse_nested(hairpins):
    """One hairpin per (contig, strand, terminal loop): the most stable, i.e. the longest stem extension."""
    best = {}
    for row in hairpins:
        key = (row[0], row[1], row[5], row[6])
        if key not in best or (row[4], row[2] - row[3]) < (best[key][4], best[key][2] - best[key][3]):
            best[key] = row
    return list(best.values())


def load_intervals(path, buffer):
    intervals = defaultdict(list)
    if not path:
        return intervals
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or not parts[3].endswith("_pre"):
                continue
            intervals[parts[0]].append((max(0, int(parts[1]) - buffer), int(parts[2]) + buffer))
    merged = {}
    for chrom, items in intervals.items():
        items.sort()
        out = [list(items[0])]
        for start, end in items[1:]:
            if start <= out[-1][1]:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        merged[chrom] = ([s for s, _ in out], [e for _, e in out])
    return merged


def overlaps_interval(chrom, start, end, intervals):
    if chrom not in intervals:
        return False
    starts, ends = intervals[chrom]
    index = bisect_left(starts, end) - 1
    return index >= 0 and ends[index] > start


def sample_chunks(genome, target_bases, chunk_size, min_contig, seed):
    """Non-overlapping chunks placed uniformly over all contigs of at least min_contig bases."""
    contigs = [(name, len(seq)) for name, seq in genome.items() if len(seq) >= min_contig]
    total = sum(length for _, length in contigs)
    if total <= target_bases:
        return [(name, start, min(start + chunk_size, length)) for name, length in contigs for start in range(0, length, chunk_size)]
    cumulative = []
    running = 0
    for name, length in contigs:
        running += length
        cumulative.append(running)
    rng = random.Random(seed)
    taken = defaultdict(list)
    chunks = []
    sampled = 0
    attempts = 0
    while sampled < target_bases and attempts < 100 * math.ceil(target_bases / chunk_size) + 1000:
        attempts += 1
        point = rng.randrange(total)
        index = bisect_left(cumulative, point + 1)
        name, length = contigs[index]
        offset = point - (cumulative[index] - length)
        start = max(0, min(offset, length - chunk_size))
        end = min(length, start + chunk_size)
        if any(start < e and s < end for s, e in taken[name]):
            continue
        taken[name].append((start, end))
        chunks.append((name, start, end))
        sampled += end - start
    return chunks


def full_chunks(genome, chunk_size, min_contig):
    return [
        (name, start, min(start + chunk_size, len(seq)))
        for name, seq in genome.items()
        if len(seq) >= min_contig
        for start in range(0, len(seq), chunk_size)
    ]


def build_windows(hairpins, genome, window, intervals, max_masked_frac, disjoint, max_shift=0, seed=0):
    """Centre a window on each hairpin; with disjoint=True, select greedily by ascending MFE so windows never overlap."""
    selected = []
    seen = set()
    accepted = defaultdict(lambda: defaultdict(list))
    skipped = {"boundary": 0, "excluded_interval": 0, "masked_or_n": 0, "overlap": 0, "duplicate_window": 0}
    for chrom, strand, hairpin_start, hairpin_end, mfe, loop_start, loop_end in sorted(hairpins, key=lambda row: (row[4], row[0], row[2], row[1])):
        start, end = window_bounds(loop_start, loop_end, window)
        shift = window_shift(seed, chrom, strand, loop_start, loop_end, max_shift)
        start, end = start + shift, end + shift
        sequence = genome[chrom]
        if start < 0 or end > len(sequence):
            skipped["boundary"] += 1
            continue
        if overlaps_interval(chrom, start, end, intervals):
            skipped["excluded_interval"] += 1
            continue
        if (chrom, start, strand) in seen:
            skipped["duplicate_window"] += 1
            continue
        bins = accepted[chrom]
        bin_index = start // window
        if disjoint and any(abs(start - other) < window for b in (bin_index - 1, bin_index, bin_index + 1) for other in bins[b]):
            skipped["overlap"] += 1
            continue
        if fails_repeat_rule(sequence, start, end, hairpin_start, hairpin_end, max_masked_frac):
            skipped["masked_or_n"] += 1
            continue
        raw = sequence[start:end]
        seen.add((chrom, start, strand))
        bins[bin_index].append(start)
        rna = raw.upper().replace("T", "U")
        if strand == "-":
            rna = reverse_complement(rna)
        selected.append(
            {
                "window_id": f"{chrom}|{start + 1}-{end}|{strand}",
                "chrom": chrom,
                "start": str(start),
                "end": str(end),
                "strand": strand,
                "sequence": rna,
                "mirna_id": "",
                "target_start": "",
                "target_end": "",
                "label": "0",
                "hairpin_start": str(hairpin_start),
                "hairpin_end": str(hairpin_end),
                "hairpin_loop_start": str(loop_start),
                "hairpin_loop_end": str(loop_end),
                "hairpin_mfe": f"{mfe:g}",
            }
        )
    return selected, skipped


def fold_windows(rows, cpus):
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta = os.path.join(tmpdir, "windows.fa")
        with open(fasta, "w") as handle:
            for row in rows:
                handle.write(f">{row['window_id']}\n{row['sequence']}\n")
        with open(fasta) as handle:
            output = subprocess.run(
                ["RNAfold", "--noPS", f"--jobs={cpus}"], stdin=handle, capture_output=True, text=True, check=True
            ).stdout
    folds = {}
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for index in range(0, len(lines) - 2, 3):
        if not lines[index].startswith(">"):
            continue
        struct_line = lines[index + 2]
        left = struct_line.rfind("(")
        folds[lines[index][1:]] = (struct_line[:left].strip(), float(struct_line[left + 1:struct_line.rfind(")")]))
    return folds


def parse_args():
    parser = argparse.ArgumentParser(description="Extract hairpin-centred candidate windows from a genome FASTA.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--exclude-bed", default=None, help="BED whose _pre intervals (plus --exclude-buffer) exclude windows.")
    parser.add_argument("--exclude-buffer", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", default=None)
    parser.add_argument("--mode", choices=("sample", "full"), default="sample")
    parser.add_argument("--sample-bases", type=int, default=10_000_000, help="Bases to fold in sample mode.")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--min-contig", type=int, default=1_000)
    parser.add_argument("--span", type=int, default=100, help="RNALfold -L maximum base-pair span.")
    parser.add_argument("--min-hairpin-length", type=int, default=50)
    parser.add_argument("--max-hairpin-length", type=int, default=100)
    parser.add_argument("--max-hairpin-mfe", type=float, default=-20.0)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--max-repeat-frac", type=float, default=0.1)
    parser.add_argument("--max-shift", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    genome = load_fasta(args.genome)
    if args.mode == "sample":
        chunks = sample_chunks(genome, args.sample_bases, args.chunk_size, args.min_contig, args.seed)
    else:
        chunks = full_chunks(genome, args.chunk_size, args.min_contig)
    folded_bases = sum(end - start for _, start, end in chunks)

    # Extend each chunk by one span so hairpins crossing a chunk boundary are found once in full.
    jobs = []
    for chrom, start, end in chunks:
        fold_end = min(len(genome[chrom]), end + args.span)
        for strand in "+-":
            jobs.append(
                (chrom, start, fold_end, strand, genome[chrom][start:fold_end], args.span,
                 args.min_hairpin_length, args.max_hairpin_length, args.max_hairpin_mfe)
            )
    hairpins = set()
    with ProcessPoolExecutor(max_workers=args.cpus) as executor:
        for rows in executor.map(fold_chunk, jobs, chunksize=4):
            hairpins.update(rows)

    collapsed = collapse_nested(hairpins)
    intervals = load_intervals(args.exclude_bed, args.exclude_buffer)
    windows, skipped = build_windows(collapsed, genome, args.window, intervals, args.max_repeat_frac, disjoint=args.mode == "sample",
                                     max_shift=args.max_shift, seed=args.seed)
    folds = fold_windows(windows, args.cpus)
    kept = []
    for row in windows:
        if row["window_id"] not in folds:
            continue
        row["structure"], row["mfe"] = folds[row["window_id"]][0], f"{folds[row['window_id']][1]:g}"
        kept.append(row)
    kept.sort(key=lambda row: (row["chrom"], int(row["start"]), row["strand"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)

    stats = {
        "mode": args.mode,
        "chunks": len(chunks),
        "folded_bases": folded_bases,
        "hairpins": len(hairpins),
        "hairpins_after_loop_collapse": len(collapsed),
        "windows_selected": len(windows),
        "windows_folded": len(kept),
        **{f"skipped_{key}": value for key, value in skipped.items()},
    }
    for key, value in stats.items():
        print(f"{key}: {value}")
    if args.stats_output:
        with open(args.stats_output, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerows(stats.items())


if __name__ == "__main__":
    main()
