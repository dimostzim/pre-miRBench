#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--window", type=int, required=True)
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--dna", action="store_true")
args = parser.parse_args()

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
with open(args.output, "w") as out:
    for seq_id, seq in sequences:
        seq = seq.upper().replace(" ", "").replace("\t", "")
        if not args.dna:
            seq = seq.replace("T", "U")
        for start in range(0, len(seq) - args.window + 1, args.step):
            end = start + args.window
            window_id = f"{seq_id}|{start + 1}-{end}"
            out.write(f">{window_id}\n{seq[start:end]}\n")
            count += 1

print(f"{count} windows -> {args.output}")
