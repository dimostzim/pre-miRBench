#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def parse_window_id(window_id):
    parts = window_id.split('|')
    chrom = parts[0]
    coords = parts[1].split('-')
    start = int(coords[0])
    end = int(coords[1])
    strand = parts[2] if len(parts) > 2 else '+'
    return chrom, start, end, strand


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--csv", required=True)
parser.add_argument("--plot", required=True)
args = parser.parse_args()

data = []
energies = []

with open(args.input) as f:
    while True:
        header = f.readline()
        if not header:
            break
        seq = f.readline().strip()
        struct = f.readline().strip()

        window_id = header[1:].strip()
        left = struct.rfind("(")
        right = struct.rfind(")")
        mfe = float(struct[left + 1:right].strip())
        structure = struct[:left].strip()

        chrom, start, end, strand = parse_window_id(window_id)

        data.append([window_id, chrom, start, end, strand, seq, structure, mfe])
        energies.append(mfe)

with open(args.csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["window_id", "chrom", "start", "end", "strand", "sequence", "structure", "mfe"])
    writer.writerows(data)

energies = np.array(energies)
mean = energies.mean()
sd = energies.std(ddof=1)
mean_minus_sd = mean - sd
mean_plus_sd = mean + sd

x_min = energies.min()
x_max = energies.max()
span = x_max - x_min
pad = 0.1 * span
xs = np.linspace(x_min - pad, x_max + pad, 200)

kde_obj = gaussian_kde(energies, bw_method="silverman")
ys = kde_obj(xs)

plt.figure(figsize=(8, 5))
plt.plot(xs, ys, color="tab:blue", linewidth=2)
plt.axvline(mean_minus_sd, color="gray", linestyle="--", linewidth=1, label=f"Mean-SD: {mean_minus_sd:.1f}")
plt.axvline(mean, color="red", linestyle="-", linewidth=1.5, label=f"Mean: {mean:.1f}")
plt.axvline(mean_plus_sd, color="gray", linestyle="--", linewidth=1, label=f"Mean+SD: {mean_plus_sd:.1f}")
plt.xlabel("Minimum free energy (kcal/mol)")
plt.ylabel("Density")
plt.legend(loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(args.plot, dpi=150)

print(f"{len(data)} windows")
print(f"Mean: {mean:.2f} | SD: {sd:.2f} | Range: [{mean_minus_sd:.2f}, {mean_plus_sd:.2f}]")
print(f"CSV: {args.csv}")
print(f"Plot: {args.plot}")
