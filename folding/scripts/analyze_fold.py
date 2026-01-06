#!/usr/bin/env python3
import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

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

        data.append([window_id, seq, structure, mfe])
        energies.append(mfe)

with open(args.csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["window_id", "sequence", "structure", "mfe"])
    writer.writerows(data)

energies = np.array(energies)
n = len(energies)
std = energies.std(ddof=1)
bandwidth = 1.06 * std * (n ** (-0.2))

mean = energies.mean()
sd = energies.std(ddof=1)
mean_minus_sd = mean - sd
mean_plus_sd = mean + sd

x_min = energies.min()
x_max = energies.max()
span = x_max - x_min
pad = 0.1 * span
xs = np.linspace(x_min - pad, x_max + pad, 200)

diff = (xs[:, None] - energies[None, :]) / bandwidth
ys = np.exp(-0.5 * diff ** 2).sum(axis=1)
ys /= (n * bandwidth * math.sqrt(2.0 * math.pi))

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
