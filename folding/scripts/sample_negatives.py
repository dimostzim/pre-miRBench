#!/usr/bin/env python3
import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--positives', required=True)
parser.add_argument('--all_windows', required=True)
parser.add_argument('--negatives', required=True)
parser.add_argument('--balanced', required=True)
parser.add_argument('--plot', required=True)
parser.add_argument('--bins', type=int, default=50)
args = parser.parse_args()

# load positives
positives = []
positive_window_ids = set()
with open(args.positives) as f:
    reader = csv.DictReader(f)
    for row in reader:
        positives.append(row)
        positive_window_ids.add(row['window_id'])

# load all windows and filter negatives
negatives_pool = []
with open(args.all_windows) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['window_id'] not in positive_window_ids:
            negatives_pool.append(row)

# extract mfe values
pos_mfes = np.array([float(row['mfe']) for row in positives])
neg_mfes = np.array([float(row['mfe']) for row in negatives_pool])

# create bins based on positive mfe range
mfe_min = pos_mfes.min()
mfe_max = pos_mfes.max()
bins = np.linspace(mfe_min, mfe_max, args.bins + 1)

# count positives per bin
pos_counts, _ = np.histogram(pos_mfes, bins=bins)

# assign negatives to bins
neg_bin_indices = np.digitize(neg_mfes, bins) - 1
negatives_by_bin = [[] for _ in range(args.bins)]
for i, row in enumerate(negatives_pool):
    bin_idx = neg_bin_indices[i]
    if 0 <= bin_idx < args.bins:
        negatives_by_bin[bin_idx].append(row)

# sample negatives to match positive counts per bin
sampled_negatives = []
for bin_idx in range(args.bins):
    needed = pos_counts[bin_idx]
    available = negatives_by_bin[bin_idx]
    if len(available) >= needed:
        sampled = np.random.choice(len(available), size=needed, replace=False)
        sampled_negatives.extend([available[i] for i in sampled])
    else:
        sampled_negatives.extend(available)

# write negatives
with open(args.negatives, 'w', newline='') as f:
    fieldnames = list(sampled_negatives[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sampled_negatives)

# write balanced dataset
with open(args.balanced, 'w', newline='') as f:
    fieldnames = list(positives[0].keys()) + ['label']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in positives:
        row_copy = row.copy()
        row_copy['label'] = 'positive'
        writer.writerow(row_copy)
    for row in sampled_negatives:
        row_copy = row.copy()
        row_copy['label'] = 'negative'
        writer.writerow(row_copy)

# plot kde comparison
pos_energies = pos_mfes
neg_energies = np.array([float(row['mfe']) for row in sampled_negatives])

def kde(energies, xs):
    n = len(energies)
    std = energies.std(ddof=1)
    bandwidth = 1.06 * std * (n ** (-0.2))
    diff = (xs[:, None] - energies[None, :]) / bandwidth
    ys = np.exp(-0.5 * diff ** 2).sum(axis=1)
    ys /= (n * bandwidth * math.sqrt(2.0 * math.pi))
    return ys

x_min = min(pos_energies.min(), neg_energies.min())
x_max = max(pos_energies.max(), neg_energies.max())
span = x_max - x_min
pad = 0.1 * span
xs = np.linspace(x_min - pad, x_max + pad, 200)

pos_ys = kde(pos_energies, xs)
neg_ys = kde(neg_energies, xs)

pos_mean = pos_energies.mean()
neg_mean = neg_energies.mean()

plt.figure(figsize=(8, 5))
plt.plot(xs, pos_ys, color="tab:blue", linewidth=2, label=f"positive (n={len(positives)})")
plt.plot(xs, neg_ys, color="tab:orange", linewidth=2, label=f"negative (n={len(sampled_negatives)})")
plt.axvline(pos_mean, color="tab:blue", linestyle="--", linewidth=1.5, label=f"pos mean: {pos_mean:.1f}")
plt.axvline(neg_mean, color="tab:orange", linestyle="--", linewidth=1.5, label=f"neg mean: {neg_mean:.1f}")
plt.xlabel("minimum free energy (kcal/mol)")
plt.ylabel("density")
plt.legend(loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(args.plot, dpi=150)

print(f"positives: {len(positives)}")
print(f"negatives sampled: {len(sampled_negatives)}")
print(f"ratio: {len(sampled_negatives)}/{len(positives)} = {len(sampled_negatives)/len(positives):.2f}")
print(f"pos mean mfe: {pos_mean:.2f}")
print(f"neg mean mfe: {neg_mean:.2f}")
