#!/usr/bin/env python3
import argparse
import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "output")

DINUCLEOTIDES = [a + b for a in 'ACGU' for b in 'ACGU']


def compute_dinucleotide_freq(seq):
    seq = seq.upper().replace('T', 'U')
    counts = {di: 0 for di in DINUCLEOTIDES}
    total = 0
    for i in range(len(seq) - 1):
        di = seq[i:i+2]
        if di in counts:
            counts[di] += 1
            total += 1
    return [counts[di] / total for di in DINUCLEOTIDES]


def compute_sequence_complexity(seq):
    seq = seq.upper().replace('T', 'U')
    n = len(seq)
    counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
    for c in seq:
        if c in counts:
            counts[c] += 1

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * math.log2(p)
    return entropy


def compute_structure_features(struct):
    n = len(struct)
    paired = struct.count('(') + struct.count(')')
    paired_frac = paired / n

    stem_length = 0
    for c in struct:
        if c == '(':
            stem_length += 1
        else:
            break

    loop_size = 0
    current_dots = 0
    for c in struct:
        if c == '.':
            current_dots += 1
            loop_size = max(loop_size, current_dots)
        else:
            current_dots = 0

    bulge_count = 0
    in_stem = False
    for c in struct:
        if c in '()':
            in_stem = True
        elif c == '.' and in_stem:
            bulge_count += 1
            in_stem = False

    return [stem_length, loop_size, bulge_count, paired_frac]


def compute_features(row, mfe_weight=2.0, dinuc_weight=1.0, struct_weight=1.0, complexity_weight=1.0):
    mfe = float(row['mfe']) * mfe_weight
    dinuc = [f * dinuc_weight for f in compute_dinucleotide_freq(row['sequence'])]
    struct_feat = [f * struct_weight for f in compute_structure_features(row['structure'])]
    complexity = compute_sequence_complexity(row['sequence']) * complexity_weight
    return [mfe] + dinuc + struct_feat + [complexity]


def zscore_normalize(features, mean=None, std=None):
    features = np.array(features)
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0, ddof=1)
        std[std == 0] = 1
    normalized = (features - mean) / std
    return normalized, mean, std


def kde(values, xs):
    n = len(values)
    std = values.std(ddof=1)
    if std == 0:
        std = 1
    bandwidth = 1.06 * std * (n ** (-0.2))
    diff = (xs[:, None] - values[None, :]) / bandwidth
    ys = np.exp(-0.5 * diff ** 2).sum(axis=1)
    ys /= (n * bandwidth * math.sqrt(2.0 * math.pi))
    return ys


parser = argparse.ArgumentParser()
parser.add_argument('--positives', required=True)
parser.add_argument('--all_windows', required=True)
parser.add_argument('--negatives', default=os.path.join(OUTPUT_DIR, 'sample_negatives_output', 'negatives.csv'))
parser.add_argument('--balanced', default=os.path.join(OUTPUT_DIR, 'sample_negatives_output', 'balanced.csv'))
parser.add_argument('--plot_dir', default=os.path.join(OUTPUT_DIR, 'sample_negatives_output', 'plots'))
parser.add_argument('--match_strand', action='store_true', default=True)
parser.add_argument('--no_match_strand', action='store_true')
parser.add_argument('--match_chr', action='store_true', default=True)
parser.add_argument('--no_match_chr', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mfe_weight', type=float, default=2.0, help='Weight for MFE feature (default: 2.0)')
parser.add_argument('--dinuc_weight', type=float, default=1.0, help='Weight for dinucleotide features (default: 1.0)')
parser.add_argument('--struct_weight', type=float, default=1.0, help='Weight for structure features (default: 1.0)')
parser.add_argument('--complexity_weight', type=float, default=1.0, help='Weight for complexity feature (default: 1.0)')
args = parser.parse_args()

np.random.seed(args.seed)
os.makedirs(args.plot_dir, exist_ok=True)
negatives_dir = os.path.dirname(args.negatives) or '.'
balanced_dir = os.path.dirname(args.balanced) or '.'
if negatives_dir != '.':
    os.makedirs(negatives_dir, exist_ok=True)
if balanced_dir != '.':
    os.makedirs(balanced_dir, exist_ok=True)

positives = []
positive_window_ids = set()
with open(args.positives) as f:
    reader = csv.DictReader(f)
    for row in reader:
        positives.append(row)
        positive_window_ids.add(row['window_id'])

negatives_pool = []
with open(args.all_windows) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['window_id'] not in positive_window_ids:
            negatives_pool.append(row)

print(f"positives: {len(positives)}")
print(f"negative pool: {len(negatives_pool)}")

do_match_strand = args.match_strand and not args.no_match_strand
do_match_chr = args.match_chr and not args.no_match_chr

def get_strand(row):
    parts = row['window_id'].split('|')
    return parts[2]

def get_chrom(row):
    parts = row['window_id'].split('|')
    return parts[0]

def get_stratum(row):
    key = []
    if do_match_chr:
        key.append(get_chrom(row))
    if do_match_strand:
        key.append(get_strand(row))
    return tuple(key) if key else ('all',)

pos_stratum_counts = defaultdict(int)
for row in positives:
    pos_stratum_counts[get_stratum(row)] += 1

if do_match_chr or do_match_strand:
    print(f"\npositive distribution by {'chr+strand' if do_match_chr and do_match_strand else 'chr' if do_match_chr else 'strand'}:")
    for stratum, count in sorted(pos_stratum_counts.items(), key=lambda x: -x[1]):
        print(f"  {stratum}: {count} ({100*count/len(positives):.1f}%)")

neg_by_stratum = defaultdict(list)
for i, row in enumerate(negatives_pool):
    neg_by_stratum[get_stratum(row)].append(i)

if do_match_chr or do_match_strand:
    print(f"negative pool by stratum: {len(neg_by_stratum)} groups")

print("\ncomputing features...")
pos_features = np.array([compute_features(row, args.mfe_weight, args.dinuc_weight, args.struct_weight, args.complexity_weight) for row in positives])
neg_features = np.array([compute_features(row, args.mfe_weight, args.dinuc_weight, args.struct_weight, args.complexity_weight) for row in negatives_pool])

pos_norm, mean, std = zscore_normalize(pos_features)
neg_norm, _, _ = zscore_normalize(neg_features, mean, std)

print("\nsampling negatives...")

if do_match_chr or do_match_strand:
    sampled_negatives = []
    for stratum in pos_stratum_counts:
        stratum_positives_idx = [i for i, row in enumerate(positives) if get_stratum(row) == stratum]
        stratum_negatives_idx = neg_by_stratum[stratum]

        stratum_neg_norm = neg_norm[stratum_negatives_idx]
        tree = cKDTree(stratum_neg_norm)

        used_indices = set()
        for pos_idx in stratum_positives_idx:
            pos_vec = pos_norm[pos_idx]
            k = min(100, len(stratum_negatives_idx))
            distances, indices = tree.query(pos_vec, k=k)

            for local_idx in indices:
                if local_idx not in used_indices:
                    used_indices.add(local_idx)
                    global_idx = stratum_negatives_idx[local_idx]
                    sampled_negatives.append(negatives_pool[global_idx])
                    break

        print(f"  {stratum}: sampled {len(used_indices)} negatives")

else:
    tree = cKDTree(neg_norm)
    used_indices = set()
    sampled_indices = []

    for i, pos_vec in enumerate(pos_norm):
        k = min(100, len(negatives_pool))
        distances, indices = tree.query(pos_vec, k=k)

        for idx in indices:
            if idx not in used_indices:
                used_indices.add(idx)
                sampled_indices.append(idx)
                break

    sampled_negatives = [negatives_pool[i] for i in sampled_indices]

with open(args.negatives, 'w', newline='') as f:
    fieldnames = list(sampled_negatives[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sampled_negatives)

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

sampled_neg_features = np.array([compute_features(row, args.mfe_weight, args.dinuc_weight, args.struct_weight, args.complexity_weight) for row in sampled_negatives])

FEATURE_NAMES = ['MFE'] + DINUCLEOTIDES + ['stem_len', 'loop_size', 'bulge_count', 'paired_frac', 'complexity']

pos_mfe = pos_features[:, 0]
neg_mfe = sampled_neg_features[:, 0]

x_min = min(pos_mfe.min(), neg_mfe.min())
x_max = max(pos_mfe.max(), neg_mfe.max())
span = x_max - x_min
pad = 0.1 * span
xs = np.linspace(x_min - pad, x_max + pad, 200)

pos_ys = kde(pos_mfe, xs)
neg_ys = kde(neg_mfe, xs)

plt.figure(figsize=(8, 5))
plt.plot(xs, pos_ys, color="tab:blue", linewidth=2, label=f"positive (n={len(positives)})")
plt.plot(xs, neg_ys, color="tab:orange", linewidth=2, label=f"negative (n={len(sampled_negatives)})")
plt.axvline(pos_mfe.mean(), color="tab:blue", linestyle="--", linewidth=1.5, label=f"pos mean: {pos_mfe.mean():.1f}")
plt.axvline(neg_mfe.mean(), color="tab:orange", linestyle="--", linewidth=1.5, label=f"neg mean: {neg_mfe.mean():.1f}")
plt.xlabel("minimum free energy (kcal/mol)")
plt.ylabel("density")
plt.legend(loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{args.plot_dir}/mfe_comparison.png", dpi=150)
plt.close()

pos_dinuc = pos_features[:, 1:17].mean(axis=0)
neg_dinuc = sampled_neg_features[:, 1:17].mean(axis=0)

x = np.arange(16)
width = 0.35

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, pos_dinuc, width, label='positive', color='tab:blue', alpha=0.8)
plt.bar(x + width/2, neg_dinuc, width, label='negative', color='tab:orange', alpha=0.8)
plt.xlabel('dinucleotide')
plt.ylabel('mean frequency')
plt.xticks(x, DINUCLEOTIDES, rotation=45)
plt.legend()
plt.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig(f"{args.plot_dir}/dinucleotide_comparison.png", dpi=150)
plt.close()

fig, axes = plt.subplots(4, 4, figsize=(14, 12))
for i, (ax, di) in enumerate(zip(axes.flat, DINUCLEOTIDES)):
    pos_vals = pos_features[:, i + 1]
    neg_vals = sampled_neg_features[:, i + 1]

    x_min = min(pos_vals.min(), neg_vals.min())
    x_max = max(pos_vals.max(), neg_vals.max())
    if x_max - x_min < 1e-6:
        x_min, x_max = 0, 0.1
    span = x_max - x_min
    pad = 0.1 * span
    xs = np.linspace(x_min - pad, x_max + pad, 100)

    pos_ys = kde(pos_vals, xs)
    neg_ys = kde(neg_vals, xs)

    ax.plot(xs, pos_ys, color="tab:blue", linewidth=1.5, label='pos')
    ax.plot(xs, neg_ys, color="tab:orange", linewidth=1.5, label='neg')
    ax.set_title(di, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)

axes[0, 0].legend(fontsize=8)
plt.suptitle('Dinucleotide frequency distributions', fontsize=12)
plt.tight_layout()
plt.savefig(f"{args.plot_dir}/dinucleotide_kde_grid.png", dpi=150)
plt.close()

struct_names = ['stem_len', 'loop_size', 'bulge_count', 'paired_frac', 'complexity']
pos_struct = pos_features[:, 17:22]
neg_struct = sampled_neg_features[:, 17:22]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (ax, name) in enumerate(zip(axes.flat, struct_names)):
    pos_vals = pos_struct[:, i]
    neg_vals = neg_struct[:, i]

    x_min = min(pos_vals.min(), neg_vals.min())
    x_max = max(pos_vals.max(), neg_vals.max())
    if x_max - x_min < 1e-6:
        x_min, x_max = 0, 1
    span = x_max - x_min
    pad = 0.1 * span
    xs = np.linspace(x_min - pad, x_max + pad, 100)

    pos_ys = kde(pos_vals, xs)
    neg_ys = kde(neg_vals, xs)

    ax.plot(xs, pos_ys, color="tab:blue", linewidth=2, label=f'pos (mean={pos_vals.mean():.2f})')
    ax.plot(xs, neg_ys, color="tab:orange", linewidth=2, label=f'neg (mean={neg_vals.mean():.2f})')
    ax.set_title(name, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

axes.flat[5].axis('off')
plt.suptitle('Structure & complexity distributions', fontsize=12)
plt.tight_layout()
plt.savefig(f"{args.plot_dir}/structure_comparison.png", dpi=150)
plt.close()

if do_match_strand:
    pos_strands = [get_strand(row) for row in positives]
    neg_strands = [get_strand(row) for row in sampled_negatives]

    strands = ['+', '-']
    pos_counts = [pos_strands.count(s) for s in strands]
    neg_counts = [neg_strands.count(s) for s in strands]

    x = np.arange(2)
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, pos_counts, width, label='positive', color='tab:blue', alpha=0.8)
    plt.bar(x + width/2, neg_counts, width, label='negative', color='tab:orange', alpha=0.8)
    plt.xlabel('strand')
    plt.ylabel('count')
    plt.xticks(x, strands)
    plt.legend()
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f"{args.plot_dir}/strand_comparison.png", dpi=150)
    plt.close()

if do_match_chr:
    pos_chroms = [get_chrom(row) for row in positives]
    neg_chroms = [get_chrom(row) for row in sampled_negatives]

    all_chroms = sorted(set(pos_chroms + neg_chroms))
    pos_counts = [pos_chroms.count(c) for c in all_chroms]
    neg_counts = [neg_chroms.count(c) for c in all_chroms]

    x = np.arange(len(all_chroms))
    width = 0.35

    plt.figure(figsize=(max(8, len(all_chroms)), 5))
    plt.bar(x - width/2, pos_counts, width, label='positive', color='tab:blue', alpha=0.8)
    plt.bar(x + width/2, neg_counts, width, label='negative', color='tab:orange', alpha=0.8)
    plt.xlabel('chromosome')
    plt.ylabel('count')
    plt.xticks(x, all_chroms, rotation=45 if len(all_chroms) > 6 else 0)
    plt.legend()
    plt.grid(alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(f"{args.plot_dir}/chr_comparison.png", dpi=150)
    plt.close()

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"positives: {len(positives)}")
print(f"negatives sampled: {len(sampled_negatives)}")
print(f"ratio: {len(sampled_negatives)}/{len(positives)} = {len(sampled_negatives)/len(positives):.2f}")

print(f"\nFeature weights:")
print(f"  MFE: {args.mfe_weight}")
print(f"  Dinucleotides: {args.dinuc_weight}")
print(f"  Structure: {args.struct_weight}")
print(f"  Complexity: {args.complexity_weight}")

print(f"\nMFE - pos mean: {pos_mfe.mean():.2f}, neg mean: {neg_mfe.mean():.2f}, diff: {abs(pos_mfe.mean()-neg_mfe.mean()):.2f}")

dinuc_diffs = np.abs(pos_dinuc - neg_dinuc)
print(f"\nDinucleotide mean abs diff: {dinuc_diffs.mean():.4f}")
print(f"  Max diff: {DINUCLEOTIDES[dinuc_diffs.argmax()]} ({dinuc_diffs.max():.4f})")
print(f"  Min diff: {DINUCLEOTIDES[dinuc_diffs.argmin()]} ({dinuc_diffs.min():.4f})")

struct_diffs = np.abs(pos_struct.mean(axis=0) - neg_struct.mean(axis=0))
print(f"\nStructure feature mean abs diff:")
for name, diff in zip(struct_names, struct_diffs):
    print(f"  {name}: {diff:.4f}")

print(f"\nPlots saved to {args.plot_dir}/")
