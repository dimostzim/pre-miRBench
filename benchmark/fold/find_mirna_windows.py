#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def parse_window_id(window_id):
    parts = window_id.split('|')
    chr_part = parts[0]
    coord_part = parts[1]
    start, end = coord_part.split('-')
    return chr_part, int(start) - 1, int(end)


def load_mirnas(bed_file):
    mirnas = []
    with open(bed_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            mirnas.append({
                'chr': parts[0],
                'start': int(parts[1]),
                'end': int(parts[2]),
                'name': parts[3]
            })
    return mirnas


def load_windows_by_chr(csv_file):
    windows_by_chr = defaultdict(list)
    window_index = {}
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            window_id = row['window_id']
            chr_part, start, end = parse_window_id(window_id)
            window_obj = {
                'window_id': window_id,
                'start': start,
                'end': end,
                'row': row
            }
            windows_by_chr[chr_part].append(window_obj)
            window_index[window_id] = window_obj
    for chr_name in windows_by_chr:
        windows_by_chr[chr_name].sort(key=lambda w: w['start'])
    return windows_by_chr, window_index


def find_containing_windows(mirna, windows_by_chr):
    chr_name = mirna['chr']
    containing = []
    for window in windows_by_chr[chr_name]:
        if window['start'] > mirna['end']:
            break
        if window['end'] < mirna['start']:
            continue
        if window['start'] <= mirna['start'] and window['end'] >= mirna['end']:
            containing.append(window)
    return containing


parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True)
parser.add_argument('--bed', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--output_collapsed',
                    help="Output CSV for one window per miRNA (center-closest). Defaults next to --output.")
parser.add_argument('--summary', required=True)
parser.add_argument('--plot', required=True)
args = parser.parse_args()

if not args.output_collapsed:
    args.output_collapsed = os.path.join(os.path.dirname(args.output), "positives_collapsed.csv")
os.makedirs(os.path.dirname(args.output_collapsed), exist_ok=True)

mirnas = load_mirnas(args.bed)
windows_by_chr, window_index = load_windows_by_chr(args.csv)

window_to_mirnas = defaultdict(list)
mirna_window_counts = defaultdict(int)

for mirna in mirnas:
    containing_windows = find_containing_windows(mirna, windows_by_chr)
    for window in containing_windows:
        window_to_mirnas[window['window_id']].append(mirna['name'])
        mirna_window_counts[mirna['name']] += 1

output_rows = []
for window_id, mirna_names in window_to_mirnas.items():
    window = window_index[window_id]
    row = window['row'].copy()
    row['contained_mirnas'] = ';'.join(mirna_names)
    row['num_mirnas'] = len(mirna_names)
    output_rows.append(row)

collapsed_rows = []
for mirna in mirnas:
    containing_windows = find_containing_windows(mirna, windows_by_chr)
    if not containing_windows:
        continue
    mirna_center = (mirna['start'] + mirna['end']) / 2.0
    best_window = None
    best_key = None
    for window in containing_windows:
        window_center = (window['start'] + window['end']) / 2.0
        dist = abs(window_center - mirna_center)
        mfe = float(window['row']['mfe'])
        key = (dist, mfe, window['start'])
        if best_key is None or key < best_key:
            best_key = key
            best_window = window
    if best_window:
        row = best_window['row'].copy()
        window_id = best_window['window_id']
        row['target_mirna'] = mirna['name']
        row['contained_mirnas'] = ';'.join(window_to_mirnas.get(window_id, [mirna['name']]))
        row['num_mirnas'] = len(window_to_mirnas.get(window_id, [mirna['name']]))
        collapsed_rows.append(row)

fieldnames = list(output_rows[0].keys())
with open(args.output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

if collapsed_rows:
    fieldnames = list(collapsed_rows[0].keys())
    with open(args.output_collapsed, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(collapsed_rows)
else:
    print("Warning: no collapsed positives generated.")

with open(args.summary, 'w') as f:
    f.write(f"total windows with mirnas: {len(output_rows)}\n")
    f.write(f"total mirnas found: {len(mirna_window_counts)}/{len(mirnas)}\n\n")
    f.write(f"{'mirna':<40} {'windows':<10}\n")
    f.write("-" * 50 + "\n")
    for mirna_name in sorted(mirna_window_counts.keys()):
        f.write(f"{mirna_name:<40} {mirna_window_counts[mirna_name]:<10}\n")
    counts = list(mirna_window_counts.values())
    f.write(f"\nmean: {sum(counts)/len(counts):.1f}\n")
    f.write(f"min: {min(counts)}\n")
    f.write(f"max: {max(counts)}\n")

energies = np.array([float(row['mfe']) for row in output_rows])
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
plt.axvline(mean_minus_sd, color="gray", linestyle="--", linewidth=1, label=f"mean-sd: {mean_minus_sd:.1f}")
plt.axvline(mean, color="red", linestyle="-", linewidth=1.5, label=f"mean: {mean:.1f}")
plt.axvline(mean_plus_sd, color="gray", linestyle="--", linewidth=1, label=f"mean+sd: {mean_plus_sd:.1f}")
plt.xlabel("minimum free energy (kcal/mol)")
plt.ylabel("density")
plt.legend(loc='upper left')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(args.plot, dpi=150)
