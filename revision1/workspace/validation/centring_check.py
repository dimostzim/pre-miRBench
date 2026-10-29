"""Read-only: window and tool-crop placement relative to the hairpin, positives against negatives, in a built dataset."""
import csv
import statistics
import sys
from collections import Counter, defaultdict

dataset = sys.argv[1]
offsets = defaultdict(list)
for row in csv.DictReader(open(f"{dataset}/dataset.csv")):
    label = "pos" if row["label"] == "1" else "neg"
    start, end = int(row["start"]), int(row["end"])
    loop = (int(row["hairpin_loop_start"]) + int(row["hairpin_loop_end"])) // 2
    hairpin = (int(row["hairpin_start"]) + int(row["hairpin_end"])) / 2
    sign = 1 if row["strand"] == "+" else -1
    offsets[(label, "window centre - loop centre")].append(sign * ((start + end) // 2 - loop))
    offsets[(label, "window centre - hairpin midpoint")].append(sign * ((start + end) / 2 - hairpin))
    offsets[(label, "hairpin length")].append(int(row["hairpin_end"]) - int(row["hairpin_start"]))
for (label, name), values in sorted(offsets.items(), key=lambda item: (item[0][1], item[0][0])):
    counts = Counter(values)
    print(f"{name:34s} {label} n={len(values):6d} mean={statistics.mean(values):6.2f} sd={statistics.pstdev(values):5.2f} "
          f"min={min(values)} max={max(values)} share_at_0={counts[0] / len(values):.3f}")
hairpins = {row["record_id"]: row for row in csv.DictReader(open(f"{dataset}/dataset.csv"))}
for tool in ("mire2e", "mirdnn", "dnnpremir"):
    shifts = defaultdict(list)
    for row in csv.DictReader(open(f"{dataset}/tool_inputs/{tool}/metadata.csv")):
        record = hairpins[row["record_id"]]
        middle = (int(record["hairpin_start"]) + int(record["hairpin_end"])) / 2
        shifts[row["label"]].append((int(row["prepared_start"]) + int(row["prepared_end"])) / 2 - middle)
    for label, values in sorted(shifts.items()):
        print(f"{tool} crop centre - hairpin midpoint {label}: n={len(values)} mean={statistics.mean(values):.2f} "
              f"sd={statistics.pstdev(values):.2f} share_at_0={sum(1 for v in values if abs(v) <= 0.5) / len(values):.3f}")
