import csv
import random
import subprocess
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "dataset"
sys.path.insert(0, str(DATASET))

import extract_hairpin_candidates as ehc  # noqa: E402

# A strong, single stem-loop hairpin (61 nt) in DNA letters.
HAIRPIN = "GGGCGCAUCGAUCGGCUAGCUACGAUCGUUUUUCGAUCGUAGCUAGCCGAUCGAUGCGCCC".replace("U", "T")


def random_dna(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("AT") for _ in range(length))  # AT-only background folds weakly


def run(script, *args):
    subprocess.run([sys.executable, str(DATASET / script), *map(str, args)], check=True, capture_output=True)


def read(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def test_positive_window_is_the_candidate_window_of_the_same_hairpin(tmp_path):
    left, right = random_dna(1500, 1), random_dna(1500, 2)
    planted = ehc.reverse_complement(HAIRPIN.replace("T", "U")).replace("U", "T")
    # Soft-masked flank inside the window: removed by the old window rule, kept by the hairpin rule.
    (tmp_path / "genome.fa").write_text(">chr\n" + left[:1400] + left[1400:].lower() + planted + right + "\n")
    # The annotated precursor is wider on one side than the planted hairpin, so its midpoint is not the loop centre.
    start, end = 1500 - 4, 1500 + len(HAIRPIN) + 10
    (tmp_path / "pre.bed").write_text(f"chr\t{start}\t{end}\tXxx-Mir-1_pre\t.\t-\nchr\t2500\t2560\tXxx-Mir-2_pre\t.\t+\n")
    common = ["--genome", tmp_path / "genome.fa", "--max-shift", 10, "--seed", 7, "--cpus", 1]
    run("extract_positives.py", "--bed", tmp_path / "pre.bed", "--output", tmp_path / "pos.csv",
        "--stats-output", tmp_path / "pos.stats.csv", *common)
    run("extract_hairpin_candidates.py", "--mode", "full", "--output", tmp_path / "cand.csv", *common)

    [positive] = read(tmp_path / "pos.csv")
    stats = {row["metric"]: row["value"] for row in read(tmp_path / "pos.stats.csv")}
    assert positive["mirna_id"] == "Xxx-Mir-1_pre" and stats["skipped_no_hairpin"] == "1"  # Mir-2 lies in background
    loop_start, loop_end = int(positive["hairpin_loop_start"]), int(positive["hairpin_loop_end"])
    shift = ehc.window_shift(7, "chr", "-", loop_start, loop_end, 10)
    assert int(positive["start"]) == (loop_start + loop_end) // 2 - 100 + shift
    candidate = next(row for row in read(tmp_path / "cand.csv")
                     if (row["hairpin_loop_start"], row["hairpin_loop_end"], row["strand"]) == (str(loop_start), str(loop_end), "-"))
    for field in ("start", "end", "sequence", "structure", "hairpin_start", "hairpin_end", "hairpin_mfe"):
        assert positive[field] == candidate[field]
