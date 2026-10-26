import csv
import random
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "dataset" / "filter_candidate_similarity.py"
COMPLEMENT = str.maketrans("ACGU", "UGCA")


def rna(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGU") for _ in range(length))


def write_pool(path, windows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window_id", "sequence"])
        writer.writeheader()
        writer.writerows({"window_id": name, "sequence": sequence} for name, sequence in windows)


def test_removes_precursor_copies_and_near_duplicates_on_either_strand(tmp_path):
    base = rna(200, 1)
    mutated = "".join("A" if i % 40 == 0 and c != "A" else c for i, c in enumerate(base))  # about 97% identity
    precursor = rna(60, 9)
    write_pool(tmp_path / "aaa.csv", [("aaa__c|1-200|+", base), ("aaa__c|501-700|+", rna(200, 2))])
    write_pool(
        tmp_path / "bbb.csv",
        [
            ("bbb__c|1-200|-", base.translate(COMPLEMENT)[::-1]),  # reverse complement of a window in aaa
            ("bbb__c|501-700|+", mutated),  # near-identical to the same window
            ("bbb__c|901-1100|+", rna(70, 3) + precursor + rna(70, 4)),  # contains an annotated precursor
            ("bbb__c|1501-1700|+", rna(200, 5)),
        ],
    )
    (tmp_path / "precursors.fa").write_text(f">p\n{precursor}\n")
    with open(tmp_path / "manifest.tsv", "w") as handle:
        handle.write("species\tpool_csv\tfiltered_csv\n")
        for species in ("aaa", "bbb"):
            handle.write(f"{species}\t{tmp_path / (species + '.csv')}\t{tmp_path / (species + '.filtered.csv')}\n")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(tmp_path / "manifest.tsv"), "--precursor-fasta", str(tmp_path / "precursors.fa"),
         "--work-dir", str(tmp_path / "work"), "--report", str(tmp_path / "report.csv"), "--threads", "2"],
        check=True,
    )
    kept = {
        species: [row["window_id"] for row in csv.DictReader(open(tmp_path / f"{species}.filtered.csv"))]
        for species in ("aaa", "bbb")
    }
    assert kept == {"aaa": ["aaa__c|1-200|+", "aaa__c|501-700|+"], "bbb": ["bbb__c|1501-1700|+"]}
    report = {row["species"]: row for row in csv.DictReader(open(tmp_path / "report.csv"))}
    assert (report["bbb"]["precursor_similar"], report["bbb"]["near_duplicate"], report["bbb"]["kept"]) == ("1", "2", "1")
