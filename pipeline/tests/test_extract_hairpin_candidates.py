import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))

import extract_hairpin_candidates as ehc  # noqa: E402

# A strong, single stem-loop hairpin (61 nt) in DNA letters.
HAIRPIN = "GGGCGCAUCGAUCGGCUAGCUACGAUCGUUUUUCGAUCGUAGCUAGCCGAUCGAUGCGCCC".replace("U", "T")


def random_dna(length, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("AT") for _ in range(length))  # AT-only background folds weakly


def test_parse_rnalfold_keeps_single_stem_loops_and_offsets_leading_dots():
    text = "\n".join(
        [
            ">x",
            "..((((((((((((((((((((((....))))))))))))))))))))))... (-30.50) 11",  # single stem-loop
            "((((....))))((((....)))) (-25.00) 3",  # two hairpin loops
            "((((((((((((((((((((((((....)))))))))))))))))))))))) ( -5.00) 40",  # too weak
            "AUGC",
            " (-99.00)",
        ]
    )
    hairpins = ehc.parse_rnalfold(text, min_length=20, max_length=100, max_mfe=-20)
    # Core starts after the two leading dots; the 4-nt terminal loop starts after 22 paired bases.
    assert hairpins == [(12, 12 + 48, -30.5, 12 + 22, 12 + 26)]


def test_minus_strand_hairpin_maps_back_to_its_genomic_interval():
    left, right = random_dna(400, 1), random_dna(400, 2)
    planted = ehc.reverse_complement(HAIRPIN.replace("T", "U")).replace("U", "T")
    chunk = left + planted + right
    rows = ehc.fold_chunk(("chr", 1000, 1000 + len(chunk), "-", chunk, 100, 50, 100, -20))
    # RNALfold may extend the stem into the flanks; the window is placed by the hairpin midpoint.
    midpoints = [(row[2] + row[3]) / 2 for row in rows if row[1] == "-"]
    expected = 1000 + 400 + len(HAIRPIN) / 2
    assert any(abs(midpoint - expected) <= 2 for midpoint in midpoints)


def test_nested_hairpins_collapse_to_the_most_stable_per_terminal_loop():
    hairpins = [
        ("chr", "+", 1010, 1050, -22.0, 1028, 1032),
        ("chr", "+", 1000, 1060, -30.0, 1028, 1032),  # same loop, longer stem
        ("chr", "-", 1000, 1060, -25.0, 1028, 1032),  # same loop interval on the other strand
    ]
    assert sorted(ehc.collapse_nested(hairpins)) == sorted([hairpins[1], hairpins[2]])


SAMPLE_HAIRPINS = [
    ("chr", "+", 990, 1060, -40.0, 1028, 1032),  # asymmetric stem: the window follows the loop centre (1030)
    ("chr", "-", 1050, 1110, -35.0, 1078, 1082),  # window overlaps the first on the other strand
    ("chr", "+", 3000, 3060, -30.0, 3028, 3032),  # inside an excluded interval
    ("chr", "-", 4000, 4060, -25.0, 4028, 4032),
]


def test_sample_mode_windows_are_centred_on_the_loop_and_never_overlap_across_strands():
    sequence = random_dna(5000, 3).upper()
    intervals = {"chr": ([2950], [3100])}
    rows, skipped = ehc.build_windows(SAMPLE_HAIRPINS, {"chr": sequence}, 200, intervals, 0.1, disjoint=True)
    assert [(row["start"], row["strand"]) for row in rows] == [("930", "+"), ("3930", "-")]
    assert skipped["overlap"] == 1 and skipped["excluded_interval"] == 1
    assert rows[1]["sequence"] == ehc.reverse_complement(sequence[3930:4130].replace("T", "U"))


def test_full_mode_keeps_overlapping_windows():
    sequence = random_dna(5000, 3).upper()
    rows, skipped = ehc.build_windows(SAMPLE_HAIRPINS, {"chr": sequence}, 200, {}, 0.1, disjoint=False)
    assert [row["start"] for row in rows] == ["930", "980", "2930", "3930"]
    assert skipped["overlap"] == 0


def test_repeat_rule_counts_masked_bases_in_the_hairpin_only():
    sequence = random_dna(1000, 4).upper()
    masked_flank = sequence[:400].lower() + sequence[400:]  # half of the 300-500 window masked, hairpin 420-480 clean
    assert not ehc.fails_repeat_rule(masked_flank, 300, 500, 420, 480, 0.1)
    masked_hairpin = sequence[:420] + sequence[420:430].lower() + sequence[430:]  # 10 of 60 hairpin bases
    assert ehc.fails_repeat_rule(masked_hairpin, 300, 500, 420, 480, 0.1)
    n_in_flank = sequence[:310] + "N" + sequence[311:]
    assert ehc.fails_repeat_rule(n_in_flank, 300, 500, 420, 480, 0.1)


def test_window_shift_is_seeded_bounded_and_applied_to_the_window():
    loops = range(1000, 1400, 4)
    shifts = [ehc.window_shift(7, "chr", "+", loop, loop + 4, 10) for loop in loops]
    assert all(-10 <= shift <= 10 for shift in shifts) and len(set(shifts)) > 10
    assert shifts == [ehc.window_shift(7, "chr", "+", loop, loop + 4, 10) for loop in loops]
    assert ehc.window_shift(7, "chr", "+", 1028, 1032, 0) == 0
    sequence = random_dna(5000, 3).upper()
    rows, _ = ehc.build_windows(SAMPLE_HAIRPINS[3:], {"chr": sequence}, 200, {}, 0.1, disjoint=False, max_shift=10, seed=7)
    assert int(rows[0]["start"]) == 3930 + ehc.window_shift(7, "chr", "-", 4028, 4032, 10)
