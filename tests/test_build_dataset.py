import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load_build_dataset_module(repo_root):
    module_path = os.path.join(repo_root, "pipeline", "build_dataset.py")
    spec = importlib.util.spec_from_file_location("pipeline_build_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildDatasetSplitTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.build_dataset = load_build_dataset_module(self.repo_root)

    def args(self, **overrides):
        values = {
            "heldout_species": "dre",
            "valid_frac": 0.0,
            "valid_heldout_family_frac": 0.0,
            "test_known_species_known_family_frac": 0.0,
            "test_known_species_heldout_family_frac": 0.0,
            "seed": 42,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def positive(self, species, family_id, canonical_sequence, mirna_id=None):
        return {
            "species": species,
            "family_id": family_id,
            "precursor_sequence": canonical_sequence,
            "canonical_100nt_sequence": canonical_sequence,
            "mirna_id": mirna_id or f"{species}-{family_id}_pre",
            "label": "1",
        }

    def negative(self, species, canonical_sequence, window_id=None, score="1.0", hard_round=""):
        return {
            "species": species,
            "window_id": window_id or f"{species}|1-200|+|{canonical_sequence}",
            "chrom": f"{species}__chr1",
            "start": "0",
            "end": "200",
            "strand": "+",
            "sequence": canonical_sequence,
            "structure": "." * len(canonical_sequence),
            "mfe": "-20",
            "mirna_id": "",
            "family_id": "",
            "precursor_sequence": "",
            "canonical_100nt_sequence": canonical_sequence,
            "target_start": "",
            "target_end": "",
            "label": "0",
            "score": score,
            "hard_round": hard_round,
        }

    def test_family_id_parser_collapses_mirgenedb_variants(self):
        parser = self.build_dataset.family_id_from_mirna_id

        self.assertEqual(parser("Hsa-Mir-153-P1_pre"), "Mir-153")
        self.assertEqual(parser("Dre-Mir-430-o1a_pre"), "Mir-430")
        self.assertEqual(parser("Hsa-Let-7-P2b1_pre"), "Let-7")
        self.assertEqual(parser("Dme-Iab-4-as_pre"), "Iab-4")

    def test_precursor_sequence_is_clipped_when_bed_target_exceeds_window(self):
        row = {
            "sequence": "A" * 200,
            "start": "8607",
            "end": "8807",
            "strand": "+",
            "target_start": "8578",
            "target_end": "8837",
            "mirna_id": "Egr-Novel-3_pre",
        }

        self.assertEqual(len(self.build_dataset.precursor_sequence_for_row(row)), 200)

    def test_validation_heldout_family_default_is_train_like(self):
        with patch("sys.argv", ["build_dataset.py"]):
            args = self.build_dataset.parse_args()

        self.assertEqual(args.valid_heldout_family_frac, 0.0)

    def test_duplicate_100nt_positive_keeps_stricter_heldout_split(self):
        rows, excluded = self.build_dataset.assign_positive_splits(
            [
                self.positive("hsa", "Mir-1", "AAAA"),
                self.positive("hsa", "Mir-1", "GGGG"),
                self.positive("dre", "Mir-1", "AAAA"),
                self.positive("dre", "Mir-1", "CCCC"),
            ],
            self.args(),
        )

        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["species"], "hsa")
        self.assertEqual(excluded[0]["canonical_100nt_sequence"], "AAAA")
        self.assertEqual(len({row["canonical_100nt_sequence"] for row in rows}), len(rows))
        self.assertEqual(
            [row["split"] for row in rows if row["species"] == "dre"],
            ["test_heldout_species_known_family", "test_heldout_species_known_family"],
        )

    def test_known_species_known_family_split_leaves_family_in_train(self):
        positives = [
            self.positive("hsa", "Mir-1", "AAAA"),
            self.positive("hsa", "Mir-1", "CCCC"),
            self.positive("hsa", "Mir-2", "GGGG"),
        ]

        rows, excluded = self.build_dataset.assign_positive_splits(
            positives,
            self.args(
                heldout_species="",
                test_known_species_known_family_frac=0.5,
            ),
        )

        self.assertEqual(excluded, [])
        train_families = {row["family_id"] for row in rows if row["split"] == "train"}
        test_rows = [row for row in rows if row["split"] == "test_known_species_known_family"]

        self.assertTrue(test_rows)
        self.assertTrue({row["family_id"] for row in test_rows} <= train_families)
        self.assertFalse(
            {row["canonical_100nt_sequence"] for row in test_rows}
            & {
                row["canonical_100nt_sequence"]
                for row in rows
                if row["split"] == "train"
            }
        )

    def test_select_all_negatives_enforces_unique_100nt_and_ratio(self):
        positives_by_split = {
            "train": [
                {
                    "species": "hsa",
                    "split": "train",
                    "label": "1",
                    "canonical_100nt_sequence": "POS1",
                }
            ],
            "valid": [
                {
                    "species": "hsa",
                    "split": "valid",
                    "label": "1",
                    "canonical_100nt_sequence": "POS2",
                }
            ],
        }
        results = [
            {
                "species": "hsa",
                "species_index": 1,
                "hard_negatives": [
                    self.negative("hsa", "POS1", window_id="conflicts_positive", hard_round="1"),
                    self.negative("hsa", "NEG1", window_id="neg1", hard_round="1"),
                ],
                "scored_negatives": [
                    self.negative("hsa", "NEG1", window_id="neg1_duplicate", score="0.9"),
                    self.negative("hsa", "NEG2", window_id="neg2", score="0.8"),
                    self.negative("hsa", "NEG3", window_id="neg3", score="0.7"),
                ],
            }
        ]

        negatives, issues = self.build_dataset.select_all_negatives(
            results,
            positives_by_split,
            SimpleNamespace(ratio=1.0, seed=42),
        )

        self.assertEqual(issues, {})
        self.assertEqual(len(negatives), 2)
        self.assertEqual(len({row["canonical_100nt_sequence"] for row in negatives}), 2)
        self.assertNotIn("POS1", {row["canonical_100nt_sequence"] for row in negatives})


if __name__ == "__main__":
    unittest.main()
