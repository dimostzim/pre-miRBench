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

    def positive(self, species, family_id, precursor_sequence, mirna_id=None):
        return {
            "species": species,
            "family_id": family_id,
            "precursor_sequence": precursor_sequence,
            "mirna_id": mirna_id or f"{species}-{family_id}_pre",
            "label": "1",
        }

    def test_family_id_parser_collapses_mirgenedb_variants(self):
        parser = self.build_dataset.family_id_from_mirna_id

        self.assertEqual(parser("Hsa-Mir-153-P1_pre"), "Mir-153")
        self.assertEqual(parser("Dre-Mir-430-o1a_pre"), "Mir-430")
        self.assertEqual(parser("Hsa-Let-7-P2b1_pre"), "Let-7")
        self.assertEqual(parser("Dme-Iab-4-as_pre"), "Iab-4")

    def test_validation_heldout_family_default_is_train_like(self):
        with patch("sys.argv", ["build_dataset.py"]):
            args = self.build_dataset.parse_args()

        self.assertEqual(args.valid_heldout_family_frac, 0.0)

    def test_heldout_species_exact_precursor_overlap_is_excluded(self):
        rows, excluded = self.build_dataset.assign_positive_splits(
            [
                self.positive("hsa", "Mir-1", "AAAA"),
                self.positive("dre", "Mir-1", "AAAA"),
                self.positive("dre", "Mir-1", "CCCC"),
            ],
            self.args(),
        )

        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["precursor_sequence"], "AAAA")
        self.assertFalse(
            {
                row["precursor_sequence"]
                for row in rows
                if row["split"] != "train"
            }
            & {
                row["precursor_sequence"]
                for row in rows
                if row["split"] == "train"
            }
        )
        self.assertEqual(
            [row["split"] for row in rows if row["species"] == "dre"],
            ["test_heldout_species_known_family"],
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
            {row["precursor_sequence"] for row in test_rows}
            & {
                row["precursor_sequence"]
                for row in rows
                if row["split"] == "train"
            }
        )


if __name__ == "__main__":
    unittest.main()
