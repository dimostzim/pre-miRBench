import gzip
import importlib.util
import contextlib
import io
import json
import os
import tempfile
from types import SimpleNamespace
import unittest


def load_evaluate_module(repo_root):
    module_path = os.path.join(repo_root, "pipeline", "evaluate.py")
    spec = importlib.util.spec_from_file_location("pipeline_evaluate", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.evaluate = load_evaluate_module(self.repo_root)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_load_requested_configs_preflights_missing_tools(self):
        class Helpers:
            @staticmethod
            def load_config(path):
                return {"config_path": str(path)}

        training_root = self.evaluate.Path(self.temp_dir.name) / "training"
        present = training_root / "deepmir" / "run1"
        present.mkdir(parents=True)
        (present / "inference_config.yaml").write_text("model: model.h5\n")

        with self.assertRaises(FileNotFoundError) as ctx:
            self.evaluate.load_requested_configs(
                Helpers,
                training_root,
                "run1",
                ["deepmir", "dnnpremir"],
            )
        self.assertIn("dnnpremir", str(ctx.exception))

        with contextlib.redirect_stdout(io.StringIO()):
            configs = self.evaluate.load_requested_configs(
                Helpers,
                training_root,
                "run1",
                ["deepmir", "dnnpremir"],
                allow_missing=True,
            )
        self.assertEqual(sorted(configs), ["deepmir"])

    def test_default_log_path_is_under_output_dir(self):
        args = SimpleNamespace(
            output_dir=os.path.join(self.temp_dir.name, "evaluation"),
            run_name="run1",
            log_file=None,
        )

        eval_dir = self.evaluate.evaluation_output_dir(args)
        self.assertEqual(self.evaluate.evaluation_log_path(args, eval_dir), eval_dir / "run.log.txt")

    def test_tee_writes_to_all_streams(self):
        one = io.StringIO()
        two = io.StringIO()
        tee = self.evaluate.Tee(one, two)

        tee.write("hello\n")

        self.assertEqual(one.getvalue(), "hello\n")
        self.assertEqual(two.getvalue(), "hello\n")

    def test_metric_row_computes_ranking_and_threshold_metrics(self):
        rows = [
            {"label": 1, "score": 0.9, "species": "hsa"},
            {"label": 0, "score": 0.8, "species": "hsa"},
            {"label": 1, "score": 0.7, "species": "hsa"},
            {"label": 0, "score": 0.1, "species": "hsa"},
        ]

        metrics = self.evaluate.metric_row("tool", "test", rows)

        self.assertAlmostEqual(metrics["auprc"], (1.0 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(metrics["auroc"], 0.75)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fn"], 0)

    def test_species_from_record_supports_dataset_record_ids(self):
        self.assertEqual(self.evaluate.species_from_record("hsa_pos_000001"), "hsa")
        self.assertEqual(self.evaluate.species_from_record("dme_neg_000123"), "dme")
        self.assertEqual(self.evaluate.species_from_record("hsa__chr1:1-200"), "hsa")

    def test_write_auprc_bar_plot(self):
        path = self.evaluate.Path(self.temp_dir.name) / "auprc_by_tool.svg"
        wrote = self.evaluate.write_auprc_bar_plot(
            path,
            [
                {"tool": "deepmir", "split": "test_chrom", "auprc": 0.81234},
                {"tool": "deepmir", "split": "test_species", "auprc": 0.71234},
                {"tool": "mirdnn", "split": "test_chrom", "auprc": 0.9},
                {"tool": "mirdnn", "split": "test_species", "auprc": 0.8},
            ],
        )

        text = path.read_text()
        self.assertTrue(wrote)
        self.assertIn("<svg", text)
        self.assertIn("AUPRC by Tool", text)
        self.assertIn(">Tool</text>", text)
        self.assertIn("Test set", text)
        self.assertIn("Left-out set", text)
        self.assertIn("deepmir", text)
        self.assertIn("mirdnn", text)
        self.assertIn("0.812", text)

    def test_write_auprc_bar_plot_png(self):
        path = self.evaluate.Path(self.temp_dir.name) / "auprc_by_tool.png"
        wrote = self.evaluate.write_auprc_bar_plot_png(
            path,
            [
                {"tool": "deepmir", "split": "test_chrom", "auprc": 0.81234},
                {"tool": "deepmir", "split": "test_species", "auprc": 0.71234},
            ],
        )
        if not wrote:
            self.skipTest("matplotlib is not available")

        with open(path, "rb") as handle:
            self.assertEqual(handle.read(8), b"\x89PNG\r\n\x1a\n")

    def test_parse_mire2e_aggregates_record_windows(self):
        output_dir = os.path.join(self.temp_dir.name, "mire2e")
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "predictions.json"), "w") as handle:
            json.dump(
                {
                    "predictions": [
                        {"record_id": "a", "score_5_3": 0.1, "score_3_5": 0.2},
                        {"record_id": "a", "score_5_3": 0.7, "score_3_5": 0.4},
                        {"record_id": "b", "score_5_3": 0.3, "score_3_5": 0.5},
                    ]
                },
                handle,
            )

        scores = self.evaluate.parse_mire2e(
            self.evaluate.Path(output_dir),
            [{"record_id": "a"}, {"record_id": "b"}],
        )

        self.assertEqual(scores, {"a": 0.7, "b": 0.5})

    def test_parse_mustard_reads_configured_positive_class_bed_scores(self):
        output_dir = os.path.join(self.temp_dir.name, "mustard")
        bed_dir = os.path.join(output_dir, "predict", "static", "results", "bed_tracks")
        os.makedirs(bed_dir)
        negative_path = os.path.join(bed_dir, "predictions.chr1.class_0.bed.gz")
        with gzip.open(negative_path, "wt") as handle:
            handle.write("chr1\t10\t20\trec1\t0.25\t+\n")
            handle.write("chr1\t30\t40\trec2\t0.75\t-\n")
        positive_path = os.path.join(bed_dir, "predictions.chr1.class_1.bed.gz")
        with gzip.open(positive_path, "wt") as handle:
            handle.write("chr1\t10\t20\trec1\t0.75\t+\n")
            handle.write("chr1\t30\t40\trec2\t0.25\t-\n")

        scores = self.evaluate.parse_mustard(self.evaluate.Path(output_dir), positive_class_index=1)

        self.assertEqual(scores, {"rec1": 0.75, "rec2": 0.25})


if __name__ == "__main__":
    unittest.main()
