import gzip
import importlib.util
import json
import os
import tempfile
import unittest


def load_evaluate_module(repo_root):
    module_path = os.path.join(repo_root, "benchmark", "evaluate.py")
    spec = importlib.util.spec_from_file_location("benchmark_evaluate", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.evaluate = load_evaluate_module(self.repo_root)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

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

    def test_parse_mustard_reads_positive_class_bed_scores(self):
        output_dir = os.path.join(self.temp_dir.name, "mustard")
        bed_dir = os.path.join(output_dir, "predict", "static", "results", "bed_tracks")
        os.makedirs(bed_dir)
        path = os.path.join(bed_dir, "predictions.chr1.class_0.bed.gz")
        with gzip.open(path, "wt") as handle:
            handle.write("chr1\t10\t20\trec1\t0.75\t+\n")
            handle.write("chr1\t30\t40\trec2\t0.25\t-\n")

        scores = self.evaluate.parse_mustard(self.evaluate.Path(output_dir))

        self.assertEqual(scores, {"rec1": 0.75, "rec2": 0.25})


if __name__ == "__main__":
    unittest.main()
