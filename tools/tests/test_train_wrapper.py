import importlib.util
import os
import tempfile
import unittest


def load_train_module(repo_root):
    module_path = os.path.join(repo_root, "tools", "train.py")
    spec = importlib.util.spec_from_file_location("train_wrapper", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainWrapperTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.train = load_train_module(self.repo_root)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def touch(self, name):
        path = os.path.join(self.temp_dir.name, name)
        with open(path, "w") as handle:
            handle.write("x\n")
        return path

    def test_builds_tool_args_and_generated_inference_configs(self):
        pos_fa = self.touch("positive.fa")
        neg_fa = self.touch("negative.fa")
        pos_bed = self.touch("positive.bed")
        neg_bed = self.touch("negative.bed")
        genome = self.touch("genome.fa")
        cons_dir = os.path.join(self.temp_dir.name, "cons")
        os.mkdir(cons_dir)
        output_dir = os.path.join(self.repo_root, "results", "training", "dummy", "unit")

        configs = {
            "mire2e": {"positive_fasta": pos_fa, "negative_fasta": neg_fa},
            "mirdnn": {"positive_fasta": pos_fa, "negative_fasta": neg_fa},
            "deepmir": {"positive_fasta": pos_fa, "negative_fasta": neg_fa},
            "deepmirgene": {"positive_fasta": pos_fa, "negative_fasta": neg_fa},
            "dnnpremir": {"positive_fasta": pos_fa, "negative_fasta": neg_fa},
            "mustard": {
                "positiveIntervals": pos_bed,
                "negativeIntervals": neg_bed,
                "genome": genome,
                "consDir": cons_dir,
            },
        }

        for tool, config in configs.items():
            with self.subTest(tool=tool):
                mounts = {}
                args = self.train.build_tool_args(tool, self.repo_root, config, output_dir, mounts)
                inference_config = self.train.generated_inference_config(tool, self.repo_root, config, output_dir)

                self.assertEqual(args[0], f"/opt/{tool}/train.py")
                self.assertIn("--output", args)
                self.assertTrue(mounts)
                self.assertTrue(inference_config)

    def test_dnnpremir_accepts_precomputed_csv_inputs(self):
        pos_csv = self.touch("positive.csv")
        neg_csv = self.touch("negative.csv")
        output_dir = os.path.join(self.repo_root, "results", "training", "dummy", "unit")

        mounts = {}
        args = self.train.build_tool_args(
            "dnnpremir",
            self.repo_root,
            {"positive_csv": pos_csv, "negative_csv": neg_csv},
            output_dir,
            mounts,
        )

        self.assertIn("--positive_csv", args)
        self.assertNotIn("--positive_fasta", args)

    def test_external_output_dir_is_writable_mount(self):
        pos_fa = self.touch("positive.fa")
        neg_fa = self.touch("negative.fa")
        output_dir = os.path.join(self.temp_dir.name, "scratch-output", "deepmir", "unit")

        mounts = {}
        args = self.train.build_tool_args(
            "deepmir",
            self.repo_root,
            {"positive_fasta": pos_fa, "negative_fasta": neg_fa},
            output_dir,
            mounts,
        )
        inference_config = self.train.generated_inference_config("deepmir", self.repo_root, {}, output_dir)

        self.assertEqual(args[args.index("--output") + 1], output_dir)
        self.assertIn(os.path.abspath(output_dir), mounts)
        self.assertFalse(mounts[os.path.abspath(output_dir)].endswith(":ro"))
        self.assertEqual(inference_config["model"], f"{output_dir}/model.h5")


if __name__ == "__main__":
    unittest.main()
