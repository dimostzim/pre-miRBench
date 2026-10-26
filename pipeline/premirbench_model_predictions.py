#!/usr/bin/env python3
"""
Write predictions.csv for one pre-miRBench model bundle in the format of evaluate_replicates.sh
(tool, split, record_id, species, label, score, predicted_label, input_order), so that
summarize_benchmark.py and genome_scale_metrics.py treat the model like the published predictors.

The model package is pre-miRBench_model_v2 unless --model-dir and --tool name another one, such as
pre-miRBench_model_v3. Its evaluate.py scores only the four test splits (eval_predictions_<split>.csv in
--evaluation-dir). This script scores the validation split with the same inference code on CPU,
because validation fixes the thresholds, and maps the exported sample IDs back to dataset record IDs
with the private ID map. Splits and labels must agree with the benchmark dataset.
Run it with the Python of the model environment.
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["tool", "split", "record_id", "species", "label", "score", "predicted_label", "input_order"]
# Exported split name -> benchmark dataset split name.
SPLITS = {
    "validation": "valid",
    "test_known_species_known_family": "test_known_species_known_family",
    "test_known_species_heldout_family": "test_known_species_heldout_family",
    "test_heldout_species_known_family": "test_heldout_species_known_family",
    "test_heldout_species_heldout_family": "test_heldout_species_heldout_family",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Exported dataset (export_agentomics_dataset.py).")
    parser.add_argument("--id-map", required=True, type=Path)
    parser.add_argument("--benchmark-dataset", required=True, type=Path, help="Benchmark dataset.csv.")
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=REPO / "pre-miRBench_model_v2", help="Model package with inference.py.")
    parser.add_argument("--tool", default="premirbench_model_v2", help="Tool name written to predictions.csv.")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.model_dir.resolve()))
    from inference import run_inference
    output = args.evaluation_dir / "predictions.csv"
    if output.exists():
        raise SystemExit(f"{output} exists; refusing to overwrite")
    validation_output = args.evaluation_dir / "eval_predictions_validation.csv"
    if not validation_output.is_file():
        run_inference(args.dataset_dir / "validation" / "input", validation_output, args.artifacts_dir, "cpu")

    record_ids = {(row["export_split"], row["sample_id"]): row["record_id"]
                  for row in csv.DictReader(open(args.id_map), delimiter="\t")}
    dataset = {row["record_id"]: row for row in csv.DictReader(open(args.benchmark_dataset))}
    partial = output.with_name(output.name + ".partial")
    with open(partial, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for export_split, split in SPLITS.items():
            split_dir = args.dataset_dir / export_split
            species = {row["id"]: row["species"] for row in csv.DictReader(open(split_dir / "input" / "samples.tsv"), delimiter="\t")}
            labels = {row["id"]: row["label"] for row in csv.DictReader(open(split_dir / "labels.csv"))}
            predictions = list(csv.DictReader(open(args.evaluation_dir / f"eval_predictions_{export_split}.csv")))
            if sorted(row["id"] for row in predictions) != sorted(labels):
                raise SystemExit(f"{export_split}: prediction IDs differ from labels.csv")
            for order, row in enumerate(predictions):
                record_id = record_ids[(export_split, row["id"])]
                record = dataset[record_id]
                if record["split"] != split or record["label"] != labels[row["id"]]:
                    raise SystemExit(f"{export_split} {row['id']} -> {record_id}: split or label differs from the benchmark dataset")
                writer.writerow([args.tool, split, record_id, species[row["id"]], labels[row["id"]], row["probability_1"], row["prediction"], order])
    partial.rename(output)
    print(f"{args.tool}: {output}")


if __name__ == "__main__":
    main()
