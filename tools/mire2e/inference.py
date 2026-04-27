#!/usr/bin/env python
import argparse
import csv
import json
import os
import tempfile
from Bio import SeqIO
from miRe2e import MiRe2e


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output", default="results")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pretrained", default="hsa")
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--max_records", type=int, default=None)
    p.add_argument(
        "--norm-output", dest="norm_output", default="y", choices=["y", "n"],
        help="y: unified predictions.csv; n: original predictions.json",
    )
    args = p.parse_args()
    norm_output = args.norm_output.lower() == "y"

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    model = MiRe2e(device=args.device, pretrained=args.pretrained)

    if norm_output:
        # --- unified output: best score per original FASTA record ---
        best_score = {}
        processed = 0
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".fa", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            for record in SeqIO.parse(args.input, "fasta"):
                if args.max_records is not None and processed >= args.max_records:
                    if args.verbose:
                        print(f"Reached max_records={args.max_records}; stopping.")
                    break
                SeqIO.write(record, tmp_path, "fasta")
                scores_5_3, scores_3_5, index = model.predict(
                    tmp_path,
                    length=args.length,
                    step=args.step,
                    batch_size=args.batch_size,
                    verbose=args.verbose,
                )
                for i, idx in enumerate(index):
                    # mire2e appends -<start>-<end> for each sub-window
                    parent_id = idx.rsplit("-", 2)[0] if "-" in idx else idx
                    score = max(float(scores_5_3[i]), float(scores_3_5[i]))
                    if score > best_score.get(parent_id, 0.0):
                        best_score[parent_id] = score
                processed += 1
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        output_file = os.path.join(args.output, "predictions.csv")
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", "probability_score"])
            for record_id, score in best_score.items():
                writer.writerow([record_id, score])

    else:
        # --- original output: per sub-window JSON ---
        # Process each FASTA record individually (model.predict accepts single records).
        predictions = []
        processed = 0
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".fa", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            for record in SeqIO.parse(args.input, "fasta"):
                if args.max_records is not None and processed >= args.max_records:
                    break
                SeqIO.write(record, tmp_path, "fasta")
                scores_5_3, scores_3_5, index = model.predict(
                    tmp_path,
                    length=args.length,
                    step=args.step,
                    batch_size=args.batch_size,
                    verbose=args.verbose,
                )
                for i, idx in enumerate(index):
                    predictions.append({
                        "window": idx,
                        "score_5_3": float(scores_5_3[i]),
                        "score_3_5": float(scores_3_5[i]),
                    })
                processed += 1
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        output_file = os.path.join(args.output, "predictions.json")
        with open(output_file, "w") as f:
            json.dump({"predictions": predictions}, f, indent=2)


if __name__ == "__main__":
    main()
