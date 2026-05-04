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
    p.add_argument("--verbose", action="store_true", default=True, help="Print status on console")
    p.add_argument("--max_records", type=int, default=None,
                   help="Process at most this many sequences from a multi-record FASTA.")
    p.add_argument("--norm-output","--norm_output",choices=["y", "n"],default="n",
                help="Also write unified_predictions.csv with window_id,probability_score",)
    args = p.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    model = MiRe2e(device=args.device, pretrained=args.pretrained)

    predictions = []
    record_ids = []
    best_score_by_record = {}
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
                verbose=args.verbose
            )
            for i, idx in enumerate(index):
                score = max(float(scores_5_3[i]), float(scores_3_5[i]))
                predictions.append({
                    "window": idx,
                    "score_5_3": float(scores_5_3[i]),
                    "score_3_5": float(scores_3_5[i]),
                })
                if score > best_score_by_record.get(record.id, 0.0):
                    best_score_by_record[record.id] = score
            record_ids.append(record.id)
            processed += 1
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    output_file = os.path.join(args.output, "predictions.json")
    with open(output_file, "w") as f:
        json.dump({"predictions": predictions}, f, indent=2)

    if args.norm_output == "y":
        unified_file = os.path.join(args.output, "unified_predictions.csv")
        with open(unified_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["window_id", "probability_score"])
            for record_id in record_ids:
                writer.writerow([record_id, best_score_by_record.get(record_id, 0.0)])


if __name__ == "__main__":
    main()
