#!/usr/bin/env python
import argparse
import os
import json
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
    args = p.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    model = MiRe2e(device=args.device, pretrained=args.pretrained)
    scores_5_3, scores_3_5, index = model.predict(
        args.input,
        length=args.length,
        step=args.step,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    output_file = os.path.join(args.output, "predictions.json")
    results = {
        "predictions": [
            {"window": idx, "score_5_3": float(scores_5_3[i]), "score_3_5": float(scores_3_5[i])}
            for i, idx in enumerate(index)
        ]
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
