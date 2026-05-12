#!/usr/bin/env python
import argparse
import os
import random

import numpy as np
import torch as tr
from miRe2e import MiRe2e


def main():
    parser = argparse.ArgumentParser(description="Train the miRe2e classifier stage.")
    parser.add_argument("--positive_fasta", required=True)
    parser.add_argument("--negative_fasta", required=True)
    parser.add_argument("--validation_positive_fasta")
    parser.add_argument("--validation_negative_fasta")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", default="hsa")
    parser.add_argument("--structure_model")
    parser.add_argument("--mfe_model")
    parser.add_argument("--predictor_model")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_split", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tr.manual_seed(args.seed)

    model = MiRe2e(
        device=args.device,
        pretrained=args.pretrained,
        structure_model_file=args.structure_model,
        mfe_model_file=args.mfe_model,
        predictor_model_file=args.predictor_model,
        length=args.length,
    )
    model.fit(
        pos_fname=args.positive_fasta,
        neg_fname=args.negative_fasta,
        val_pos_fname=args.validation_positive_fasta,
        val_neg_fname=args.validation_negative_fasta,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        length=args.length,
    )

    tr.save(model._structure.state_dict(), os.path.join(args.output, "structure.pkl"))
    tr.save(model._mfe.state_dict(), os.path.join(args.output, "mfe.pkl"))
    tr.save(model._predictor.state_dict(), os.path.join(args.output, "predictor.pkl"))
    print(f"Saved miRe2e model files to {args.output}")


if __name__ == "__main__":
    main()
