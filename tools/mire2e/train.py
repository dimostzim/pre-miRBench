#!/usr/bin/env python
import argparse
import os
import random
import re
import subprocess
import sys

import numpy as np
import torch as tr
from miRe2e import MiRe2e
from miRe2e.mfe import MFE
from miRe2e.structure import Structure


def require_torch_gpu(device):
    if not str(device).startswith("cuda"):
        raise SystemExit(f"miRe2e training requires a CUDA device; got --device {device}")
    if not tr.cuda.is_available():
        raise SystemExit("miRe2e training requires a visible PyTorch CUDA GPU")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def read_fasta(path):
    name = None
    parts = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if name is not None:
        yield name, "".join(parts)


def parse_rnafold_structure_line(name, line):
    match = re.match(r"^(?P<structure>[().]+)\s+\(\s*(?P<mfe>[-+]?\d+(?:\.\d+)?)\s*\)", line)
    if not match:
        raise ValueError(f"RNAfold produced unexpected structure/MFE line for {name}: {line}")
    return match.group("structure"), f"({match.group('mfe')})"


def fold_sequence(name, sequence):
    rnafold = os.path.join(os.path.dirname(sys.executable), "RNAfold")
    clean_sequence = sequence.upper().replace("T", "U")
    process = subprocess.run(
        [rnafold, "--noPS"],
        input=f">{name}\n{clean_sequence}\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"RNAfold produced unexpected output for {name}: {process.stdout}")
    structure, mfe = parse_rnafold_structure_line(name, lines[2])
    return lines[1], structure, mfe


def write_fold_training_fasta(input_fastas, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(output_path, "w") as output:
        for fasta_path in input_fastas:
            for name, sequence in read_fasta(fasta_path):
                folded_sequence, structure, mfe = fold_sequence(name, sequence)
                output.write(f">{name}\n")
                output.write(f"{folded_sequence}{structure}{mfe}\n")
                count += 1
    if count == 0:
        raise ValueError("No sequences were available to build miRe2e fold training FASTA")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train the miRe2e classifier stage.")
    parser.add_argument("--positive_fasta", required=True)
    parser.add_argument("--negative_fasta", required=True)
    parser.add_argument("--validation_positive_fasta")
    parser.add_argument("--validation_negative_fasta")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", default="hsa")
    parser.add_argument("--train_structure", type=parse_bool, default=True)
    parser.add_argument("--train_mfe", type=parse_bool, default=True)
    parser.add_argument("--structure_training_fasta")
    parser.add_argument("--mfe_training_fasta")
    parser.add_argument("--structure_model")
    parser.add_argument("--mfe_model")
    parser.add_argument("--predictor_model")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--structure_batch_size", type=int, default=None)
    parser.add_argument("--mfe_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--structure_epochs", type=int, default=200)
    parser.add_argument("--mfe_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_split", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    require_torch_gpu(args.device)
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

    generated_folds = os.path.join(args.output, "preprocessed", "fold_training.fa")
    if args.train_structure:
        structure_training_fasta = args.structure_training_fasta or generated_folds
        if not args.structure_training_fasta:
            write_fold_training_fasta([args.positive_fasta, args.negative_fasta], structure_training_fasta)
        model._structure = Structure(device=args.device)
        model._structure.fit(
            structure_training_fasta,
            batch_size=args.structure_batch_size or args.batch_size,
            max_epochs=args.structure_epochs,
            length=args.length,
        )

    if args.train_mfe:
        mfe_training_fasta = args.mfe_training_fasta or generated_folds
        if not args.mfe_training_fasta and not os.path.isfile(mfe_training_fasta):
            write_fold_training_fasta([args.positive_fasta, args.negative_fasta], mfe_training_fasta)
        model._mfe = MFE(device=args.device)
        model._mfe.fit(
            mfe_training_fasta,
            model._structure,
            batch_size=args.mfe_batch_size or args.batch_size,
            max_epochs=args.mfe_epochs,
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
