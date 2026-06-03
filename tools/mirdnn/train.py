#!/usr/bin/env python
import argparse
import os
import random
import subprocess
import sys

import numpy as np
import torch as tr
import torch.utils.data as dt


def require_torch_gpu(device):
    if not str(device).startswith("cuda"):
        raise SystemExit(f"mirDNN training requires a CUDA device; got --device {device}")
    if not tr.cuda.is_available():
        raise SystemExit("mirDNN training requires a visible PyTorch CUDA GPU")


def fold_fasta(fasta_path, output_path):
    conda_bin = os.path.dirname(sys.executable)
    cmd = [os.path.join(conda_bin, "RNAfold"), "--noPS"]
    with open(fasta_path) as input_handle, open(output_path, "w") as output_handle:
        subprocess.check_call(cmd, stdin=input_handle, stdout=output_handle)


def parse_optional_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def average_precision(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.size == 0 or np.sum(labels) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / (np.arange(sorted_labels.size) + 1)
    return float(np.sum(precision * sorted_labels) / np.sum(sorted_labels))


def load_training_components():
    mirdnn_src = "/opt/mirdnn/mirdnn_src" if os.path.isdir("/opt/mirdnn/mirdnn_src") else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mirdnn_src"
    )
    sys.path.insert(0, mirdnn_src)
    from src.fold_dataset import FoldDataset
    from src.model import mirDNN
    from src.sampler import ImbalancedDatasetSampler

    return FoldDataset, mirDNN, ImbalancedDatasetSampler


class ModelParameters:
    def __init__(self, args):
        self.device = tr.device(args.device)
        self.seq_len = args.seq_length
        self.width = 64
        self.n_resnets = 3
        self.kernel_size = 3
        self.focal_loss = parse_optional_bool(args.focal_loss, True)


def set_random_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    tr.manual_seed(seed)
    if tr.device(device).type == "cuda":
        tr.backends.cudnn.deterministic = True
        tr.backends.cudnn.benchmark = False


def split_dataset(dataset, valid_prop, seed):
    valid_size = int(valid_prop * len(dataset))
    valid_size = max(1, valid_size)
    if valid_size >= len(dataset):
        raise ValueError("Validation split leaves no training examples")
    generator = tr.Generator().manual_seed(seed)
    return dt.random_split(dataset, (len(dataset) - valid_size, valid_size), generator=generator)


def make_train_loader(dataset, batch_size, upsample, sampler_factory):
    if upsample:
        sampler = sampler_factory(dataset, max_imbalance=1.0, num_samples=8 * batch_size)
        return dt.DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    return dt.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def evaluate_auprc(model, valid_loader):
    model.eval()
    scores = []
    labels = []
    with tr.no_grad():
        for x, v, y in valid_loader:
            prediction = model(x, v).detach().cpu().view(-1).numpy()
            scores.extend(prediction.tolist())
            labels.extend(y.detach().cpu().view(-1).numpy().tolist())
    model.train()
    return average_precision(labels, scores)


def train_model(args, train_dataset, valid_dataset, model_factory, sampler_factory):
    params = ModelParameters(args)
    model = model_factory(params)
    model.train()

    model_path = os.path.join(args.output, "model.pmt")
    log_path = os.path.join(args.output, "model.log")
    train_loader = make_train_loader(
        train_dataset,
        args.batch_size,
        parse_optional_bool(args.upsample, False),
        sampler_factory,
    )
    valid_loader = dt.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True)

    best_valid_auprc = -1.0
    last_improvement = 0
    with open(log_path, "w") as log:
        log.write("epoch\ttrainLoss\tvalidAUPRC\tlast_imp\n")
        for epoch in range(args.epochs):
            batch_losses = []
            for x, v, y in train_loader:
                batch_losses.append(model.train_step(x, v, y))
            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            valid_auprc = evaluate_auprc(model, valid_loader)

            if valid_auprc > best_valid_auprc:
                best_valid_auprc = valid_auprc
                last_improvement = 0
                model.save(model_path)
            else:
                last_improvement += 1

            log.write(f"{epoch}\t{train_loss:.4f}\t{valid_auprc:.6f}\t{last_improvement}\n")
            log.flush()
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"valid_auprc={valid_auprc:.6f} last_imp={last_improvement}"
            )
            if args.early_stop > 0 and last_improvement >= args.early_stop:
                break

    if not os.path.isfile(model_path):
        model.save(model_path)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train a mirDNN model.")
    parser.add_argument("--positive_fasta", required=True)
    parser.add_argument("--negative_fasta", required=True)
    parser.add_argument("--validation_positive_fasta")
    parser.add_argument("--validation_negative_fasta")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seq_length", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--early_stop", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--valid_prop", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--upsample")
    parser.add_argument("--focal_loss")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    require_torch_gpu(args.device)
    if args.early_stopping_patience is not None:
        args.early_stop = args.early_stopping_patience
    set_random_seed(args.seed, args.device)

    negative_fold = os.path.join(args.output, "negative.fold")
    positive_fold = os.path.join(args.output, "positive.fold")
    fold_fasta(args.negative_fasta, negative_fold)
    fold_fasta(args.positive_fasta, positive_fold)

    FoldDataset, mirDNN, ImbalancedDatasetSampler = load_training_components()
    train_dataset = FoldDataset([negative_fold, positive_fold], args.seq_length)
    if bool(args.validation_positive_fasta) != bool(args.validation_negative_fasta):
        raise ValueError("Provide both validation_positive_fasta and validation_negative_fasta, or neither")

    if args.validation_positive_fasta and args.validation_negative_fasta:
        validation_negative_fold = os.path.join(args.output, "validation_negative.fold")
        validation_positive_fold = os.path.join(args.output, "validation_positive.fold")
        fold_fasta(args.validation_negative_fasta, validation_negative_fold)
        fold_fasta(args.validation_positive_fasta, validation_positive_fold)
        valid_dataset = FoldDataset([validation_negative_fold, validation_positive_fold], args.seq_length)
    else:
        train_dataset, valid_dataset = split_dataset(train_dataset, args.valid_prop, args.seed)

    model_path = train_model(args, train_dataset, valid_dataset, mirDNN, ImbalancedDatasetSampler)
    print(f"Saved mirDNN model to {model_path}")


if __name__ == "__main__":
    main()
