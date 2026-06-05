#!/usr/bin/env python
import argparse
import os
import random

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping


def require_tensorflow_gpu(device):
    if str(device).lower() == "cpu":
        raise SystemExit("deepMiRGene training requires a CUDA GPU; got --device cpu")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise SystemExit("deepMiRGene training requires a visible TensorFlow GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_upstream_namespace(source_dir):
    upstream = os.path.join(source_dir, "inference", "deepMiRGene.py")
    with open(upstream) as handle:
        text = handle.read()
    prefix = text.split("## create directories for results and models", 1)[0]
    namespace = {"__name__": "deepmirgene_training_defs"}
    exec(compile(prefix, upstream, "exec"), namespace)
    return namespace


def split_examples(positives, negatives, validation_split, seed):
    rng = random.Random(seed)
    positives = list(positives)
    negatives = list(negatives)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_cut = max(1, int((1.0 - validation_split) * len(positives)))
    neg_cut = max(1, int((1.0 - validation_split) * len(negatives)))
    return positives[:pos_cut], negatives[:neg_cut], positives[pos_cut:], negatives[neg_cut:]


def average_precision(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if labels.size == 0 or np.sum(labels) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / (np.arange(sorted_labels.size) + 1)
    return float(np.sum(precision * sorted_labels) / np.sum(sorted_labels))


class ValidationAUPRCCallback(Callback):
    def __init__(self, validation_data, positive_class_index):
        super().__init__()
        self.validation_data = validation_data
        self.positive_class_index = positive_class_index

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        x_valid, y_valid = self.validation_data
        predictions = self.model.predict(x_valid, verbose=0)
        scores = predictions[:, self.positive_class_index] if predictions.ndim > 1 else predictions.ravel()
        labels = np.argmax(y_valid, axis=1) == self.positive_class_index
        logs["val_auprc"] = average_precision(labels, scores)
        print(f" - val_auprc: {logs['val_auprc']:.4f}")


def build_callbacks(early_stopping_patience, monitor, validation_data, positive_class_index):
    callbacks = []
    if early_stopping_patience is None:
        return callbacks
    if monitor == "val_auprc":
        if validation_data is None:
            raise ValueError("AUPRC early stopping requires explicit validation FASTA inputs")
        callbacks.append(ValidationAUPRCCallback(validation_data, positive_class_index))
    mode = "max" if monitor in {"val_auprc", "val_auc", "val_accuracy", "val_acc"} else "min"
    callbacks.append(
        EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1,
        )
    )
    return callbacks


def main():
    parser = argparse.ArgumentParser(description="Train a deepMiRGene model.")
    parser.add_argument("--positive_fasta", required=True)
    parser.add_argument("--negative_fasta", required=True)
    parser.add_argument("--validation_positive_fasta")
    parser.add_argument("--validation_negative_fasta")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_monitor", default="val_loss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    require_tensorflow_gpu(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepmirgene_src")
    namespace = load_upstream_namespace(source_dir)

    import_data = namespace["import_data"]
    one_hot_wrap = namespace["one_hot_wrap"]
    mymodel = namespace["mymodel"]
    to_categorical = namespace["to_categorical"]
    perfeval = namespace["perfeval"]

    max_len = 400
    dim_enc = 16
    model = mymodel(max_len, dim_enc, 20, 10, 400, 100)

    pos_train = import_data(args.positive_fasta)
    neg_train = import_data(args.negative_fasta)
    if args.validation_positive_fasta and args.validation_negative_fasta:
        pos_valid = import_data(args.validation_positive_fasta)
        neg_valid = import_data(args.validation_negative_fasta)
    else:
        pos_train, neg_train, pos_valid, neg_valid = split_examples(
            pos_train,
            neg_train,
            args.validation_split,
            args.seed,
        )

    x_train = one_hot_wrap(pos_train + neg_train, max_len, dim_enc)
    y_train = to_categorical([0] * len(pos_train) + [1] * len(neg_train), num_classes=2)

    validation_data = None
    if pos_valid and neg_valid:
        x_valid = one_hot_wrap(pos_valid + neg_valid, max_len, dim_enc)
        y_valid = to_categorical([0] * len(pos_valid) + [1] * len(neg_valid), num_classes=2)
        validation_data = (x_valid, y_valid)

    callbacks = build_callbacks(
        args.early_stopping_patience,
        args.early_stopping_monitor,
        validation_data,
        positive_class_index=0,
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        verbose=1,
        batch_size=args.batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
    )
    if validation_data:
        predictions = model.predict(validation_data[0], verbose=0)
        metrics = perfeval(predictions, validation_data[1], verbose=1)
        with open(os.path.join(args.output, "validation_metrics.txt"), "w") as handle:
            handle.write("SE SP F-score PPV g-mean AUROC AUPR\n")
            handle.write(" ".join("{:.6f}".format(value) for value in metrics[:-1]))
            handle.write("\n")

    model_path = os.path.join(args.output, "new_test.hdf5")
    model.save_weights(filepath=model_path, overwrite=True)
    with open(os.path.join(args.output, "history.txt"), "w") as handle:
        handle.write(str(history.history))
    print(f"Saved deepMiRGene weights to {model_path}")


if __name__ == "__main__":
    main()
