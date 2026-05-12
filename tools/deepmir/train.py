#!/usr/bin/env python
import argparse
import os
import random
import re
import subprocess
import sys

import imageio
import keras
import numpy as np
from keras import backend as K


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


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def generate_image(source_dir, sequence, name, image_dir):
    output_path = os.path.join(image_dir, safe_name(name) + ".png")
    jar_path = os.path.join(source_dir, "hairpin_image_generator", "ImageCalc.jar")
    process = subprocess.run(
        ["java", "-jar", jar_path, "-o", output_path, "-s", sequence],
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stderr:
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"Skipping {name}: {process.stderr.strip()}", file=sys.stderr)
        return None
    return output_path


def build_dataset(source_dir, positive_fasta, negative_fasta, output_dir):
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    images = []
    labels = []
    names = []

    for fasta_path, label in ((negative_fasta, 0), (positive_fasta, 1)):
        for name, sequence in read_fasta(fasta_path):
            image_path = generate_image(source_dir, sequence, name, image_dir)
            if not image_path:
                continue
            image = imageio.imread(image_path)
            if image.shape != (25, 100, 3):
                print(f"Skipping {name}: expected image shape (25, 100, 3), got {image.shape}", file=sys.stderr)
                continue
            images.append(image)
            labels.append(label)
            names.append(safe_name(name))

    if not images:
        raise ValueError("No valid DeepMir hairpin images were generated")

    x = np.array(images).astype("float32")
    if K.image_data_format() == "channels_first":
        x = np.swapaxes(x, 1, 3)
    if np.amax(x) > 1:
        x /= 255
    y = keras.utils.to_categorical(np.array(labels), 2)
    return x, y, np.array(names, dtype=np.string_)


def build_model(source_dir, architecture, modules, dense_units, filters):
    model_selection_dir = os.path.join(source_dir, "model_selection")
    sys.path.insert(0, model_selection_dir)

    if architecture == "vgg":
        from model_generators import vgg_model_generator as generator

        builders = {
            1: generator.build_model_one_module_3x3,
            2: generator.build_model_two_modules_3x3,
            3: generator.build_model_three_modules_3x3,
            4: generator.build_model_four_modules_3x3,
        }
        return builders[modules](dense_units)[0]
    if architecture == "resnet":
        from model_generators import resnet_model_generator as generator

        builders = {
            1: generator.build_model_one_module,
            2: generator.build_model_two_modules,
            3: generator.build_model_three_modules,
            4: generator.build_model_four_modules,
        }
        return builders[modules](filters)[0]
    if architecture == "inception":
        from model_generators import inception_model_generator as generator

        builders = {
            1: generator.build_model_one_module,
            2: generator.build_model_two_modules,
            3: generator.build_model_three_modules,
            4: generator.build_model_four_modules,
        }
        return builders[modules](filters)[0]
    raise ValueError(f"Unsupported architecture: {architecture}")


def main():
    parser = argparse.ArgumentParser(description="Train a DeepMir image model.")
    parser.add_argument("--positive_fasta", required=True)
    parser.add_argument("--negative_fasta", required=True)
    parser.add_argument("--validation_positive_fasta")
    parser.add_argument("--validation_negative_fasta")
    parser.add_argument("--pretrain_positive_fasta")
    parser.add_argument("--pretrain_negative_fasta")
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=["vgg", "resnet", "inception"], default="vgg")
    parser.add_argument("--training_mode", choices=["base", "fine_tune"], default="base")
    parser.add_argument("--modules", type=int, choices=[1, 2, 3, 4], default=3)
    parser.add_argument("--dense_units", type=int, default=256)
    parser.add_argument("--filters", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepmir_src")
    model = build_model(source_dir, args.architecture, args.modules, args.dense_units, args.filters)

    if args.training_mode == "fine_tune" and args.pretrain_positive_fasta and args.pretrain_negative_fasta:
        pretrain_x, pretrain_y, _ = build_dataset(
            source_dir,
            args.pretrain_positive_fasta,
            args.pretrain_negative_fasta,
            os.path.join(args.output, "pretrain_dataset"),
        )
        model.fit(pretrain_x, pretrain_y, batch_size=args.batch_size, epochs=args.pretrain_epochs, shuffle=True)

    train_x, train_y, train_names = build_dataset(
        source_dir,
        args.positive_fasta,
        args.negative_fasta,
        os.path.join(args.output, "train_dataset"),
    )
    np.savez_compressed(os.path.join(args.output, "train_names.npz"), train_names)

    validation_data = None
    validation_split = args.validation_split
    if args.validation_positive_fasta and args.validation_negative_fasta:
        valid_x, valid_y, valid_names = build_dataset(
            source_dir,
            args.validation_positive_fasta,
            args.validation_negative_fasta,
            os.path.join(args.output, "validation_dataset"),
        )
        np.savez_compressed(os.path.join(args.output, "validation_names.npz"), valid_names)
        validation_data = (valid_x, valid_y)
        validation_split = 0.0

    model.fit(
        train_x,
        train_y,
        validation_data=validation_data,
        validation_split=validation_split,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
    )
    model_path = os.path.join(args.output, "model.h5")
    model.save(model_path)
    print(f"Saved DeepMir model to {model_path}")


if __name__ == "__main__":
    main()
