#!/usr/bin/env python

import os
import sys
from subprocess import PIPE, Popen

import imageio
import keras
import numpy as np
from keras import backend as K
from keras.models import load_model
from pyfaidx import Fasta

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FILENAME = CURRENT_DIR + "/models/fine_tuned_cnn.h5"
HAIRPIN_IMAGE_GENERATOR_JAR = "./hairpin_image_generator/ImageCalc.jar"


def generate_hairpin_images(fasta_filename, data_directory):
    images_directory = data_directory + "/images"
    try:
        if not os.path.exists(images_directory):
            os.makedirs(images_directory)
    except OSError:
        print("Error: Creating directory {}".format(images_directory))
        sys.exit(1)

    sequences = Fasta(fasta_filename)
    seq_fold_dict = {}
    skipped = 0
    for key in sequences.keys():
        sequence = str(sequences[key][:].seq)
        try:
            fold = generate_hairpin_image(sequence, key, images_directory)
        except Exception as exc:
            skipped += 1
            print("Skipping {}: {}".format(key, exc), file=sys.stderr)
            continue
        seq_fold_dict[key] = (sequence, fold)

    if skipped:
        print(
            "Skipped {} sequences that DeepMir could not render as hairpin images".format(skipped),
            file=sys.stderr,
        )

    return seq_fold_dict


def generate_hairpin_image(sequence, sequence_identifier, output_directory):
    hairpin_image_name = output_directory + "/" + sequence_identifier + ".png"
    process = Popen(
        ["java", "-jar", HAIRPIN_IMAGE_GENERATOR_JAR, "-o", hairpin_image_name, "-s", sequence],
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = process.communicate()

    if not stderr.decode("utf-8"):
        stdout = stdout.decode("utf-8")
        fold_info = stdout.split("\n")[1]
        return fold_info.split(":")[1].strip()

    if os.path.exists(hairpin_image_name):
        os.remove(hairpin_image_name)
    msg = "Cannot generate a hairpin image for {}.\nError message: {}"
    raise Exception(msg.format(sequence, stderr.decode("utf-8")))


def generate_hairpin_array(data_directory, allowed_names):
    images_directory = data_directory + "/images"
    image_short_filenames = []
    image_long_filenames = []

    for dirpath, _, filenames in os.walk(images_directory):
        for filename in filenames:
            if not filename.endswith(".png"):
                continue
            name = filename.split(".")[0]
            if name not in allowed_names:
                continue
            im = imageio.imread(dirpath + "/" + filename)
            if im.shape != (25, 100, 3):
                continue

            image_long_filenames.append(dirpath + "/" + filename)
            image_short_filenames.append(name)

    num_images = len(image_long_filenames)
    print("Images to process: {}".format(num_images))

    images_tensor = np.zeros(shape=(num_images, 25, 100, 3), dtype=float)
    names_tensor = np.array(image_short_filenames, dtype=np.string_)
    for image_index, image_filename in enumerate(image_long_filenames):
        im = imageio.imread(image_filename)
        if im.shape == (25, 100, 3):
            images_tensor[image_index] = im
        if image_index > 0 and (image_index % 1000) == 0:
            print("Number of images processed: {}".format(image_index))

    output_images_filename = data_directory + "/images.npz"
    output_names_filename = data_directory + "/names.npz"
    np.savez_compressed(output_images_filename, images_tensor)
    np.savez_compressed(output_names_filename, names_tensor)
    print("Numpy arrays for images were created in: {}".format(data_directory))
    return num_images


def load_image_data(filename):
    x = np.load(filename)["arr_0"]

    if K.image_data_format() == "channels_first":
        x = np.swapaxes(x, 1, 3)

    x = x.astype("float32")
    if np.amax(x) > 1:
        x /= 255

    return x


def compute_predictions(data_directory, seq_fold_dict):
    images = load_image_data(data_directory + "/images.npz")
    names = np.load(data_directory + "/names.npz")["arr_0"]

    model = load_model(MODEL_FILENAME)
    raw_preds = model.predict(images)  # shape (n, 2); col 1 = pre-miRNA

    import csv as _csv
    results_filename = data_directory + "/predictions.csv"
    with open(results_filename, "w", newline="") as results:
        _w = _csv.writer(results)
        _w.writerow(["window_id", "probability_score"])
        for name, pred_row in zip(names.tolist(), raw_preds.tolist()):
            name = name.decode("utf-8")
            _w.writerow([name, float(pred_row[1])])

    print("Prediction results were written to: {}".format(results_filename))


def write_empty_results(data_directory):
    import csv as _csv
    results_filename = data_directory + "/predictions.csv"
    with open(results_filename, "w", newline="") as results:
        _csv.writer(results).writerow(["window_id", "probability_score"])
    print("No valid DeepMir hairpin images were generated; wrote empty results to: {}".format(results_filename))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        instructions = "Usage: python predictor.py [input_filename]\n"
        instructions += "\tThe input_filename is the name of the fasta file containing RNA sequences.\n"
        instructions += "\tThe examples directory contains some fasta files that this program can process.\n"
        print("\nError: Please provide the name of the file containing the RNA sequences to process.")
        print(instructions)
        exit(0)

    input_filename = sys.argv[1]
    base_input_filename = os.path.basename(input_filename)
    data_directory = CURRENT_DIR + "/user_data/" + base_input_filename.split(".")[0]

    seq_fold_dict = generate_hairpin_images(input_filename, data_directory)
    if not seq_fold_dict:
        write_empty_results(data_directory)
        exit(0)

    image_count = generate_hairpin_array(data_directory, set(seq_fold_dict.keys()))
    if image_count == 0:
        write_empty_results(data_directory)
        exit(0)

    compute_predictions(data_directory, seq_fold_dict)
