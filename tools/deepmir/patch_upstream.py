#!/usr/bin/env python3
"""
Patch deepmir's predictor.py to output a unified CSV (window_id, probability_score)
instead of the original results.csv with binary string labels.

Class convention in the upstream model: argmax==1 means pre-miRNA.
So raw_preds[:, 1] is the pre-miRNA probability.

No model weights or architecture are changed.
"""
from pathlib import Path

PATCH_TARGET = Path("/opt/deepmir/deepmir_src/predictor.py")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found in {PATCH_TARGET}:\n{repr(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATCH_TARGET.read_text()

    # Replace compute_predictions to use raw softmax col 1 instead of argmax label.
    # raw_preds shape: (n, 2); col 1 = pre-miRNA probability.
    text = replace_once(
        text,
        "    model = load_model(MODEL_FILENAME)\n"
        "    predictions = np.argmax(model.predict(images), axis=1)\n"
        "    \n"
        "    results_filename = data_directory + \"/results.csv\"\n"
        "    with open(results_filename, 'w') as results:\n"
        "        results.write(\"hairpin,sequence,fold,label\\n\")\n"
        "        \n"
        "        for name, prediction in zip(names.tolist(), predictions.tolist()):\n"
        "            name = name.decode('utf-8')\n"
        "            if prediction == 1:\n"
        "                label = \"pre-miRNA\"\n"
        "            else:\n"
        "                label = \"not pre-miRNA\"\n"
        "\n"
        "            seq_fold = seq_fold_dict[name]            \n"
        "            results.write(\"{},{},{},{}\\n\".format(name,\n"
        "                                                 seq_fold[0],\n"
        "                                                 seq_fold[1],\n"
        "                                                 label))\n"
        "\n"
        "    print(\"Prediction results were written to: {}\".format(results_filename))",
        "    model = load_model(MODEL_FILENAME)\n"
        "    raw_preds = model.predict(images)  # shape (n, 2); col 1 = pre-miRNA\n"
        "\n"
        "    import csv as _csv\n"
        "    results_filename = data_directory + \"/predictions.csv\"\n"
        "    with open(results_filename, 'w', newline='') as results:\n"
        "        _w = _csv.writer(results)\n"
        "        _w.writerow([\"window_id\", \"probability_score\"])\n"
        "        for name, pred_row in zip(names.tolist(), raw_preds.tolist()):\n"
        "            name = name.decode('utf-8')\n"
        "            _w.writerow([name, float(pred_row[1])])\n"
        "\n"
        "    print(\"Prediction results were written to: {}\".format(results_filename))",
    )

    PATCH_TARGET.write_text(text)
    print(f"Patched {PATCH_TARGET}")


if __name__ == "__main__":
    main()
