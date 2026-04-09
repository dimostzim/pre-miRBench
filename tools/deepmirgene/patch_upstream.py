#!/usr/bin/env python3
from pathlib import Path


PATCH_TARGET = Path("/opt/deepmirgene/deepmirgene_src/inference/deepMiRGene.py")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found:\n{old}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATCH_TARGET.read_text()

    text = replace_once(
        text,
        "from keras.utils import to_categorical\n",
        "try:\n"
        "    from keras.utils import to_categorical\n"
        "except ImportError:\n"
        "    from tensorflow.keras.utils import to_categorical\n",
    )

    text = replace_once(
        text,
        "from keras.layers.wrappers import Wrapper\n",
        "try:\n"
        "    from keras.layers.wrappers import Wrapper\n"
        "except ImportError:\n"
        "    from keras.layers import Wrapper\n",
    )

    text = replace_once(
        text,
        "from keras.engine.topology import InputSpec\n",
        "try:\n"
        "    from keras.engine.topology import InputSpec\n"
        "except ImportError:\n"
        "    from keras.engine.base_layer import InputSpec\n",
    )

    text = replace_once(
        text,
        "from keras import backend as K\n\n\n",
        "from keras import backend as K\n\n\n"
        "def backend_name():\n"
        " try:\n"
        "  return K.backend()\n"
        " except AttributeError:\n"
        "  return getattr(K, '_BACKEND', None)\n\n\n"
        "def epsilon_value():\n"
        " try:\n"
        "  return K.epsilon()\n"
        " except AttributeError:\n"
        "  return K.common._EPSILON\n\n\n",
    )

    text = replace_once(
        text,
        "def make_safe(x):\n return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)\n",
        "def make_safe(x):\n return K.clip(x, epsilon_value(), 1.0 - epsilon_value())\n",
    )

    text = replace_once(
        text,
        "  if K._BACKEND == 'tensorflow':\n",
        "  if backend_name() == 'tensorflow':\n",
    )

    text = replace_once(
        text,
        "  if mask is not None:\n"
        "   mask = self.squash_mask(mask)\n"
        "   p_matrix = make_safe(p_matrix * mask)\n"
        "   p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask\n",
        "  if mask is not None:\n"
        "   mask = self.squash_mask(mask)\n"
        "   mask = K.cast(mask, K.dtype(p_matrix))\n"
        "   p_matrix = make_safe(p_matrix * mask)\n"
        "   p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask\n",
    )

    text = replace_once(
        text,
        "  expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)\n",
        "  expanded_p = K.tile(p_vectors, [1, 1, K.shape(x)[2]])\n",
    )

    text = replace_once(
        text,
        "  if mask is None or mask.ndim==2:\n",
        "  if mask is None or K.ndim(mask) == 2:\n",
    )

    text = replace_once(
        text,
        "  return [K.reshape(mul,[K.shape(x)[0],K.shape(x)[1]*K.shape(x)[2]]), p_vector]\n",
        "  flat_dim = K.int_shape(x)[1] * K.int_shape(x)[2]\n"
        "  return [K.reshape(mul, (-1, flat_dim)), p_vector]\n",
    )

    # --- Unified CSV output patches ---
    # Class convention: 0 = pre-miRNA, 1 = pseudo pre-miRNA.
    # predictions[:, 0] is the pre-miRNA probability.
    # import_data discards IDs; we re-parse for IDs inline (FASTA is small).

    # Patch 1: collect IDs alongside the existing X_test encoding call.
    text = replace_once(
        text,
        " X_test = one_hot_wrap(import_data(\"%s\" % INFILE), MAX_LEN, DIM_ENC)\n",
        " X_test = one_hot_wrap(import_data(\"%s\" % INFILE), MAX_LEN, DIM_ENC)\n"
        " seq_ids = [record.id for record in SeqIO.parse(\"%s\" % INFILE, \"fasta\")]\n",
    )

    # Patch 2: replace argmax + np.savetxt block with CSV probability output.
    text = replace_once(
        text,
        " predictions = model.predict(X_test,verbose=0)\n"
        " class_label = np.uint8(np.argmax(predictions,axis=1))  # 0: pre-miRNA, 1: pseudo pre-miRNA \n"
        " print(\"True pre-miRNA: %d, Pseudo pre-miRNA: %d\" % (sum(class_label == 0), sum(class_label == 1)))\n"
        " \n"
        " print(\"Wrighting the results on \\\"%s\\\". (0:true pre-miRNA, 1: pseudo pre-miRNA)\" % OUTFILE)\n"
        " np.savetxt(OUTFILE, class_label, fmt='%d')",
        " predictions = model.predict(X_test,verbose=0)\n"
        " class_label = np.uint8(np.argmax(predictions,axis=1))\n"
        " print(\"True pre-miRNA: %d, Pseudo pre-miRNA: %d\" % (sum(class_label == 0), sum(class_label == 1)))\n"
        " import csv as _csv\n"
        " with open(OUTFILE, \"w\", newline=\"\") as _fd:\n"
        "  _w = _csv.writer(_fd)\n"
        "  _w.writerow([\"window_id\", \"probability_score\"])\n"
        "  for seq_id, pred_row in zip(seq_ids, predictions.tolist()):\n"
        "   _w.writerow([seq_id, float(pred_row[0])])  # col 0 = pre-miRNA",
    )

    PATCH_TARGET.write_text(text)
    print(f"Patched {PATCH_TARGET}")


if __name__ == "__main__":
    main()
