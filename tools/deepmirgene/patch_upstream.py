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

    PATCH_TARGET.write_text(text)


if __name__ == "__main__":
    main()
