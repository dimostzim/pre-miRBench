import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("mustard_train", Path(__file__).resolve().parents[2] / "tools" / "mustard" / "train.py")
mustard_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mustard_train)


def test_run_checked_fails_when_a_traceback_is_printed_despite_exit_code_zero():
    # MuStARD's Perl step exits 0 after its Python trainer crashes, printing only the traceback
    crashed = [sys.executable, "-c", "print('epoch 80'); print('Traceback (most recent call last):'); print('InternalError')"]
    with pytest.raises(SystemExit, match="traceback in output: True"):
        mustard_train.run_checked(crashed)
    with pytest.raises(SystemExit, match="exit code 3"):
        mustard_train.run_checked([sys.executable, "-c", "raise SystemExit(3)"])
    mustard_train.run_checked([sys.executable, "-c", "print('[0.055, 0.986]')"])


def test_seed_training_script_replaces_the_three_hard_coded_seeds():
    script = "import numpy as np\nnp.random.seed(20)\nimport random as rn\nrn.seed(1984)\nx = 1\ntf.random.set_seed(8)\n"
    seeded = mustard_train.seed_training_script(script, 3)
    assert "np.random.seed(3)" in seeded and "rn.seed(3)" in seeded and "tf.random.set_seed(3)" in seeded
    assert "20)" not in seeded and "1984" not in seeded and "set_seed(8)" not in seeded
    with pytest.raises(RuntimeError, match="rn.seed"):
        mustard_train.seed_training_script(script.replace("rn.seed(1984)", ""), 3)
