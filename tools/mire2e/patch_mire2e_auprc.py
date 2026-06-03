#!/usr/bin/env python
import site
import sys
from pathlib import Path


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match for miRe2e patch block, found {count}: {old[:80]!r}")
    return text.replace(old, new)


def find_predictor_path():
    search_roots = []
    search_roots.extend(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        search_roots.append(site.getusersitepackages())
    search_roots.extend(sys.path)
    for root in dict.fromkeys(search_roots):
        if not root:
            continue
        candidate = Path(root) / "miRe2e" / "predictor.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate installed miRe2e/predictor.py")


path = find_predictor_path()
text = path.read_text()

text = replace_once(text, "import numpy as np\n", "import numpy as np\nimport sklearn.metrics\n")
text = replace_once(
    text,
    "        f1_max = 0\n",
    "        auprc_max = 0\n        epoch_max = 0\n        best_epoch = 0\n",
)
text = replace_once(
    text,
    "            aucv, f1v, prev, recv = get_error(labels_all_valid[:, 0].cpu(),\n"
    "                                              predictions_all_valid[:,\n"
    "                                              0].cpu())\n",
    "            aucv, f1v, prev, recv = get_error(labels_all_valid[:, 0].cpu(),\n"
    "                                              predictions_all_valid[:,\n"
    "                                              0].cpu())\n"
    "            valid_auprc = sklearn.metrics.average_precision_score(\n"
    "                labels_all_valid[:, 0].cpu().numpy(),\n"
    "                predictions_all_valid[:, 0].cpu().numpy())\n",
)
text = replace_once(
    text,
    '                print(f"Valid: Loss {lossv: .3f} F1 {f1v: .3f} REC "\n'
    '                      f"{recv: .3f} PRE {prev: .3f}")\n',
    '                print(f"Valid: Loss {lossv: .3f} AUPRC {valid_auprc: .3f} "\n'
    '                      f"F1 {f1v: .3f} REC {recv: .3f} PRE {prev: .3f}")\n',
)
text = replace_once(text, "            if f1v > f1_max:\n", "            if valid_auprc > auprc_max:\n")
text = replace_once(text, "                f1_max = f1v\n", "                auprc_max = valid_auprc\n")
text = replace_once(text, "            if epoch_max >= 30:\n", "            if epoch_max >= 20:\n")
text = replace_once(
    text,
    '                    print(f"Best epoch {best_epoch}: F1 {f1_max}")\n',
    '                    print(f"Best epoch {best_epoch}: AUPRC {auprc_max}")\n',
)

path.write_text(text)
