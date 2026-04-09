#!/usr/bin/env python3
"""
Patch isPreMiR.py to output a unified CSV (window_id, probability_score)
instead of the original True/False block format.

Class convention in the upstream model: index 0 = pre-miRNA, index 1 = non-pre-miRNA.
So result[:, 0] is the pre-miRNA probability.

No model weights or architecture are changed.
"""
from pathlib import Path

PATCH_TARGET = Path("/opt/dnnpremir/dnnpremir_src/isPreMiR.py")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found in {PATCH_TARGET}:\n{repr(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATCH_TARGET.read_text()

    # Replace the file-output block to write CSV with probability scores.
    # 'result' is the raw (n, 2) softmax output; result[i][0] = pre-miRNA probability.
    # name_list entries are raw FASTA header lines e.g. ">record_id\n".
    text = replace_once(
        text,
        "        if outfile:\n"
        "            fd = open(outfile,\"w\")\n"
        "            for i in range(len(name_list)):\n"
        "                fd.write(name_list[i])\n"
        "                fd.write(seq_list[i])\n"
        "                if prediction[i] == 0:\n"
        "                    fd.write(\"  True\\n\")\n"
        "                else:\n"
        "                    fd.write(\"  False\\n\")\n"
        "                fd.write(\"===========================\\n\")\n"
        "            fd.close()",
        "        if outfile:\n"
        "            import csv as _csv\n"
        "            with open(outfile, \"w\", newline=\"\") as _fd:\n"
        "                _w = _csv.writer(_fd)\n"
        "                _w.writerow([\"window_id\", \"probability_score\"])\n"
        "                for i in range(len(name_list)):\n"
        "                    record_id = name_list[i].strip().lstrip(\">\")\n"
        "                    _w.writerow([record_id, float(result[i][0])])",
    )

    PATCH_TARGET.write_text(text)
    print(f"Patched {PATCH_TARGET}")


if __name__ == "__main__":
    main()
