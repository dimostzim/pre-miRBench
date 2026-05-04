#!/usr/bin/env python3
from pathlib import Path


PATCH_TARGET = Path("/opt/dnnpremir/dnnpremir_src/isPreMiR.py")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected snippet not found:\n{old}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATCH_TARGET.read_text()

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
        "            fd.close()\n",
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
        "            fd.close()\n"
        "        import os as _os\n"
        "        unified_outfile = _os.environ.get(\"PREMIRBENCH_UNIFIED_OUTPUT\")\n"
        "        if unified_outfile:\n"
        "            import csv as _csv\n"
        "            with open(unified_outfile, \"w\", newline=\"\") as _fd:\n"
        "                _w = _csv.writer(_fd)\n"
        "                _w.writerow([\"window_id\", \"probability_score\"])\n"
        "                for i in range(len(name_list)):\n"
        "                    record_id = name_list[i].strip().lstrip(\">\")\n"
        "                    _w.writerow([record_id, float(result[i][0])])\n",
    )

    PATCH_TARGET.write_text(text)


if __name__ == "__main__":
    main()
