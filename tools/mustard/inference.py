#!/usr/bin/env python
import argparse
import os
import subprocess
from pathlib import Path


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one MuStARD patch match, found {count}: {old[:80]!r}")
    return text.replace(old, new)


def replace_first(text, old, new):
    if old not in text:
        raise RuntimeError(f"Expected MuStARD patch match: {old[:80]!r}")
    return text.replace(old, new, 1)


def patch_safe_chromosome_filenames(base_dir):
    """Avoid shell/file breakage for chromosome names containing |, :, etc."""
    format_path = Path(base_dir) / "mustard_src" / "src" / "lib" / "perl" / "Files" / "Format.pm"
    dnn_path = Path(base_dir) / "mustard_src" / "src" / "lib" / "perl" / "Models" / "DNN.pm"

    format_text = format_path.read_text()
    if "sub safe_chrom_file_id" not in format_text:
        format_text = replace_once(
            format_text,
            "use Files::CleanUp;\n",
            "use Files::CleanUp;\n"
            "use Digest::MD5 qw(md5_hex);\n\n"
            "sub safe_chrom_file_id {\n"
            "\tmy ($name) = @_;\n"
            "\tmy $safe = $name;\n"
            "\t$safe =~ s/[^A-Za-z0-9_.-]/_/g;\n"
            "\t$safe .= \"__\".substr(md5_hex($name), 0, 10) if $safe ne $name;\n"
            "\treturn $safe;\n"
            "}\n",
        )
        format_text = replace_once(
            format_text,
            "foreach my $chr (sort { $a cmp $b } keys(%regions)){\n\n",
            "foreach my $chr (sort { $a cmp $b } keys(%regions)){\n\n"
            "\t\tmy $chr_file = safe_chrom_file_id($chr);\n\n",
        )
        format_text = format_text.replace("$working_dir/targets.$chr", "$working_dir/targets.$chr_file")
        format_text = format_text.replace('"targets.$chr"', '"targets.$chr_file"')
        format_text = replace_first(
            format_text,
            "foreach my $chrom (@chroms){\n\n\t\tfor(my $i = 0; $i < $class_num; $i++){\n",
            "foreach my $chrom (@chroms){\n\n"
            "\t\tmy $chrom_file = safe_chrom_file_id($chrom);\n\n"
            "\t\tfor(my $i = 0; $i < $class_num; $i++){\n",
        )
        format_text = replace_first(
            format_text,
            "foreach my $chrom (@chroms){\n\n\t\tfor(my $i = 0; $i < $class_num; $i++){\n",
            "foreach my $chrom (@chroms){\n\n"
            "\t\tmy $chrom_file = safe_chrom_file_id($chrom);\n\n"
            "\t\tfor(my $i = 0; $i < $class_num; $i++){\n",
        )
        format_text = format_text.replace("targets.$chrom.", "targets.$chrom_file.")
        format_text = format_text.replace("predictions.$chrom.", "predictions.$chrom_file.")
        format_path.write_text(format_text)

    dnn_text = dnn_path.read_text()
    if "safe_chrom_file_id" not in dnn_text:
        dnn_text = replace_once(
            dnn_text,
            "\tforeach my $chr (@chroms){\n\n\t\twarn \"\\t\\t\\tOn $chr.\\n\";\n\n",
            "\tforeach my $chr (@chroms){\n\n"
            "\t\tmy $chr_file = Files::Format::safe_chrom_file_id($chr);\n\n"
            "\t\twarn \"\\t\\t\\tOn $chr.\\n\";\n\n",
        )
        dnn_text = dnn_text.replace("targets.$chr", "targets.$chr_file")
        dnn_path.write_text(dnn_text)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--targetIntervals", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--consDir", required=True)
    
    p.add_argument("--chromList", default="all")
    p.add_argument("--dir", default="results")
    p.add_argument("--model", default="MuStARD-mirSFC-U")
    p.add_argument("--classNum", type=int, default=2)
    p.add_argument("--modelType", default="CNN")
    
    # optional
    p.add_argument("--modelDirName", default="results")
    p.add_argument("--intermDir", default="same")
    p.add_argument("--winSize", type=int, default=100)
    p.add_argument("--staticPredFlag", type=int, default=0)
    p.add_argument("--inputMode", default="sequence,RNAfold,conservation")  # best model uses all 3 sequence types
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--step", type=int, default=5)
    args = p.parse_args()

    perl_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mustard_src", "MuStARD.pl")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    patch_safe_chromosome_filenames(base_dir)

    # Resolve bundled model names and explicit trained model paths.
    if os.path.isfile(args.model):
        model_path = os.path.abspath(args.model)
    else:
        model_path = os.path.join(base_dir, "data", "models", args.model, "CNNonRaw.hdf5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MuStARD model not found: {model_path}")
    args.model = model_path

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    cmd = [
        "perl",
        perl_script,
        "predict",
        "--chromList", args.chromList,
        "--targetIntervals", os.path.abspath(args.targetIntervals),
        "--genome", os.path.abspath(args.genome),
        "--consDir", os.path.abspath(args.consDir),
        "--dir", os.path.abspath(args.dir),
        "--model", os.path.abspath(args.model),
        "--classNum", str(args.classNum),
        "--modelType", args.modelType,
        "--winSize", str(args.winSize),
        "--step", str(args.step),
        "--staticPredFlag", str(args.staticPredFlag),
        "--inputMode", args.inputMode,
        "--threads", str(args.threads),
        "--modelDirName", args.modelDirName,
    ]
    
    if args.intermDir != "same":
        cmd.extend(["--intermDir", os.path.abspath(args.intermDir)])

    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
