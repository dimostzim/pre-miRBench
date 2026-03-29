#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

from tool_adapters import (
    TOOLS,
    WINDOW_TOOLS,
    iter_global_window_positions,
    load_chr_sequence,
    load_truth_rows,
    normalize_sequence,
    parse_tools,
    reverse_complement_rna,
    write_csv_rows,
    write_truth_bed,
    write_yaml_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare chr14 scan inputs for all tools, using native scan mode where available and generated windows otherwise."
    )
    parser.add_argument("--genome", default="benchmark/data/chr14.fa", help="Genome FASTA file")
    parser.add_argument("--truth-bed", default="benchmark/data/hsa-precursors-no-v2.bed", help="BED file with truth precursor loci")
    parser.add_argument("--cons-dir", default="benchmark/data", help="Conservation directory for MuStARD scan mode")
    parser.add_argument("--output-dir", default="benchmark/prepared_inputs/scan_benchmark", help="Directory to write prepared scan inputs into")
    parser.add_argument("--chrom", default="chr14", help="Chromosome to scan")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Core chunk size in base pairs")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Extra context around each chunk for native scan tools")
    parser.add_argument("--window-stride", type=int, default=50, help="Global stride for externally generated scan windows")
    parser.add_argument("--tools", default="all", help="Comma-separated tool list or 'all'")
    return parser.parse_args()


def make_chunks(chrom, chrom_length, chunk_size, overlap):
    rows = []
    chunk_index = 1
    for core_start in range(1, chrom_length + 1, chunk_size):
        core_end = min(chrom_length, core_start + chunk_size - 1)
        native_start = max(1, core_start - overlap)
        native_end = min(chrom_length, core_end + overlap)
        rows.append({
            "chunk_id": f"{chrom}_chunk{chunk_index:04d}",
            "chrom": chrom,
            "core_start": core_start,
            "core_end": core_end,
            "native_start": native_start,
            "native_end": native_end,
        })
        chunk_index += 1
    return rows


def write_fasta(path, records):
    with open(path, "w") as handle:
        for record_id, sequence in records:
            handle.write(f">{record_id}\n{sequence}\n")


def write_bed(path, rows):
    with open(path, "w") as handle:
        for chrom, start0, end1, record_id, score, strand in rows:
            handle.write(f"{chrom}\t{start0}\t{end1}\t{record_id}\t{score}\t{strand}\n")


def repo_relpath(path, repo_root):
    return Path(path).resolve().relative_to(repo_root).as_posix()


def write_tool_manifest(path, rows):
    fieldnames = ["chunk_id", "config_path", "input_path", "metadata_path", "window_count"]
    write_csv_rows(path, fieldnames, rows)


def build_runtime_config(tool, input_relpath, chunk_id, chrom, genome_relpath, cons_dir_relpath):
    if tool == "deepmir":
        return {
            "input": input_relpath,
            "model": "fine_tuned_cnn",
        }
    if tool == "deepmirgene":
        return {
            "input": input_relpath,
            "model": None,
        }
    if tool == "dnnpremir":
        return {
            "input": input_relpath,
            "seq_length": 180,
        }
    if tool == "mirdnn":
        return {
            "input": input_relpath,
            "model": "animal",
            "device": "cuda:0",
            "seq_length": 160,
            "batch_size": 1024,
        }
    if tool == "mire2e":
        return {
            "input": input_relpath,
            "device": "cuda",
            "pretrained": "hsa",
            "length": 100,
            "step": 20,
            "batch_size": 4096,
        }
    if tool == "mustard":
        return {
            "targetIntervals": input_relpath,
            "genome": genome_relpath,
            "consDir": cons_dir_relpath,
            "chromList": chrom,
            "model": "MuStARD-mirSFC-U",
            "classNum": 2,
            "modelType": "CNN",
            "winSize": 100,
            "step": 5,
            "staticPredFlag": 0,
            "inputMode": "sequence,RNAfold,conservation",
            "threads": 10,
            "modelDirName": "results",
            "intermDir": f"results/mustard_intermediate/scan/{chunk_id}",
        }
    raise ValueError(f"Unsupported tool: {tool}")


def is_supported_external_window(sequence):
    return not (set(sequence) - set("ACGU"))


def prepare_native_mire2e(chunk_rows, chrom_sequence, tool_dir, repo_root, chrom, genome_relpath, cons_dir_relpath):
    chunk_dir = tool_dir / "chunks"
    config_dir = tool_dir / "configs"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for chunk in chunk_rows:
        fasta_path = chunk_dir / f"{chunk['chunk_id']}.fa"
        sequence = normalize_sequence(chrom_sequence[chunk["native_start"] - 1:chunk["native_end"]])
        write_fasta(fasta_path, [(chunk["chunk_id"], sequence)])

        config_path = config_dir / f"{chunk['chunk_id']}.yaml"
        write_yaml_config(
            config_path,
            build_runtime_config(
                "mire2e",
                repo_relpath(fasta_path, repo_root),
                chunk["chunk_id"],
                chrom,
                genome_relpath,
                cons_dir_relpath,
            ),
        )
        manifest_rows.append({
            "chunk_id": chunk["chunk_id"],
            "config_path": repo_relpath(config_path, repo_root),
            "input_path": repo_relpath(fasta_path, repo_root),
            "metadata_path": "",
            "window_count": "",
        })

    write_tool_manifest(tool_dir / "scan_manifest.csv", manifest_rows)
    return len(manifest_rows)


def prepare_native_mustard(chunk_rows, tool_dir, repo_root, chrom, genome_relpath, cons_dir_relpath):
    chunk_dir = tool_dir / "chunks"
    config_dir = tool_dir / "configs"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for chunk in chunk_rows:
        bed_path = chunk_dir / f"{chunk['chunk_id']}.bed"
        rows = [
            (
                chunk["chrom"],
                chunk["native_start"] - 1,
                chunk["native_end"],
                f"{chunk['chunk_id']}__plus",
                0,
                "+",
            ),
            (
                chunk["chrom"],
                chunk["native_start"] - 1,
                chunk["native_end"],
                f"{chunk['chunk_id']}__minus",
                0,
                "-",
            ),
        ]
        write_bed(bed_path, rows)

        config_path = config_dir / f"{chunk['chunk_id']}.yaml"
        write_yaml_config(
            config_path,
            build_runtime_config(
                "mustard",
                repo_relpath(bed_path, repo_root),
                chunk["chunk_id"],
                chrom,
                genome_relpath,
                cons_dir_relpath,
            ),
        )
        manifest_rows.append({
            "chunk_id": chunk["chunk_id"],
            "config_path": repo_relpath(config_path, repo_root),
            "input_path": repo_relpath(bed_path, repo_root),
            "metadata_path": "",
            "window_count": "",
        })

    write_tool_manifest(tool_dir / "scan_manifest.csv", manifest_rows)
    return len(manifest_rows)


def prepare_window_tool(tool, chunk_rows, chrom_sequence, tool_dir, repo_root, chrom, genome_relpath, cons_dir_relpath, window_stride):
    window_length = WINDOW_TOOLS[tool]
    chunk_dir = tool_dir / "chunks"
    config_dir = tool_dir / "configs"
    metadata_dir = tool_dir / "metadata"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    chrom_length = len(chrom_sequence)
    manifest_rows = []
    for chunk in chunk_rows:
        fasta_records = []
        bed_rows = []
        metadata_rows = []
        for start, end in iter_global_window_positions(
            chrom_length,
            window_length,
            window_stride,
            chunk["core_start"],
            chunk["core_end"],
        ):
            plus_sequence = normalize_sequence(chrom_sequence[start - 1:end])
            if not is_supported_external_window(plus_sequence):
                continue
            minus_sequence = reverse_complement_rna(plus_sequence)
            for strand, sequence in (("+", plus_sequence), ("-", minus_sequence)):
                strand_name = "plus" if strand == "+" else "minus"
                record_id = f"{chunk['chrom']}__{start}__{end}__{strand_name}"
                fasta_records.append((record_id, sequence))
                bed_rows.append((chunk["chrom"], start - 1, end, record_id, 0, strand))
                metadata_rows.append({
                    "record_id": record_id,
                    "chunk_id": chunk["chunk_id"],
                    "chrom": chunk["chrom"],
                    "start": start,
                    "end": end,
                    "strand": strand,
                })

        if not metadata_rows:
            continue

        fasta_path = chunk_dir / f"{chunk['chunk_id']}.fa"
        bed_path = chunk_dir / f"{chunk['chunk_id']}.bed"
        metadata_path = metadata_dir / f"{chunk['chunk_id']}.csv"
        config_path = config_dir / f"{chunk['chunk_id']}.yaml"

        write_fasta(fasta_path, fasta_records)
        write_bed(bed_path, bed_rows)
        write_csv_rows(
            metadata_path,
            ["record_id", "chunk_id", "chrom", "start", "end", "strand"],
            metadata_rows,
        )
        write_yaml_config(
            config_path,
            build_runtime_config(
                tool,
                repo_relpath(fasta_path, repo_root),
                chunk["chunk_id"],
                chrom,
                genome_relpath,
                cons_dir_relpath,
            ),
        )
        manifest_rows.append({
            "chunk_id": chunk["chunk_id"],
            "config_path": repo_relpath(config_path, repo_root),
            "input_path": repo_relpath(fasta_path, repo_root),
            "metadata_path": repo_relpath(metadata_path, repo_root),
            "window_count": len(metadata_rows),
        })

    write_tool_manifest(tool_dir / "scan_manifest.csv", manifest_rows)
    return len(manifest_rows)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    genome_path = (repo_root / args.genome).resolve()
    cons_dir_path = (repo_root / args.cons_dir).resolve()
    truth_bed_path = (repo_root / args.truth_bed).resolve()
    genome_relpath = repo_relpath(genome_path, repo_root)
    cons_dir_relpath = repo_relpath(cons_dir_path, repo_root)

    tools = parse_tools(args.tools)
    chrom_sequence = load_chr_sequence(genome_path, args.chrom)
    chunk_rows = make_chunks(args.chrom, len(chrom_sequence), args.chunk_size, args.chunk_overlap)

    write_csv_rows(
        output_dir / "chunks.csv",
        ["chunk_id", "chrom", "core_start", "core_end", "native_start", "native_end"],
        chunk_rows,
    )

    truth_rows = load_truth_rows(truth_bed_path, chrom=args.chrom)
    truth_output = output_dir / f"truth_{args.chrom}.bed"
    write_truth_bed(truth_output, truth_rows)
    print(f"truth loci: {len(truth_rows)} -> {truth_output}")
    print(f"chunks: {len(chunk_rows)} -> {output_dir / 'chunks.csv'}")

    for tool in tools:
        tool_dir = output_dir / tool
        if tool_dir.exists():
            shutil.rmtree(tool_dir)
        if tool == "mire2e":
            count = prepare_native_mire2e(
                chunk_rows,
                chrom_sequence,
                tool_dir,
                repo_root,
                args.chrom,
                genome_relpath,
                cons_dir_relpath,
            )
            print(f"{tool}: prepared {count} native scan chunks")
        elif tool == "mustard":
            count = prepare_native_mustard(
                chunk_rows,
                tool_dir,
                repo_root,
                args.chrom,
                genome_relpath,
                cons_dir_relpath,
            )
            print(f"{tool}: prepared {count} native scan chunks")
        else:
            count = prepare_window_tool(
                tool,
                chunk_rows,
                chrom_sequence,
                tool_dir,
                repo_root,
                args.chrom,
                genome_relpath,
                cons_dir_relpath,
                args.window_stride,
            )
            print(f"{tool}: prepared {count} window-scan chunks")


if __name__ == "__main__":
    main()
