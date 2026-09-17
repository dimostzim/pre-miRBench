#!/usr/bin/env python3
"""Simple entry point for building the canonical pre-miRBench dataset.

Normal usage from anywhere::

    python pipeline/build.py

The script derives all repository/data paths automatically and forwards the
actual work to build_dataset.py. Use --data-dir only when the raw/work/output
data should live somewhere other than <repo>/data.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DATASET_NAME = "mirgenedb_71"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the canonical pre-miRBench MirGeneDB-71 dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data root. Default: <repo>/data",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Approximate total CPU budget. Default: all detected CPUs.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=10.0,
        help="Negatives per positive. Default: 10.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Optional comma-separated species codes to build.",
    )
    parser.add_argument(
        "--heldout-species",
        default="gga,dme",
        help="Comma-separated held-out species. Default: gga,dme.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-species intermediates from an interrupted build.",
    )
    return parser.parse_args()


def choose_parallelism(total_jobs):
    """Convert one user-facing CPU budget into the builder's two-level parallelism."""
    total_jobs = max(1, total_jobs)

    # Keep several RNAfold workers inside each species while allowing multiple
    # species to progress concurrently. On 96 CPUs this reproduces 12 x 8.
    cpus_per_species = min(8, max(1, total_jobs // 4))
    species_jobs = max(1, total_jobs // cpus_per_species)
    species_jobs = min(12, species_jobs)

    # Do not exceed the requested total CPU budget after the cap above.
    cpus_per_species = max(1, total_jobs // species_jobs)
    cpus_per_species = min(8, cpus_per_species)
    return species_jobs, cpus_per_species


def main():
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    raw_dir = data_dir / "raw" / DATASET_NAME
    panel = raw_dir / "panel.tsv"
    output_dir = data_dir / "datasets" / DATASET_NAME
    work_dir = data_dir / "work" / f"build_{DATASET_NAME}"

    if not panel.exists():
        default_raw = DEFAULT_DATA_DIR / "raw" / DATASET_NAME
        if data_dir == DEFAULT_DATA_DIR:
            download_cmd = "bash pipeline/download_data.sh"
        else:
            download_cmd = f'bash pipeline/download_data.sh "{raw_dir}"'
        raise SystemExit(
            "Raw MirGeneDB data are not present.\n"
            f"Expected: {panel}\n\n"
            "Download them first with:\n"
            f"  {download_cmd}\n"
        )

    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.ratio <= 0:
        raise SystemExit("--ratio must be positive")

    species_jobs, cpus = choose_parallelism(args.jobs)

    cmd = [
        sys.executable,
        str(PIPELINE_DIR / "build_dataset.py"),
        "--panel",
        str(panel),
        "--output-dir",
        str(output_dir),
        "--work-dir",
        str(work_dir),
        "--ratio",
        str(args.ratio),
        "--cpus",
        str(cpus),
        "--species-jobs",
        str(species_jobs),
        "--heldout-species",
        args.heldout_species,
    ]

    if args.species:
        cmd.extend(["--species", args.species])
    if args.resume:
        cmd.append("--reuse-existing")

    print("pre-miRBench build")
    print(f"data:          {data_dir}")
    print(f"panel:         {panel}")
    print(f"output:        {output_dir}")
    print(f"work:          {work_dir}")
    print(f"CPU budget:    {args.jobs}")
    print(f"species jobs:  {species_jobs}")
    print(f"CPUs/species:  {cpus}")
    print(f"ratio:         1:{args.ratio:g}")
    print()

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
