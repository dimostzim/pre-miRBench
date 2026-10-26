#!/usr/bin/env python3
"""
MirMachine retrained on the pre-miRBench training partition.

Follows the MirMachine procedure (Umu et al. 2023, STAR Methods; MirMachine 0.3.0.5
workflow), with the pre-miRBench training partition in place of all MirGeneDB species:

build     one covariance model (CM) per miRNA family represented among training
          positives: training precursors aligned with mafft-xinsi, near-identical
          sequences removed with esl-weight -f --idf 0.90 --rna, consensus structure
          from RNAalifold -r --noPS, cmbuild and cmcalibrate with default settings.
cutoffs   MirMachine trusted cutoffs, fitted on the data the models were built from:
          the library is searched against all training windows (cmsearch --incE 0.2,
          both strands); included hits of at least 50 nt are true positives when the
          model's family matches a training precursor the hit overlaps on the same
          strand, and false positives otherwise. Per family, the cutoff is the hit
          bitscore maximizing sqrt(TP/(TP+FN+1) * TP/(TP+FP)), where FN are true-positive
          hits below the cutoff (determine_mirmachine_cutoffs.R).
score     benchmark windows, given strand only: score = best (bitscore - family cutoff)
          over included hits of at least 50 nt; called when the score is >= 0. With
          --fallback-work-dir, families without any fitting hit (cutoff 0 from
          genome-cutoffs) take the cutoff of that work directory instead.
genome    whole-genome search restricted to the families MirMachine searches at the
          species' node (read from a MirMachine GFF header), with the same inclusion,
          length and cutoff rules; writes all included hits and the filtered hits as
          GFF with score = bitscore - cutoff.
genome-cutoffs
          MirMachine trusted cutoffs fitted as MirMachine fitted its own: each training
          species' genome is searched with the families MirMachine searches for that
          species (mirmachine_species_inputs.py), and hits are labelled by locus. Only
          training information is used: hits overlapping a validation or test window
          (either strand, plus --margin) are dropped; a hit is a true positive when it
          overlaps a training precursor of its family on the same strand; hits on other
          annotated MirGeneDB precursors (not in the training partition) are dropped;
          every other hit is a false positive. Cutoffs use the same formula as cutoffs.
"""
import argparse
import csv
import re
import subprocess
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path

NO_HIT_SCORE = -1000.0
MIN_HIT_LENGTH = 50
INCLUSION_EVALUE = 0.2


def read_dataset(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def safe_name(family):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", family)


def read_fasta(path):
    records, name, chunks = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].split()[0], []
        elif line:
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def write_stockholm(path, aligned, structure=None):
    with open(path, "w") as handle:
        handle.write("# STOCKHOLM 1.0\n")
        for name, sequence in aligned:
            handle.write(f"{name} {sequence}\n")
        if structure is not None:
            handle.write(f"#=GC SS_cons {structure}\n")
        handle.write("//\n")


def read_stockholm(path):
    aligned = {}
    order = []
    for line in open(path):
        if not line.strip() or line.startswith("#") or line.startswith("//"):
            continue
        name, sequence = line.split()
        if name not in aligned:
            order.append(name)
            aligned[name] = ""
        aligned[name] += sequence
    return [(name, aligned[name]) for name in order]


def alifold_structure(alignment):
    output = subprocess.run(["RNAalifold", "-r", "--noPS", str(alignment)], capture_output=True, text=True, check=True).stdout
    return output.splitlines()[1].split()[0]


def build_family(job):
    family, sequences, model_dir = job
    stem = Path(model_dir) / safe_name(family)
    unaligned = stem.with_suffix(".fa")
    with open(unaligned, "w") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">s{index}\n{sequence}\n")
    if len(sequences) == 1:
        filtered = [("s0", sequences[0])]
    else:
        aligned_fasta = stem.with_suffix(".aln.fa")
        with open(aligned_fasta, "w") as handle:
            subprocess.run(["mafft-xinsi", "--quiet", "--thread", "1", str(unaligned)], stdout=handle, check=True)
        aligned = [(name, sequence.upper().replace("T", "U")) for name, sequence in read_fasta(aligned_fasta)]
        alignment = stem.with_suffix(".aln.sto")
        write_stockholm(alignment, aligned)
        weighted = stem.with_suffix(".weighted.sto")
        with open(weighted, "w") as handle:
            subprocess.run(["esl-weight", "-f", "--idf", "0.90", "--rna", str(alignment)], stdout=handle, check=True)
        filtered = read_stockholm(weighted)
    filtered_alignment = stem.with_suffix(".filtered.sto")
    write_stockholm(filtered_alignment, filtered)
    structure = alifold_structure(filtered_alignment)
    stockholm = stem.with_suffix(".sto")
    write_stockholm(stockholm, filtered, structure)
    model = stem.with_suffix(".cm")
    subprocess.run(["cmbuild", "-F", "-n", family, str(model), str(stockholm)], check=True, capture_output=True)
    subprocess.run(["cmcalibrate", "--cpu", "1", str(model)], check=True, capture_output=True)
    return family, len(sequences), len(filtered)


def build(args):
    families = defaultdict(set)
    for row in read_dataset(args.dataset):
        if row["split"] == "train" and row["label"] == "1" and row["precursor_sequence"]:
            families[row["family_id"]].add(row["precursor_sequence"].upper().replace("T", "U"))
    model_dir = Path(args.work_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(family, sorted(sequences), str(model_dir)) for family, sequences in sorted(families.items())]
    with ProcessPoolExecutor(max_workers=args.cpus) as executor:
        built = list(executor.map(build_family, jobs))
    library = Path(args.work_dir) / "library.cm"
    with open(library, "w") as out:
        for family, _, _ in built:
            out.write((model_dir / f"{safe_name(family)}.cm").read_text())
    with open(Path(args.work_dir) / "families.tsv", "w") as handle:
        handle.write("family_id\ttraining_sequences\tsequences_after_esl_weight\n")
        handle.writelines(f"{family}\t{n}\t{kept}\n" for family, n, kept in built)
    print(f"built {len(built)} family models -> {library}")


def write_windows(rows, path):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(f">{row['record_id']}\n{row['sequence'].upper().replace('U', 'T')}\n")


def included_hits(table):
    """MirMachine keeps hits passing the inclusion threshold (cmsearch '!') of at least 50 nt."""
    hits = []
    for line in open(table):
        if line.startswith("#"):
            continue
        fields = line.split()
        if fields[16] != "!":
            continue
        seq_from, seq_to = int(fields[7]), int(fields[8])
        start, end = min(seq_from, seq_to), max(seq_from, seq_to)
        if end - start < MIN_HIT_LENGTH:
            continue
        hits.append({"target": fields[0], "family": fields[2], "start": start - 1, "end": end, "strand": fields[9], "bitscore": float(fields[14])})
    return hits


def cmsearch(cm_file, sequences, table, cpus, extra=()):
    subprocess.run(
        ["cmsearch", "--incE", str(INCLUSION_EVALUE), "--cpu", str(cpus), "--noali", "--tblout", str(table), *extra, str(cm_file), str(sequences)],
        check=True, stdout=subprocess.DEVNULL,
    )
    return included_hits(table)


def trusted_cutoff(hits):
    """determine_mirmachine_cutoffs.R: the hit bitscore maximizing sqrt(TPR * PPV), first maximum in hit order."""
    best_value, best_score = None, -1.0
    for value in [bitscore for bitscore, _ in hits]:
        tp = sum(1 for bitscore, is_tp in hits if bitscore >= value and is_tp)
        fn = sum(1 for bitscore, is_tp in hits if bitscore < value and is_tp)
        fp = sum(1 for bitscore, is_tp in hits if bitscore >= value and not is_tp)
        tpr = tp / (tp + fn + 1)
        ppv = tp / (tp + fp) if tp + fp else 0.0
        score = sqrt(tpr * ppv)
        if score > best_score:
            best_value, best_score = value, score
    return best_value


def precursor_interval(row):
    """Precursor interval in window coordinates (window orientation = '+')."""
    start, end = int(row["start"]), int(row["end"])
    target_start, target_end = int(row["target_start"]), int(row["target_end"])
    if row["strand"] == "-":
        return end - target_end, end - target_start
    return target_start - start, target_end - start


def cutoffs(args):
    work = Path(args.work_dir)
    rows = [row for row in read_dataset(args.dataset) if row["split"] == "train"]
    by_id = {row["record_id"]: row for row in rows}
    windows = work / "cutoff_training_windows.fa"
    write_windows(rows, windows)
    hits = cmsearch(work / "library.cm", windows, work / "cutoff_training_windows.tbl", args.cpus)
    per_family = defaultdict(list)
    for hit in sorted(hits, key=lambda h: -h["bitscore"]):
        row = by_id[hit["target"]]
        is_tp = False
        if row["label"] == "1" and row["family_id"] == hit["family"] and hit["strand"] == "+":
            pre_start, pre_end = precursor_interval(row)
            is_tp = hit["start"] < pre_end and pre_start < hit["end"]
        per_family[hit["family"]].append((hit["bitscore"], is_tp))
    families = [line.split("\t")[0] for line in open(work / "families.tsv").read().splitlines()[1:]]
    with open(work / "trusted_cutoffs.tsv", "w") as handle:
        handle.write("family_id\ttrusted\ttp_hits\tfp_hits\n")
        for family in families:
            data = per_family.get(family, [])
            # MirMachine reads missing cutoffs as 0 (defaultdict(int)).
            value = trusted_cutoff(data) if data else 0.0
            handle.write(f"{family}\t{value}\t{sum(t for _, t in data)}\t{sum(not t for _, t in data)}\n")
    print(f"cutoffs for {len(families)} families -> {work / 'trusted_cutoffs.tsv'}")


def load_cutoffs(work):
    with open(Path(work) / "trusted_cutoffs.tsv") as handle:
        return {row["family_id"]: float(row["trusted"]) for row in csv.DictReader(handle, delimiter="\t")}


def score(args):
    work = Path(args.work_dir)
    cutoff = load_cutoffs(work)
    if args.fallback_work_dir:
        with open(work / "trusted_cutoffs.tsv") as handle:
            unfitted = [row["family_id"] for row in csv.DictReader(handle, delimiter="\t") if int(row["tp_hits"]) + int(row["fp_hits"]) == 0]
        fallback = load_cutoffs(args.fallback_work_dir)
        cutoff.update({family: fallback[family] for family in unfitted})
        print(f"{len(unfitted)} families without a fitting hit use the cutoffs in {args.fallback_work_dir}")
    rows = [row for row in read_dataset(args.dataset) if row["split"] in set(args.splits.split(","))]
    windows = work / f"score_{args.name}.fa"
    write_windows(rows, windows)
    hits = cmsearch(work / "library.cm", windows, work / f"score_{args.name}.tbl", args.cpus, ("--toponly",))
    best = {}
    for hit in hits:
        margin = hit["bitscore"] - cutoff[hit["family"]]
        best[hit["target"]] = max(best.get(hit["target"], NO_HIT_SCORE), margin)
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tool", "split", "record_id", "species", "label", "score", "called"])
        for row in rows:
            value = best.get(row["record_id"], NO_HIT_SCORE)
            writer.writerow([args.tool_name, row["split"], row["record_id"], row["species"], row["label"], value, int(value >= 0)])
    print(f"scored {len(rows)} records; {len(best)} with an included hit -> {args.output}")


def node_families(gff):
    for line in open(gff):
        if line.startswith("# microRNA families searched:"):
            return {name.strip(" '").removesuffix(".PRE").upper() for name in line.split(":", 1)[1].strip(" []\n").split(",")}
    raise ValueError(f"No family list in {gff}")


def genome(args):
    work = Path(args.work_dir)
    cutoff = load_cutoffs(work)
    wanted = node_families(args.mirmachine_gff)
    selected = sorted(family for family in cutoff if family.upper() in wanted)
    names = work / f"genome_{args.name}.families.txt"
    names.write_text("\n".join(selected) + "\n")
    node_cm = work / f"genome_{args.name}.cm"
    with open(node_cm, "w") as handle:
        subprocess.run(["cmfetch", "-f", str(work / "library.cm"), str(names)], stdout=handle, check=True)
    hits = cmsearch(node_cm, args.genome, work / f"genome_{args.name}.tbl", args.cpus)
    with open(args.all_gff, "w") as all_out, open(args.filtered_gff, "w") as filtered_out:
        for hit in sorted(hits, key=lambda h: (h["target"], h["start"])):
            margin = hit["bitscore"] - cutoff[hit["family"]]
            line = "\t".join([hit["target"], "MirMachine_retrained", "microRNA", str(hit["start"] + 1), str(hit["end"]), f"{margin:.2f}", hit["strand"], ".", f"gene_id={hit['family'].upper()}.PRE;bitscore={hit['bitscore']}"]) + "\n"
            all_out.write(line)
            if margin >= 0:
                filtered_out.write(line)
    print(f"{args.name}: {len(selected)} of {len(wanted)} node families have a retrained model; {len(hits)} included hits")


def interval_index(intervals):
    """{chrom: (starts, records, longest)} for overlap queries; records are (start, end, payload)."""
    by_chrom = defaultdict(list)
    for chrom, start, end, payload in intervals:
        by_chrom[chrom].append((start, end, payload))
    index = {}
    for chrom, records in by_chrom.items():
        records.sort(key=lambda record: record[0])
        index[chrom] = ([record[0] for record in records], records, max(end - start for start, end, _ in records))
    return index


def overlapping_intervals(index, chrom, start, end, margin=0):
    """Records overlapping [start - margin, end + margin) on chrom (half-open coordinates)."""
    if chrom not in index:
        return []
    starts, records, longest = index[chrom]
    low, high = start - margin, end + margin
    first = bisect_left(starts, low - longest)
    last = bisect_left(starts, high)
    return [record for record in records[first:last] if record[0] < high and low < record[1]]


def label_genome_hit(hit, training, evaluation, annotated, margin):
    """Label one genome hit for cutoff fitting from training information only (see genome-cutoffs)."""
    if overlapping_intervals(evaluation, hit["target"], hit["start"], hit["end"], margin):
        return "excluded_evaluation_window"
    on_training = overlapping_intervals(training, hit["target"], hit["start"], hit["end"])
    if any(strand == hit["strand"] and family == hit["family"] for _, _, (strand, family) in on_training):
        return "TP"
    if not on_training and overlapping_intervals(annotated, hit["target"], hit["start"], hit["end"]):
        return "excluded_other_annotation"
    return "FP"


def search_species_genome(job):
    """cmsearch one training species' genome with its MirMachine families; a finished table is reused."""
    species, genome_fasta, families_file, work, library_families, cpus = job
    out = Path(work) / "genome_cutoffs"
    table = out / f"{species}.tbl"
    if not table.exists():
        partial = out / f"{species}.tbl.partial"
        if partial.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            partial.rename(out / f"{species}.tbl_INCOMPLETE_{stamp}")
        wanted = {line.strip().upper() for line in open(families_file) if line.strip()}
        selected = sorted(family for family in library_families if family.upper() in wanted)
        names = out / f"{species}.families.txt"
        names.write_text("\n".join(selected) + "\n")
        species_cm = out / f"{species}.cm"
        with open(species_cm, "w") as handle:
            subprocess.run(["cmfetch", "-f", str(Path(work) / "library.cm"), str(names)], stdout=handle, check=True)
        subprocess.run(
            ["cmsearch", "--incE", str(INCLUSION_EVALUE), "--cpu", str(cpus), "--noali", "--tblout", str(partial), str(species_cm), str(genome_fasta)],
            check=True, stdout=subprocess.DEVNULL,
        )
        partial.rename(table)
    return species, included_hits(table)


def genome_cutoffs(args):
    work = Path(args.work_dir)
    output = work / "trusted_cutoffs.tsv"
    if output.exists():
        raise SystemExit(f"{output} exists; use a new work directory")
    (work / "genome_cutoffs").mkdir(exist_ok=True)
    rows = read_dataset(args.dataset)
    training_species = {row["species"] for row in rows if row["split"] == "train" and row["label"] == "1"}
    held_out_species = {row["species"] for row in rows if row["split"].startswith("test_heldout_species")}
    species_table = list(csv.DictReader(open(args.species_table), delimiter="\t"))
    searched = {entry["species"] for entry in species_table}
    if searched - training_species or searched & held_out_species:
        raise SystemExit(f"species table must list training species only: {sorted(searched - training_species | searched & held_out_species)}")
    training = interval_index((row["chrom"], int(row["target_start"]), int(row["target_end"]), (row["strand"], row["family_id"]))
                              for row in rows if row["split"] == "train" and row["label"] == "1")
    evaluation = interval_index((row["chrom"], int(row["start"]), int(row["end"]), row["record_id"])
                                for row in rows if row["split"] != "train")
    annotated = []
    for entry in species_table:
        for line in open(entry["precursor_bed"]):
            fields = line.split("\t")
            annotated.append((fields[0], int(fields[1]), int(fields[2]), fields[3]))
    annotated = interval_index(annotated)
    library_families = [line.split("\t")[0] for line in open(work / "families.tsv").read().splitlines()[1:]]
    jobs = [(entry["species"], entry["genome"], entry["families"], str(work), library_families, args.cpus) for entry in species_table]
    per_family = defaultdict(list)
    excluded = defaultdict(int)
    counts = defaultdict(int)
    with ProcessPoolExecutor(max_workers=args.parallel) as executor, open(work / "genome_cutoffs" / "labelled_hits.tsv", "w") as labelled:
        labelled.write("species\ttarget\tstart\tend\tstrand\tfamily\tbitscore\tlabel\n")
        for species, hits in executor.map(search_species_genome, jobs):
            for hit in hits:
                label = label_genome_hit(hit, training, evaluation, annotated, args.margin)
                counts[label] += 1
                labelled.write(f"{species}\t{hit['target']}\t{hit['start']}\t{hit['end']}\t{hit['strand']}\t{hit['family']}\t{hit['bitscore']}\t{label}\n")
                if label in ("TP", "FP"):
                    if overlapping_intervals(evaluation, hit["target"], hit["start"], hit["end"]):
                        raise RuntimeError(f"hit used for fitting overlaps a validation or test window: {species} {hit}")
                    per_family[hit["family"]].append((hit["bitscore"], label == "TP"))
                else:
                    excluded[hit["family"]] += 1
            print(f"{species}: {len(hits)} included hits", flush=True)
    with open(output, "w") as handle:
        handle.write("family_id\ttrusted\ttp_hits\tfp_hits\texcluded_hits\n")
        for family in library_families:
            data = sorted(per_family.get(family, []), key=lambda item: -item[0])
            # MirMachine reads missing cutoffs as 0 (defaultdict(int)).
            value = trusted_cutoff(data) if data else 0.0
            handle.write(f"{family}\t{value}\t{sum(t for _, t in data)}\t{sum(not t for _, t in data)}\t{excluded[family]}\n")
    print(f"{len(species_table)} species; hits by label: {dict(counts)}; hits used for fitting that overlap a validation or test window: 0")
    print(f"cutoffs for {len(library_families)} families -> {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "cutoffs", "score", "genome", "genome-cutoffs"):
        command = sub.add_parser(name)
        command.add_argument("--work-dir", required=True)
        command.add_argument("--cpus", type=int, default=8)
        if name != "genome":
            command.add_argument("--dataset", required=True)
    sub.choices["score"].add_argument("--splits", required=True)
    sub.choices["score"].add_argument("--name", required=True)
    sub.choices["score"].add_argument("--output", required=True)
    sub.choices["score"].add_argument("--tool-name", default="mirmachine_retrained")
    sub.choices["score"].add_argument("--fallback-work-dir", default=None)
    genome_cutoffs_cmd = sub.choices["genome-cutoffs"]
    genome_cutoffs_cmd.add_argument("--species-table", required=True, help="mirmachine_species_inputs.py output (species, genome, families, precursor_bed).")
    genome_cutoffs_cmd.add_argument("--margin", type=int, default=100, help="Hits within this distance of a validation or test window are dropped.")
    genome_cutoffs_cmd.add_argument("--parallel", type=int, default=1, help="Species searched at the same time, each with --cpus threads.")
    genome_cmd = sub.choices["genome"]
    genome_cmd.add_argument("--genome", required=True)
    genome_cmd.add_argument("--name", required=True)
    genome_cmd.add_argument("--mirmachine-gff", required=True, help="GFF from a default MirMachine run on this genome (family list in its header).")
    genome_cmd.add_argument("--all-gff", required=True)
    genome_cmd.add_argument("--filtered-gff", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    {"build": build, "cutoffs": cutoffs, "score": score, "genome": genome, "genome-cutoffs": genome_cutoffs}[args.command](args)


if __name__ == "__main__":
    main()
