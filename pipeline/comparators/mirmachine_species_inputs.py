#!/usr/bin/env python3
"""
Inputs for `mirmachine_retrained.py genome-cutoffs`: one row per species with its genome, the miRNA
families MirMachine searches in it, and its MirGeneDB precursors.

Node      the species' nearest ancestor in MirMachine's tree (meta/tree.newick) whose name holds a node
          label MirMachine accepts (MirMachine.py --print-all-nodes: name parts split on "_", longer
          than 2 characters, capitalised). This gives Drosophila for the three Drosophila genomes and
          Neognathae for chicken, the nodes used for the held-out genomes.
Families  MirMachine's own selection for that node: MirMachine.py --dry writes the families and the
          families lost along the lineage to data/yamls/<species>.yaml; the MirMachine workflow
          searches the families minus the losses.
Genome    the species' sequences ("<species>__" prefix) copied from the combined benchmark genome
          (needs its .fai index).
Precursors the species' MirGeneDB precursor BED, chromosome names prefixed as in the genome.

Run with the MirMachine environment's Python (newick, yaml).
"""
import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path

import newick
import yaml

# The panel names these species by common name or genus only; leaf names as in MirMachine's tree.
LEAF_ALIASES = {"war": "W_argentea", "eba": "E_babai", "asp": "A_sp", "gsp": "G_sp"}
NODE_LABEL = re.compile("[A-Z][a-z]+")


def leaf_name(code, species_name):
    if code in LEAF_ALIASES:
        return LEAF_ALIASES[code]
    match = re.search(r"\(([^)]+)\)", species_name)
    genus, epithet = (match.group(1) if match else species_name).split()[:2]
    return f"{genus[0]}_{epithet}"


def node_labels(name):
    """MirMachine's accepted labels in a node name (_walk_tree_nodes)."""
    return [part for part in (name or "").split("_") if len(part) > 2 and NODE_LABEL.match(part)]


def species_node(tree, leaf):
    """Nearest ancestor with an accepted label; checks that MirMachine's keyword search resolves to it."""
    leaves = [node for node in tree.walk() if node.is_leaf and node.name == leaf]
    if len(leaves) != 1:
        raise ValueError(f"leaf {leaf} found {len(leaves)} times in the MirMachine tree")
    ancestor = leaves[0].ancestor
    while ancestor is not None and not node_labels(ancestor.name):
        ancestor = ancestor.ancestor
    if ancestor is None:
        raise ValueError(f"no accepted node above {leaf}")
    keyword = node_labels(ancestor.name)[0]
    # MirMachine (_search_tree_for_keyword) takes the first node in walk order whose name contains the keyword;
    # it may be a composite ancestor (Neodermata -> Tricladida_Neodermata) but must lie on the species' lineage.
    first = next(node for node in tree.walk() if node.name is not None and re.search(keyword.title(), node.name))
    lineage, node = [], ancestor
    while node is not None:
        lineage.append(node)
        node = node.ancestor
    if not any(first is node for node in lineage):
        raise ValueError(f"MirMachine would resolve {keyword} to {first.name}, outside the lineage of {leaf}")
    return keyword


def mirmachine_families(mirmachine, node, species):
    """Families MirMachine searches for this node: its --dry YAML families minus the lineage losses."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "dummy.fa").write_text(">dummy\nACGU\n")
        subprocess.run([mirmachine, "--node", node, "--species", species, "--genome", "dummy.fa", "--model", "combined", "--dry"],
                       cwd=tmp, check=True, capture_output=True)
        config = yaml.safe_load(open(Path(tmp) / "data" / "yamls" / f"{species}.yaml"))
    return sorted(set(config["mirnas"]) - set(config.get("losses", [])))


def copy_species_genome(genome, fai_rows, prefix, output):
    """Copy every sequence named <prefix>* byte for byte, using the .fai offsets."""
    with open(genome, "rb") as source, open(output, "wb") as out:
        for name, length, offset, line_bases, line_width in fai_rows:
            if not name.startswith(prefix):
                continue
            full_lines, rest = divmod(length, line_bases)
            size = full_lines * line_width + (rest + line_width - line_bases if rest else 0)
            source.seek(offset)
            out.write(f">{name}\n".encode())
            out.write(source.read(size))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--panel", required=True, help="panel.tsv with columns code, species")
    parser.add_argument("--mirmachine", required=True, help="MirMachine.py")
    parser.add_argument("--tree", required=True, help="MirMachine meta/tree.newick")
    parser.add_argument("--genome", required=True, help="combined genome FASTA with .fai")
    parser.add_argument("--bed-dir", required=True, help="directory with <code>/<code>-precursors.normalized.bed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--species", default=None, help="comma-separated codes; default: every species with training positives")
    args = parser.parse_args()

    if args.species:
        codes = args.species.split(",")
    else:
        with open(args.dataset, newline="") as handle:
            codes = sorted({row["species"] for row in csv.DictReader(handle) if row["split"] == "train" and row["label"] == "1"})
    names = {row["code"]: row["species"] for row in csv.DictReader(open(args.panel), delimiter="\t")}
    tree = newick.read(args.tree)[0]
    fai_rows = []
    for line in open(f"{args.genome}.fai"):
        name, length, offset, line_bases, line_width = line.split("\t")[:5]
        fai_rows.append((name, int(length), int(offset), int(line_bases), int(line_width)))
    sequence_names = {row[0] for row in fai_rows}

    out = Path(args.output_dir)
    for sub in ("genomes", "families", "precursors"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    table = out / "species_table.tsv"
    if table.exists():
        raise SystemExit(f"{table} exists; use a new output directory")
    rows = []
    for code in codes:
        leaf = leaf_name(code, names[code])
        node = species_node(tree, leaf)
        families = mirmachine_families(args.mirmachine, node, code)
        (out / "families" / f"{code}.txt").write_text("\n".join(families) + "\n")
        bed_out = out / "precursors" / f"{code}.bed"
        with open(Path(args.bed_dir) / code / f"{code}-precursors.normalized.bed") as source, open(bed_out, "w") as handle:
            for line in source:
                fields = line.rstrip("\n").split("\t")
                fields[0] = f"{code}__{fields[0]}"
                if fields[0] not in sequence_names:
                    raise ValueError(f"{code}: precursor chromosome {fields[0]} is not in the genome")
                handle.write("\t".join(fields) + "\n")
        genome_out = out / "genomes" / f"{code}.fa"
        copy_species_genome(args.genome, fai_rows, f"{code}__", genome_out)
        rows.append({"species": code, "leaf": leaf, "node": node, "n_families": len(families),
                     "families": str(out / "families" / f"{code}.txt"), "genome": str(genome_out), "precursor_bed": str(bed_out)})
        print(f"{code} {leaf} -> {node}: {len(families)} families", flush=True)
    with open(table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} species -> {table}")


if __name__ == "__main__":
    main()
