#!/usr/bin/env python3
"""v3 result tables in a plain booktabs layout (user, 25 Oct: tables instead of the dot plots).

Table 1 (table_benchmark_v3): AP with its 95% bootstrap interval on the four tests.
Table 2 (table_genome_v3): genome AP over unambiguous candidates on the four held-out genomes.
Table 3 (table_genome_family_v3): genome AP and recall of unmasked MirGeneDB loci, for loci of training families and of
held-out families (recall at each method's calls; predictors: validation max-F1 threshold).
Table 4 (table_genome_calls_v3): calls per Mb and the precision of the calls.
A rule separates the predictors from the homology methods, which are a reference block (user, 26 Oct). Bold: in each
column the highest value among the predictors; the homology methods and calls per Mb are not ranked.
Inputs: benchmark_tests.csv and genome_wide.csv next to this script (make_tables_v3.py), and for the note under Table 4
the comparator summaries in comparator_summaries/ (node 4 v3/results/genome/comparators/). For each table, writes the
tabular alone (<name>.tex, for the manuscript), a standalone document (<name>_standalone.tex) and its PDF, SVG and
300-dpi PNG renders, and the same with a bold title above the table and its note below (<name>_titled.*, for the
co-author copies; in the manuscript, titles and notes go in the caption).
Needs pdflatex, pdftocairo and pdftoppm.
"""
import csv
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREDICTORS = {"premirbench_model_v3": "pre-miRBench", "mustard": "MuStARD", "deepmirgene": "deepMiRGene", "mirdnn": "mirDNN",
              "dnnpremir": "dnnPreMiR", "mire2e": "miRe2e", "deepmir": "DeepMir"}  # order of the benchmark figure (Test 4 AP)
HOMOLOGY = {"mirmachine_retrained_genome_cutoffs": "MirMachine 2", "ncortho_training_family_refs": "ncOrtho"}
NAMES = {**PREDICTORS, **HOMOLOGY}
TESTS = ["test_known_species_known_family", "test_known_species_heldout_family", "test_heldout_species_known_family",
         "test_heldout_species_heldout_family"]
GENOMES = [("dme", "D. melanogaster"), ("dan", "D. ananassae"), ("dya", "D. yakuba"), ("gga", "G. gallus")]
TITLES = {
    "table_benchmark_v3": "Benchmark: average precision on the four test sets (95\\% interval)",
    "table_genome_v3": "Genome-wide: average precision on the four held-out genomes",
    "table_genome_family_v3": "Genome-wide: average precision and recall for loci of training and held-out families",
    "table_genome_calls_v3": "Genome-wide: calls per Mb and precision of the calls on the four held-out genomes",
}
PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[scaled]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{booktabs}
\begin{document}
"""


def ap_text(value):
    return r"\textless0.001" if value < 0.0005 else f"{value:.3f}"


def bold_best(cells, values, methods):
    """Bold the highest predictor value; the homology methods are never bold."""
    best = max(v for v, m in zip(values, methods) if m in PREDICTORS and v is not None)
    return [rf"\textbf{{{c}}}" if m in PREDICTORS and v == best else c for c, v, m in zip(cells, values, methods)]


def tabular(header, methods, blocks):
    """blocks: (label or None, columns), each column the cell texts in method order."""
    width = 1 + len(blocks[0][1])
    lines = [r"\begin{tabular}{l" + "c" * (width - 1) + "}", r"\toprule", *header]
    for label, columns in blocks:
        lines.append(r"\midrule")
        if label:
            lines.append(rf"\multicolumn{{{width}}}{{l}}{{\textit{{{label}}}}} \\")
        for index, method in enumerate(methods):
            if index and method in HOMOLOGY and methods[index - 1] not in HOMOLOGY:
                lines.append(r"\midrule")
            lines.append(" & ".join([NAMES[method]] + [column[index] for column in columns]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render(name, body, note=()):
    (HERE / f"{name}.tex").write_text(body)
    note_rows = "".join(rf"\\[2pt]\multicolumn{{1}}{{@{{}}l@{{}}}}{{\footnotesize {line}}}" + "\n" for line in note)
    titled = "\\begin{tabular}{@{}c@{}}\n\\textbf{" + TITLES[name] + "}\\\\[6pt]\n" + body + note_rows + "\\end{tabular}\n"
    for output, document in ((name, body), (f"{name}_titled", titled)):
        standalone = HERE / f"{output}_standalone.tex"
        standalone.write_text(PREAMBLE + document + "\\end{document}\n")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", standalone.name], cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL)
        pdf = standalone.with_suffix(".pdf")
        subprocess.run(["pdftocairo", "-svg", pdf.name, f"{output}.svg"], cwd=HERE, check=True)
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", pdf.name, output], cwd=HERE, check=True)
        pdf.rename(HERE / f"{output}.pdf")
        for extension in (".aux", ".log"):
            standalone.with_suffix(extension).unlink()


def benchmark_table():
    rows = {(r["tool"], r["test"]): r for r in csv.DictReader(open(HERE / "benchmark_tests.csv"))}
    methods = [*PREDICTORS, "mirmachine_retrained_genome_cutoffs"]
    columns = []
    for test in TESTS:
        values = [float(rows[(m, test)]["ap_mean"]) for m in methods]
        cells = bold_best([f"{v:.3f}" for v in values], values, methods)
        columns.append([f"{c} ({float(rows[(m, test)]['ap_ci_low']):.3f}--{float(rows[(m, test)]['ap_ci_high']):.3f})"
                        for c, m in zip(cells, methods)])
    header = [r" & \multicolumn{2}{c}{Known species} & \multicolumn{2}{c}{Held-out species} \\",
              r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
              r"Method & Known family & Held-out family & Known family & Held-out family \\"]
    render("table_benchmark_v3", tabular(header, methods, [(None, columns)]))


def genome_tables():
    rows = {(r["tool"], r["genome"]): r for r in csv.DictReader(open(HERE / "genome_wide.csv"))}
    methods = [*PREDICTORS, *HOMOLOGY]

    def column(field, sp, kind):
        values = [float(rows[(m, sp)][field]) if (m, sp) in rows else None for m in methods]
        if kind == "calls":
            return ["--" if v is None else f"{v:.1f}" for v in values]
        text = (lambda v: f"{v:.3f}") if kind == "recall" else ap_text  # recall is a share of loci: exactly 0 when none is called
        return bold_best(["--" if v is None else text(v) for v in values], values, methods)

    genome_names = " & ".join(rf"\textit{{{s}}}" for _, s in GENOMES)
    columns = [column("ap_unambiguous_candidates", sp, "ap") for sp, _ in GENOMES]
    render("table_genome_v3", tabular([rf"Method & {genome_names} \\"], methods, [(None, columns)]))

    def grouped(first, second):
        return [rf" & \multicolumn{{4}}{{c}}{{{first}}} & \multicolumn{{4}}{{c}}{{{second}}} \\",
                r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}", rf"Method & {genome_names} & {genome_names} \\"]

    ap = [column(field, sp, "ap") for field in ("ap_training_family_loci", "ap_heldout_family_loci") for sp, _ in GENOMES]
    recall = [column(field, sp, "recall") for field in ("unmasked_training_family_loci_recall", "unmasked_heldout_family_loci_recall")
              for sp, _ in GENOMES]
    render("table_genome_family_v3", tabular(grouped("Training families", "Held-out families"), methods,
                                             [("Average precision", ap), ("Recall", recall)]))

    columns = [column("calls_per_mb", sp, "calls") for sp, _ in GENOMES] + [column("precision_unambiguous_calls", sp, "ap") for sp, _ in GENOMES]
    per_hit = {}
    for path in (HERE / "comparator_summaries").glob("*.summary.csv"):
        summary = next(csv.DictReader(open(path)))
        per_hit[(summary["tool"], path.name.removesuffix(".summary.csv").rsplit("_", 1)[1])] = float(
            summary["precision_calls_on_annotated_loci"])
    mirmachine = ", ".join(f"{per_hit[('mirmachine_retrained_genome_cutoffs', sp)]:.3f}" for sp, _ in GENOMES)
    note = ["Precision: share of calls that match a MirGeneDB precursor; candidates overlapping a precursor without "
            "matching it are left out.",
            f"Per hit (share of hits overlapping a precursor): MirMachine 2 {mirmachine}; "
            f"ncOrtho {per_hit[('ncortho_training_family_refs', 'gga')]:.3f}."]
    render("table_genome_calls_v3", tabular(grouped("Calls per Mb", "Precision of the calls"), methods, [(None, columns)]),
           note)


if __name__ == "__main__":
    benchmark_table()
    genome_tables()
