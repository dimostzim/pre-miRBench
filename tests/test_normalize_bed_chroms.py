import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NORMALIZER_PATH = ROOT / "pipeline" / "dataset" / "normalize_bed_chroms.py"

spec = importlib.util.spec_from_file_location("normalize_bed_chroms", NORMALIZER_PATH)
normalize_bed_chroms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalize_bed_chroms)


class NormalizeBedChromsTests(unittest.TestCase):
    def test_ucsc_chrom_alias_maps_assembly_name_to_refseq_fasta_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fasta = tmp / "genome.fa"
            alias = tmp / "chromAlias.txt"

            fasta.write_text(">NC_023179.1\nAAAA\n")
            alias.write_text(
                "# refseq\tassembly\tgenbank\tncbi\tucsc\n"
                "NC_023179.1\tLG1\tCM001404.1\tLG1\tchrLG1\n"
            )

            genome_names = normalize_bed_chroms.fasta_headers(fasta)
            aliases = normalize_bed_chroms.load_aliases(alias, genome_names)

            self.assertEqual(
                normalize_bed_chroms.mapped_chrom("LG1", genome_names, aliases),
                "NC_023179.1",
            )

    def test_ncbi_assembly_report_maps_sequence_name_to_fasta_accession(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fasta = tmp / "genome.fa"
            alias = tmp / "assembly_report.txt"

            fasta.write_text(
                ">NW_001939295.1 Drosophila ananassae scaffold_12916 genomic scaffold\n"
                "AAAA\n"
            )
            alias.write_text(
                "scaffold_12916\tunplaced-scaffold\tna\tna\tCH902620.1\t=\t"
                "NW_001939295.1\tPrimary Assembly\t16180835\tscaffold_12916\n"
            )

            genome_names = normalize_bed_chroms.fasta_headers(fasta)
            aliases = normalize_bed_chroms.load_aliases(alias, genome_names)

            self.assertEqual(
                normalize_bed_chroms.mapped_chrom("scaffold_12916", genome_names, aliases),
                "NW_001939295.1",
            )

    def test_fasta_header_aliases_map_combined_schmidtea_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta = Path(tmpdir) / "genome.fa"
            fasta.write_text(
                ">NNSW01000001.1 Schmidtea mediterranea strain S2 dd_Smes_g4_1, whole genome shotgun sequence\n"
                "AAAA\n"
            )

            genome_names = normalize_bed_chroms.fasta_headers(fasta)
            aliases = normalize_bed_chroms.fasta_header_aliases(fasta, genome_names)

            self.assertEqual(
                normalize_bed_chroms.mapped_chrom(
                    "NNSW01000001_dd_Smes_g4_1", genome_names, aliases
                ),
                "NNSW01000001.1",
            )


if __name__ == "__main__":
    unittest.main()
