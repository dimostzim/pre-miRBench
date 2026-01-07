#!/usr/bin/env python3
import os
import argparse
import requests
from bs4 import BeautifulSoup
import time
import random
from tqdm import tqdm
import multiprocessing
from Bio import SeqIO
import re
import sys


def get_page(url, retry_count=2):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    for attempt in range(retry_count + 1):
        try:
            time.sleep(random.uniform(0.3, 1.0))
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.text
            if response.status_code == 429:
                time.sleep((attempt + 1) * random.uniform(5, 10))
        except Exception:
            if attempt < retry_count:
                time.sleep(random.uniform(1, 3))
    return None


def download_fasta(url, output_path):
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {url} (status {response.status_code})")
    with open(output_path, 'wb') as f:
        f.write(response.content)


def load_fasta_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences


def scrape_hsa_mirna_ids():
    url = "https://mirgenedb.org/browse/hsa"
    html = get_page(url)
    if not html:
        raise RuntimeError("Failed to fetch browse page")

    soup = BeautifulSoup(html, "html.parser")
    mir_ids = []

    for a in soup.find_all("a", href=re.compile(r"/show/hsa/")):
        text = a.get_text().strip()
        if text and text.startswith("Hsa-") and text not in mir_ids:
            mir_ids.append(text)

    return mir_ids


def parse_orthologues(html):
    soup = BeautifulSoup(html, "html.parser")

    for tr in soup.find_all("tr"):
        th = tr.find("th")
        if th and "Orthologues" in th.get_text():
            td = tr.find("td")
            if td:
                return [a.get_text().strip() for a in td.find_all("a") if a.get_text().strip()]
    return []


def find_sequence(target_id, fasta_sequences, seq_type="primary"):
    suffix = "_pri" if seq_type == "primary" else "_pre"

    candidates = [
        target_id,
        f"{target_id}{suffix}",
        f"{target_id}{suffix.replace('_', '-')}",
        target_id.replace("-", "_") + suffix,
        f"{target_id}-v1{suffix}",
        f"{target_id}-v1{suffix.replace('_', '-')}",
    ]

    for candidate in candidates:
        if candidate in fasta_sequences:
            return candidate, fasta_sequences[candidate]
    return None, None


def process_mirna(mir_id, output_folder, pri_sequences, pre_sequences):
    pri_folder = os.path.join(output_folder, "primary")
    pre_folder = os.path.join(output_folder, "precursor")
    pri_file = os.path.join(pri_folder, f"{mir_id}.fasta")
    pre_file = os.path.join(pre_folder, f"{mir_id}.fasta")

    if os.path.exists(pri_file) and os.path.exists(pre_file):
        return "skipped"

    main_seq_id, main_sequence = find_sequence(mir_id, pri_sequences, "primary")
    if not main_seq_id:
        return f"error: sequence not found"

    parts = mir_id.split("-")
    species = parts[0].lower()
    rest_of_id = mir_id[len(parts[0]) + 1:]
    url = f"https://mirgenedb.org/show/{species}/{rest_of_id}"

    html = get_page(url)

    if not html and "-v" not in mir_id:
        html = get_page(f"https://mirgenedb.org/show/{species}/{rest_of_id}-v1")

    if not html:
        return "error: page not found"

    orthologues_list = parse_orthologues(html)

    confirmed_orthologues = {}
    for orth_id in orthologues_list:
        orth_seq_id, orth_sequence = find_sequence(orth_id, pri_sequences, "primary")
        if orth_seq_id:
            confirmed_orthologues[orth_seq_id] = orth_sequence

    with open(pri_file, "w") as f:
        f.write(f">{main_seq_id}\n{main_sequence}\n")
        for orth_seq_id, orth_sequence in confirmed_orthologues.items():
            f.write(f">{orth_seq_id}\n{orth_sequence}\n")

    all_seq_ids = [main_seq_id] + list(confirmed_orthologues.keys())

    with open(pre_file, "w") as f:
        for seq_id in all_seq_ids:
            base_name = seq_id.rstrip("_pri").rstrip("-pri")
            pre_seq_id, pre_sequence = find_sequence(base_name, pre_sequences, "precursor")

            if pre_seq_id:
                pre_header = seq_id.replace("_pri", "_pre").replace("-pri", "_pre")
                if not pre_header.endswith("_pre"):
                    pre_header = f"{seq_id}_pre"
                f.write(f">{pre_header}\n{pre_sequence}\n")

    return f"success: {len(confirmed_orthologues)} orthologues"


def worker_wrapper(args):
    return process_mirna(*args)


def main():
    parser = argparse.ArgumentParser(description="Scrape MirGeneDB human miRNA orthologues.")
    parser.add_argument("--output_folder", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers (default: 4)")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading FASTA files")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "primary"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "precursor"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "downloads"), exist_ok=True)

    pri_fasta = os.path.join(args.output_folder, "downloads", "all_primary.fa")
    pre_fasta = os.path.join(args.output_folder, "downloads", "all_precursor.fa")

    if not args.skip_download or not os.path.exists(pri_fasta):
        print("Downloading primary sequences...")
        download_fasta("https://mirgenedb.org/fasta/ALL?pri=1", pri_fasta)

    if not args.skip_download or not os.path.exists(pre_fasta):
        print("Downloading precursor sequences...")
        download_fasta("https://mirgenedb.org/fasta/ALL?pre=1", pre_fasta)

    pri_sequences = load_fasta_sequences(pri_fasta)
    pre_sequences = load_fasta_sequences(pre_fasta)

    if not pri_sequences or not pre_sequences:
        raise RuntimeError("Failed to load FASTA files")

    mir_ids = scrape_hsa_mirna_ids()
    if not mir_ids:
        raise RuntimeError("No miRNA IDs found")

    valid_ids = [mir_id for mir_id in mir_ids
                 if find_sequence(mir_id, pri_sequences, "primary")[0] is not None]

    print(f"Processing {len(valid_ids)} miRNAs...")

    tasks = [(mir_id, args.output_folder, pri_sequences, pre_sequences) for mir_id in valid_ids]

    with multiprocessing.Pool(processes=args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(worker_wrapper, tasks), total=len(tasks)):
            pass

    print(f"Output: {args.output_folder}/{{primary,precursor}}/")


if __name__ == "__main__":
    main()
