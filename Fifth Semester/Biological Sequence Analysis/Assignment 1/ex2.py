# This program scans a DNA sequence and identifies restriction sites
# for three enzymes: PpuMI, MspAII, and MslI. Each enzyme recognizes
# a specific DNA motif and cuts the DNA at a defined position.
#
# Input:  txt file (.txt) containing a DNA sequence
# Output: Text file listing all cut positions and matching subsequences
#
# Example usage:
#   python ex2.py input.txt output.txt
#
# Author: Giovanni Filomeno K12315325

import sys
import re

# Define recognition patterns using regular expressions.
# Ambiguity codes (IUPAC) are expanded to corresponding character sets.
enzyme_patterns = {
    "PpuMI": {
        "pattern": r"R[GT]G[AT]WCCY".replace("R", "[AG]").replace("Y", "[CT]").replace("W", "[AT]"),
        "cut_offset": 2  # position after G (2nd character)
    },
    "MspAII": {
        "pattern": r"C[AC]G[GT]C[GT]G",
        "cut_offset": 3  # after the middle G
    },
    "MslI": {
        "pattern": r"CA[CT]NNNN[AG]TG".replace("N", "[ACGT]"),
        "cut_offset": 6  # after the 6th position (N^)
    }
}


def read_dna_sequence(filename):
    """Reads a DNA sequence file (FASTA-like), skipping header lines."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Ignore header lines starting with '>'
    seq = ''.join([l.strip().upper() for l in lines if not l.startswith('>')])
    return seq


def find_sites(sequence, enzyme_name, pattern, cut_offset):
    """Finds all restriction sites for a given enzyme pattern."""
    matches = []
    for match in re.finditer(pattern, sequence):
        start = match.start() + cut_offset  # Calculate cut position
        matches.append((start + 1, match.group()))  # +1 for 1-based indexing
    return matches


def main():
    if len(sys.argv) != 3:
        print("Usage: python restriction_sites.py input.txt output.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read the DNA sequence
    dna_sequence = read_dna_sequence(input_file)

    # Search for each enzyme’s restriction sites
    results = []
    for enzyme, info in enzyme_patterns.items():
        sites = find_sites(dna_sequence, enzyme, info["pattern"], info["cut_offset"])
        for pos, seq in sites:
            results.append(f"{enzyme} cuts at position {pos}: {seq}")

    # Write results to output file
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')

    print(f"Done. Found {len(results)} restriction sites in total.")


if __name__ == "__main__":
    main()

# Example with chr22.txt
# >hg38_dna range=chr22:37200001-40600000 5'pad=0 3'pad=0 strand=+ repeatMasking=none
# CTCTGGGGCCTGTTGAGCCAGCAGTTCCCCTGAGCAAATATTGACACATT
# TGCTGGCCTTTAAAGCGGACAGGAGGGTGGAGAGGCCACATCCCAGCTCT
# TCCCCTGCTAGGATCCGATACACCCCACCCCACCGTAGGCCTCAGTTTCT
# ...................................................
# ===================================
# From terminal
# python ex2.py chr22.txt chr22_output.txt
# Results in chr22_output.txt
# -----------------------------------
# PpuMI cuts at position 2884: AGGATCCC
# PpuMI cuts at position 2933: GGGTTCCC
# PpuMI cuts at position 4102: ATGTTCCC
# PpuMI cuts at position 5657: GGGAACCC
# PpuMI cuts at position 6144: GTGTTCCT
# PpuMI cuts at position 8495: GTGAACCT
# PpuMI cuts at position 9030: ATGATCCC
# PpuMI cuts at position 9345: AGGAACCC
# .....................................
# -----------------------------------