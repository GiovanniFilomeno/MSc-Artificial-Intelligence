import sys

# Check if DNA valid
def is_valid_dna(seq):
    # base in ATCG: check if base is A, T, C, G
    # all gives true if all char pass the check, otherwise False
    # .upper(): to make case-insensitive
    return all(base in "ATCG" for base in seq.upper())

# Transcribing DNA -> mRNA 
def transcribe_to_mrna(dna_seq):
    complement = {"A": "U", "T": "A", "C": "G", "G": "C"} # mapping the complement
    # for every base, convert DNA to mRNA based on the map
    # per everybase, do the conversion, then join
    # polymerasi goes antiparallel to template 
    return "".join(complement[b] for b in dna_seq.upper())[::-1]  # invert direction 5'->3'

# Translation: mRNA -> amminoacidi
def translate_to_protein(mrna_seq):
    codon_table = { # map 3 mRNA base to amnioacid 
        "AUG": "M",
        "UUU": "F", "UUC": "F",
        "UUA": "L", "UUG": "L",
        "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
        "UAU": "Y", "UAC": "Y",
        "UGU": "C", "UGC": "C",
        "UGG": "W",
        "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
        "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAU": "H", "CAC": "H",
        "CAA": "Q", "CAG": "Q",
        "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AUU": "I", "AUC": "I", "AUA": "I",
        "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAU": "N", "AAC": "N",
        "AAA": "K", "AAG": "K",
        "AGU": "S", "AGC": "S",
        "AGA": "R", "AGG": "R",
        "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
        "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAU": "D", "GAC": "D",
        "GAA": "E", "GAG": "E",
        "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        "UAA": "STOP", "UAG": "STOP", "UGA": "STOP",
    }

    start = mrna_seq.find("AUG") # find the starting seguence, if not return -1
    if start == -1:
        return ""
    protein = []
    for i in range(start, len(mrna_seq), 3): # from start to end, +3 jump
        codon = mrna_seq[i:i+3]
        if len(codon) < 3: #if seguence not valid
            break
        aa = codon_table.get(codon, "") # convert based to map
        if aa == "STOP": # if stop seguence, then end
            break
        if aa:
            protein.append(aa)
    return "".join(protein)

# Read Fasta 
def read_fasta(file_path):
    sequences = {}
    header = None
    seq = []
    with open(file_path) as f:
        for line in f:
            line = line.strip() # remove spaces or newline
            if not line:
                continue
            if line.startswith(">"): # if starts with > then it is a header
                if header: # if header is initialized, then previous sequence is finished
                    sequences[header] = "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            sequences[header] = "".join(seq)
    return sequences

# Write output
def write_output(results, out_path):
    with open(out_path, "w") as f:
        for header, (dna, mrna, protein) in results.items():
            f.write(f">{header}\n")
            f.write(f"DNA-Sequence: {dna}\n")
            f.write(f"RNA-Sequence: {mrna}\n")
            f.write(f"AA-Sequence: {protein}\n\n")

# Main
def main():
    if len(sys.argv) != 3:
        print("Usage: ex05_K12315325.py <input_file> <output_file>")
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]
    sequences = read_fasta(input_file)
    results = {}

    for header, dna in sequences.items():
        if not is_valid_dna(dna):
            print(f"Warning: invalid DNA sequence in {header}")
            continue
        mrna = transcribe_to_mrna(dna)
        protein = translate_to_protein(mrna)
        results[header] = (dna, mrna, protein)

    write_output(results, output_file)

if __name__ == "__main__":
    main()
