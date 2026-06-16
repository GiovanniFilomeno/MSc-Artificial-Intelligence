import sys

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python ex03_dotplot.py <sequence_one> <sequence_two> <window_size>")
        sys.exit(1)

    seq1 = sys.argv[1].lower()
    seq2 = sys.argv[2].lower()

    try:
        window = int(sys.argv[3])
    except ValueError:
        print("Error: window_size must be an integer.")
        sys.exit(1)

    if window <= 0 or window % 2 == 0:
        print("Error: window_size must be a positive odd integer.")
        sys.exit(1)

    len1, len2 = len(seq1), len(seq2)
    mid = window // 2

    # Calculate the dot plot
    plot = [[" " for _ in range(len1)] for _ in range(len2)]

    for i in range(len1 - window + 1):
        for j in range(len2 - window + 1):
            if seq1[i:i+window] == seq2[j:j+window]:
                plot[j + mid][i + mid] = "*"

    # Print header
    print("  ", end="")
    for c in seq1:
        print(c.upper(), end=" ")
    print()

    # Print matrix
    for j in range(len2):
        print(seq2[j].upper(), end=" ")
        for i in range(len1):
            print(plot[j][i], end=" ")
        print()

if __name__ == "__main__":
    main()
