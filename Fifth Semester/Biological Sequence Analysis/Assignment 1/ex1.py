# This program reads a text file and searches for all palindromic words.
# A palindrome is a word that reads the same forwards and backwards,
# such as "level", or "1234321".
#
# The program:
#   1. Takes the input and output file names as command-line arguments.
#   2. Reads the text from the input file.
#   3. Extracts all alphanumeric words (ignoring punctuation and case).
#   4. Detects which words are palindromes.
#   5. Counts how many times each palindrome occurs.
#   6. Writes the results (word + frequency) to the output file.
#
# Example usage:
#   python ex1.py input.txt output.txt
#
# Author: Giovanni Filomeno K12315325

import sys
import re
from collections import Counter


# Function: is_palindrome(word)
# Checks whether a given word is a palindrome.
# - The comparison is case-insensitive (the word is lowercased before comparison).
# - Words of length 1 are ignored, since they trivially read the same.
# - Uses a two-pointer approach for efficiency (O(N/2) time, O(1) space).
def is_palindrome(word):
    word = word.lower()
    if len(word) <= 1:
        return True
    for i in range(len(word) // 2): # Complexity O(N/2) --> Approach learn in LeetCode
        if word[i] != word[-i - 1]:
            return False
    return True

# Main execution block
def main():
    # Expect exactly two command-line arguments:
    # 1) input file path
    # 2) output file path
    if len(sys.argv) != 3:
        print("Usage: python palindrome_finder.py input.txt output.txt")
        sys.exit(1)

    # Extract file names from command-line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Read the content of the input text file
    # The encoding 'utf-8' ensures compatibility with most languages.
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the text into words
    # The regular expression \b[a-zA-Z0-9]+\b extracts sequences of letters or digits bounded by word boundaries.
    # Everything is converted to lowercase for case-insensitive comparison.
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    # Filter only the words that are palindromes
    # The list comprehension applies the is_palindrome() function to each word and keeps only the True cases.
    palindromes = [w for w in words if is_palindrome(w)]

    # Count occurrences of each palindrome
    # Counter creates a dictionary-like object where:
    # key = palindrome word
    # value = number of times it appears in the text
    counts = Counter(palindromes)

    # Write the results to the output file
    # Each line of the file will have the format:
    # word: frequency
    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in counts.items():
            f.write(f"{word}: {count}\n")

if __name__ == "__main__":
    main()


# Example with pali.txt
# Level madam radar noon 1234321 test
# ===================================
# From terminal
# python ex1.py pali.txt pali_output.txt
# Results in pali_output.txt
# -----------------------------------
# level: 1
# madam: 1
# radar: 1
# noon: 1
# 1234321: 1
# -----------------------------------
