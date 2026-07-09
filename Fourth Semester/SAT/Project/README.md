# SAT Solver Project

This coursework project explores multiple SAT-solving strategies in C++17.

> **Validation status:** retained as an algorithm and debugging case study, not as featured portfolio evidence. Cross-checking the stored benchmark results against PicoSAT exposed disagreements in the custom recursive and CDCL-style solvers. Until those implementations are repaired and independently rerun, this repository makes no correctness or performance claim for them.

| Component | Intended role |
| --- | --- |
| `main.cpp` | Orchestrates solver runs over the retained test formulas. |
| `parser.hpp` | Parses DIMACS-style SAT input formulas. |
| `solver1_recursive.cpp` | Recursive DPLL-style baseline. |
| `solver2_watched.cpp` | Watched-literal propagation. |
| `solver3_cdcl.cpp` | CDCL-style conflict analysis and clause learning. |
| `solver4_picosat.cpp` | PicoSAT integration for reference comparison. |
| `test-formulas/` | Small formulas used for cross-solver checks and debugging. |

The source is useful for discussing algorithm design, state management, C/C++ integration, and—most importantly—why differential testing matters in solver engineering. Generated binaries and duplicate submission copies were removed from the public edition; the retained result files are archival evidence, not validated benchmarks.
