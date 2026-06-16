# SAT Solver Project

This project implements and compares multiple SAT-solving strategies in C++17.

| Component | What it does | Skills demonstrated |
| --- | --- | --- |
| `main.cpp` | Runs the solver workflow over test formulas. | CLI-style systems workflow, benchmarking setup. |
| `parser.hpp` | Parses SAT input formulas. | Input parsing, formula representation, defensive validation. |
| `solver1_recursive.cpp` | Implements a recursive baseline solver. | DPLL-style recursion, backtracking, satisfiability basics. |
| `solver2_watched.cpp` | Implements watched-literal propagation. | Efficient Boolean constraint propagation, watchlists, solver optimization. |
| `solver3_cdcl.cpp` | Implements a CDCL-style solver. | Conflict analysis, implication reasoning, clause learning concepts. |
| `solver4_picosat.cpp` | Integrates PicoSAT. | External solver integration, comparative evaluation, C/C++ interfacing. |
| `test-formulas/` | Contains benchmark/test formulas. | Validation, regression testing, solver comparison. |

Portfolio takeaway: this is one of the strongest systems projects in the repository because it combines algorithm design, C++ implementation, debugging, and solver benchmarking.
