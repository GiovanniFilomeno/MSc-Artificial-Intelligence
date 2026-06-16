#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept> // Per std::runtime_error

// Includi solver.hpp per le definizioni di base (Clause, Result, Stats, Solver)
#include "solver.hpp" 
// Includi solver2_watched_debug.hpp per la definizione di Solver2
#include "solver2_watched_debug.cpp"


// Funzione per leggere una formula CNF da file
bool read_cnf(const std::string& filename, std::vector<Clause>& clauses, int& num_vars, int& num_clauses_expected) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    std::cerr << "[INFO] main_debug: Reading CNF file: " << filename << std::endl;

    clauses.clear();
    num_vars = 0;
    num_clauses_expected = 0;
    std::string line;
    bool p_line_found = false;

    while (std::getline(file, line)) {
        // Rimuovi eventuali caratteri di ritorno a capo \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue; // Salta linee vuote

        std::stringstream ss(line);
        char type;
        ss >> type;

        if (type == 'c') { // Commento
            continue;
        } else if (type == 'p') { // Problem line
            std::string format;
            ss >> format >> num_vars >> num_clauses_expected;
            if (format != "cnf") {
                std::cerr << "Error: Unsupported format. Expected 'p cnf'." << std::endl;
                file.close();
                return false;
            }
            p_line_found = true;
            std::cerr << "[INFO] main_debug: Problem line: vars=" << num_vars << ", clauses_expected=" << num_clauses_expected << std::endl;
        } else { 
            if (!p_line_found) {
                 std::cerr << "Warning: Data found before 'p cnf' line, or malformed line (treating as start of clause if possible): " << line << std::endl;
                 // If there is no p-line, we could try to infer num_vars, but that is risky.
                 // For now, if the p-line was not found and this is not a comment, treat it as a clause.
                 // This allows formats without a p-line, but num_vars must be inferred later.
                 ss.clear();
                 ss.str(line); // Reload the full line to read it again as literals.
            } else { // p_line_found, so this should be a clause.
                 ss.clear();
                 ss.str(line); // Reload the full line.
            }


            Clause current_clause;
            int lit;
            while (ss >> lit) {
                if (lit == 0) { 
                    break;
                }
                current_clause.lit.push_back(lit);
            }
            if (!current_clause.lit.empty()) {
                clauses.push_back(current_clause);
                 std::cerr << "[INFO] main_debug: Read clause: ";
                 for(int l : current_clause.lit) std::cerr << l << " ";
                 std::cerr << "0" << std::endl;
            } else if (p_line_found && type != 'p' && type != 'c' && !line.empty() && line.find_first_not_of(" \t\n\v\f\r") != std::string::npos) {
                // The line was not a comment or p-line, but did not produce a valid clause, e.g. only "0" or empty after ss.
                std::cerr << "Warning: Line interpreted as clause but resulted in empty literal list: " << line << std::endl;
            }
        }
    }
    file.close();

    if (!p_line_found && num_vars == 0 && !clauses.empty()) {
        std::cerr << "Warning: 'p cnf' line not found. Deducing num_vars from max literal." << std::endl;
        int max_var_seen = 0;
        for(const auto& c : clauses) {
            for (int l : c.lit) {
                if (std::abs(l) > max_var_seen) {
                    max_var_seen = std::abs(l);
                }
            }
        }
        num_vars = max_var_seen;
        std::cerr << "[INFO] main_debug: Deduced num_vars = " << num_vars << std::endl;
    } else if (!p_line_found && clauses.empty()){
         std::cerr << "Warning: 'p cnf' line not found and no clauses read. Assuming empty formula if num_vars is also 0." << std::endl;
    }


    if (clauses.size() != (size_t)num_clauses_expected && p_line_found) { // Warn only when p_line_found makes this meaningful.
        std::cerr << "Warning: Number of read clauses (" << clauses.size()
                  << ") does not match the expected count (" << num_clauses_expected << ")." << std::endl;
    }
    std::cerr << "[INFO] main_debug: Finished reading CNF. Total clauses read: " << clauses.size() << std::endl;
    return true;
}


int main() {
    std::vector<Clause> clauses;
    int num_vars = 0;
    int num_clauses_expected = 0; // Not strictly needed here if read_cnf handles it.
    std::string filename = "test-formulas/add2.in"; 

    if (!read_cnf(filename, clauses, num_vars, num_clauses_expected)) {
        std::cerr << "Failed to read CNF file: " << filename << std::endl;
        return 1;
    }

    if (num_vars == 0 && !clauses.empty()) {
        std::cerr << "Warning: num_vars is 0 after read_cnf, but clauses were read. Recheck the CNF file or read_cnf logic." << std::endl;
        int max_var_seen = 0;
        for(const auto& c : clauses) {
            for (int l : c.lit) {
                if (std::abs(l) > max_var_seen) {
                    max_var_seen = std::abs(l);
                }
            }
        }
        if (max_var_seen > 0) {
            std::cerr << "Warning: Max var seen in clauses is " << max_var_seen << ". Using this as num_vars." << std::endl;
            num_vars = max_var_seen;
        } else if (!clauses.empty()) {
             std::cerr << "Error: Clauses are present, but no var > 0 was found." << std::endl;
             return 1;
        }
    }


    Solver2 solver_instance; 
    Stats statistics;

    std::cout << "Solving " << filename << " (vars=" << num_vars << ", clauses=" << clauses.size() << ") using " << solver_instance.name() << "..." << std::endl;

    Result result = solver_instance.solve(clauses, num_vars, statistics);

    std::cout << "Risultato: " << (result == Result::SAT ? "SAT" : (result == Result::UNSAT ? "UNSAT" : "UNKNOWN")) << std::endl;
    std::cout << "Statistiche:" << std::endl;
    std::cout << "  Tempo: " << statistics.millis << " ms" << std::endl;
    std::cout << "  Decisioni: " << statistics.decisions << std::endl;
    std::cout << "  Propagazioni: " << statistics.propagations << std::endl;

    return 0;
}
