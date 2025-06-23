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
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
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
                std::cerr << "Errore: Formato non supportato. Atteso 'p cnf'." << std::endl;
                file.close();
                return false;
            }
            p_line_found = true;
            std::cerr << "[INFO] main_debug: Problem line: vars=" << num_vars << ", clauses_expected=" << num_clauses_expected << std::endl;
        } else { 
            if (!p_line_found) {
                 std::cerr << "Warning: Data found before 'p cnf' line, or malformed line (treating as start of clause if possible): " << line << std::endl;
                 // Se non c'è p-line, potremmo provare a dedurre num_vars, ma è rischioso.
                 // Per ora, se la p-line non è stata trovata, e non è un commento, la trattiamo come una clausola.
                 // Questo permette formati senza p-line, ma num_vars deve essere dedotto dopo.
                 ss.clear();
                 ss.str(line); // Ricarica l'intera linea per rileggerla come letterali
            } else { // p_line_found, quindi questa dovrebbe essere una clausola
                 ss.clear();
                 ss.str(line); // Ricarica l'intera linea
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
                // La linea non era un commento, non era la p-line, ma non ha prodotto una clausola valida (es. solo "0" o vuota dopo ss)
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


    if (clauses.size() != (size_t)num_clauses_expected && p_line_found) { // Solo se p_line_found ha senso il warning
        std::cerr << "Attenzione: Numero di clausole lette (" << clauses.size()
                  << ") non corrisponde a quello atteso (" << num_clauses_expected << ")." << std::endl;
    }
    std::cerr << "[INFO] main_debug: Finished reading CNF. Total clauses read: " << clauses.size() << std::endl;
    return true;
}


int main() {
    std::vector<Clause> clauses;
    int num_vars = 0;
    int num_clauses_expected = 0; // Non strettamente necessario qui se read_cnf lo gestisce
    std::string filename = "test-formulas/add2.in"; 

    if (!read_cnf(filename, clauses, num_vars, num_clauses_expected)) {
        std::cerr << "Fallimento nella lettura del file CNF: " << filename << std::endl;
        return 1;
    }

    if (num_vars == 0 && !clauses.empty()) {
        std::cerr << "Attenzione: num_vars è 0 dopo read_cnf, ma sono state lette delle clausole. Ricontrolla il file CNF o la logica di read_cnf." << std::endl;
        int max_var_seen = 0;
        for(const auto& c : clauses) {
            for (int l : c.lit) {
                if (std::abs(l) > max_var_seen) {
                    max_var_seen = std::abs(l);
                }
            }
        }
        if (max_var_seen > 0) {
            std::cerr << "Attenzione: Max var visto nelle clausole e' " << max_var_seen << ". Usando questo come num_vars." << std::endl;
            num_vars = max_var_seen;
        } else if (!clauses.empty()) {
             std::cerr << "Errore: Clausole presenti ma nessun var > 0 trovato." << std::endl;
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

