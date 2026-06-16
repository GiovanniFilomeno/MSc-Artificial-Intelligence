#pragma once
#include "solver.hpp"
#include <fstream>
#include <sstream>

inline bool read_cnf(const std::string& path,
                     std::vector<Clause>& clauses,
                     int& num_vars) {
    std::ifstream in(path);
    if (!in) return false;
    std::string tok;
    while (in >> tok) {
        if (tok == "c") { std::getline(in, tok); continue; }      // comment
        if (tok == "p") { in >> tok >> num_vars >> tok; continue; }
        Clause c;   // tok already contains first literal
        int lit = std::stoi(tok);
        while (lit != 0) { c.lit.push_back(lit); in >> lit; }
        clauses.push_back(std::move(c));
    }
    return true;
}
