#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <optional>

struct Clause { std::vector<int> lit; };

enum class Result { SAT, UNSAT, UNKNOWN };

struct Stats {
    uint64_t decisions = 0, propagations = 0, conflicts = 0;
    double   millis    = 0.0;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual std::string name() const = 0;
    virtual Result solve(const std::vector<Clause>&, int num_vars,
                         Stats& out) = 0;
};
