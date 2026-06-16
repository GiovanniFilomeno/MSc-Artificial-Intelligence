#include "solver.hpp"
#include <vector>

class Solver1 : public Solver {
public:
    std::string name() const override { return "Recursive DPLL"; }

    Result solve(const std::vector<Clause>& in, int n, Stats& st) override {
        clauses  = in;
        num_vars = n;
        assign.assign(n + 1, 0);
        decs = props = 0;

        const auto start = std::chrono::steady_clock::now();
        Result res      = dpll();                
        st.millis       = elapsed(start);
        st.decisions    = decs;
        st.propagations = props;
        return res;
    }

private:
    std::vector<Clause> clauses;
    std::vector<int8_t> assign;   
    int           num_vars = 0;
    uint64_t      decs = 0, props = 0;

    Result dpll() {
        std::vector<int> local_trail;     

        bool changed;
        do {
            changed = false;
            for (const Clause& c : clauses) {
                int unassigned = 0, last_lit = 0;
                bool clause_sat = false;

                for (int lit : c.lit) {
                    int val = assign[std::abs(lit)];
                    if (val == 0) { unassigned++; last_lit = lit; }
                    else if (val == (lit > 0 ? 1 : -1)) { clause_sat = true; break; }
                }

                if (clause_sat) continue;
                if (unassigned == 0) { undo(local_trail); return Result::UNSAT; }

                if (unassigned == 1) {     
                    int v = std::abs(last_lit);
                    assign[v] = (last_lit > 0 ? 1 : -1);
                    local_trail.push_back(v);
                    ++props; changed = true;
                }
            }
        } while (changed);

        int var = 0;
        for (int i = 1; i <= num_vars; ++i)
            if (assign[i] == 0) { var = i; break; }

        if (var == 0) { undo(local_trail); return Result::SAT;}

        ++decs;
        for (int val : { 1, -1 }) {
            assign[var] = val;
            local_trail.push_back(var);

            if (dpll() == Result::SAT) return Result::SAT;

            undo(local_trail);           
            assign[var] = 0;      
            local_trail.pop_back();
        }

        undo(local_trail);
        return Result::UNSAT;
    }

    static void undo(const std::vector<int>& trail) {
        for (int v : trail)        
            ;                                        
    }
    static double elapsed(std::chrono::steady_clock::time_point s) {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now() - s).count();
    }
};
