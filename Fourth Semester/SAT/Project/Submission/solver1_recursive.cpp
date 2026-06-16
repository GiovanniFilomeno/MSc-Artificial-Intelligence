#include "solver.hpp"
#include <vector>

class Solver1 : public Solver {
public:
    // Name of the solver
    std::string name() const override { return "Recursive DPLL"; }

    // Solving method: receive the CNF and update the statistics
    Result solve(const std::vector<Clause>& in, int n, Stats& st) override {
        clauses  = in; // saving the formula
        num_vars = n; // num of variables
        assign.assign(n + 1, 0); // all variables as non assigned 
        decs = props = 0; // counter to zero

        const auto start = std::chrono::steady_clock::now();
        Result res      = dpll();        // call ing the solver        
        st.millis       = elapsed(start); //time
        st.decisions    = decs; //decision
        st.propagations = props; //propagation
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
            // loop over clause to search implications 
            for (const Clause& c : clauses) {
                int unassigned = 0; // num of literals still not assigned in the clause
                int last_lit = 0; // last literals not assigned found
                bool clause_sat = false; // flag for satistied clause

                // looping over all the literals in the clause
                for (int lit : c.lit) {
                    int val = assign[std::abs(lit)]; //taking the value assigned to the variable
                    if (val == 0) { unassigned++; last_lit = lit; } // not assigned --> cxould be literals
                    else if (val == (lit > 0 ? 1 : -1)) { clause_sat = true; break; } // true literals 
                }

                if (clause_sat) continue; //already satisfied clause --> skip
                if (unassigned == 0) { 
                    undo(local_trail); // back to local assignment
                    return Result::UNSAT; // no literal assigned and none of them is true --> clause violated
                } 

                // unit clause --> assign the variable 
                if (unassigned == 1) {     
                    int v = std::abs(last_lit);
                    assign[v] = (last_lit > 0 ? 1 : -1); // take the value with the right sign
                    local_trail.push_back(v); // save --> later for undo
                    ++props; // count propagation
                    changed = true;
                }
            }
        } while (changed);

        // Branching: finding the first not assigned variable 
        int var = 0;
        for (int i = 1; i <= num_vars; ++i)
            if (assign[i] == 0) { var = i; break; }

        // if no variable is assigned and no clause are violated --> sat
        if (var == 0) { undo(local_trail); return Result::SAT;}

        ++decs;
        // Recursive branching su var: first TRUE, then FALSE
        for (int val : { 1, -1 }) {
            assign[var] = val; // assigning the variable
            local_trail.push_back(var); // add local trail

            if (dpll() == Result::SAT) return Result::SAT;  // recurvise call, if SAT, go to the next

            undo(local_trail); // reset the implicit assignment --> unit propagation    
            assign[var] = 0;    // reset the variable
            local_trail.pop_back(); // remove from trail
        }

        undo(local_trail);
        return Result::UNSAT; // if both branch fail: UNSAT
    }

    static void undo(const std::vector<int>& trail) {
        for (int v : trail)      
            assign[v] = 0;                                        
    }
    static double elapsed(std::chrono::steady_clock::time_point s) {
        using namespace std::chrono;
        return duration<double, std::milli>(steady_clock::now() - s).count();
    }
};
