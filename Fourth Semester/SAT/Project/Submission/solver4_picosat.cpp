#include "solver.hpp"

extern "C" {
  #include "picosat/picosat.h"
}

class SolverPico final : public Solver {
public:
    std::string name() const override { return "PicoSAT-ref"; }

    Result solve(const std::vector<Clause>& cls, int n, Stats& st) override {
        auto start = std::chrono::steady_clock::now();

        PicoSAT *ps = picosat_init();
        picosat_set_verbosity(ps, 0);
        picosat_set_global_default_phase(ps, 0);

        for (const Clause& c : cls) {
            for (int lit : c.lit) picosat_add(ps, lit);
            picosat_add(ps, 0);
        }

        int res = picosat_sat(ps, -1); 

        st.decisions    = picosat_decisions(ps);
        st.propagations = picosat_propagations(ps);
        st.conflicts    = 0;         
        st.millis       = std::chrono::duration<double,std::milli>
                          (std::chrono::steady_clock::now() - start).count();

        picosat_reset(ps);

        return   res == PICOSAT_SATISFIABLE ? Result::SAT
               : res == PICOSAT_UNSATISFIABLE ? Result::UNSAT
               : Result::UNKNOWN;
    }
};
