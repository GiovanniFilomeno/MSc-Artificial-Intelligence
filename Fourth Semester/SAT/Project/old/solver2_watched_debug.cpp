#ifndef SOLVER2_WATCHED_DEBUG_HPP
#define SOLVER2_WATCHED_DEBUG_HPP

#include "solver.hpp" // Contiene la definizione di Solver, Clause, Result, Stats
#include <vector>
#include <chrono>
#include <iostream> // Per std::cerr
#include <iomanip>  // Per std::setw

class Solver2 : public Solver {
public:
    std::string name() const override { return "Watched‑literal DPLL (Debug)"; }

    /* helper minimali */
    static inline auto   now() { return std::chrono::steady_clock::now(); }
    static inline double elapsed(const std::chrono::steady_clock::time_point &s){
        using namespace std::chrono;
        return duration<double,std::milli>(steady_clock::now() - s).count(); }
    
    static inline int lit2idx(int lit, int n_vars) {
        if (lit > 0) return lit;
        // Per lit < 0, vogliamo mappare -1 a n+1, -2 a n+2, ..., -n_vars a 2*n_vars
        // Esempio: n=3. lit=-1 -> idx=4. lit=-2 -> idx=5. lit=-3 -> idx=6.
        // La formula n_vars - lit funziona:
        // n=3, lit=-1: 3 - (-1) = 4
        // n=3, lit=-2: 3 - (-2) = 5
        // n=3, lit=-3: 3 - (-3) = 6
        // Gli indici usati in 'watch' andranno da 1 a 2*n_vars.
        // Il vettore 'watch' sarà dimensionato 2*n_vars + 1, quindi l'indice 0 non è usato.
        return n_vars - lit; 
    }


    /* entry ---------------------------------------------------------------- */
    Result solve(const std::vector<Clause>& in,int n,Stats& st) override {
        std::cerr << "[DEBUG] Solver2::solve BEGIN. n_vars = " << n << ", num_clauses_in = " << in.size() << std::endl;
        if(n == 0 && in.empty()){ 
            std::cerr << "[DEBUG] Solver2::solve: Empty formula (n=0, in.empty()). Returning SAT." << std::endl;
            st.decisions     = 0;
            st.propagations  = 0;
            st.conflicts     = 0;
            st.millis        = 0.0;
            return Result::SAT;
        }

        decisions = propagations = conflicts = 0; 
        std::cerr << "[DEBUG] Solver2::solve: Initializing stats: decisions=0, propagations=0, conflicts=0" << std::endl;

        if(!build(in,n)){                 
            std::cerr << "[DEBUG] Solver2::solve: build() returned false. Returning UNSAT." << std::endl;
            st.decisions     = 0;
            st.propagations  = 0;
            st.conflicts     = 0;
            st.millis        = 0.0;
            return Result::UNSAT;
        }
        std::cerr << "[DEBUG] Solver2::solve: build() returned true." << std::endl;
        std::cerr << "[DEBUG] Solver2::solve: Stats after build: decisions=" << decisions << ", propagations=" << propagations << ", conflicts=" << conflicts << std::endl;


        auto start_time = now();
        Result res = search();
        st.millis = elapsed(start_time);
        st.decisions = decisions;
        st.propagations = propagations;
        
        std::cerr << "[DEBUG] Solver2::solve: search() returned " << (res == Result::SAT ? "SAT" : "UNSAT") << std::endl;
        std::cerr << "[DEBUG] Solver2::solve END. Final Stats: decisions=" << st.decisions
                  << ", propagations=" << st.propagations << ", time=" << st.millis << "ms" << std::endl;
        return res;
    }

private:
    struct WClause {
        std::vector<int> lit;
        int w1, w2; 
    };

    std::vector<WClause>          cls;
    std::vector<std::vector<int>> watch;          
    std::vector<int8_t>           val, phase;     
    std::vector<int>              trail;          
    std::vector<size_t>           level_ofs;      
    std::vector<int>              reason;         

    std::vector<double> activity;  double var_inc = 1.0;  const double decay = 0.95;

    size_t   conflicts   = 0; 
    int      num_vars    = 0;
    uint64_t decisions   = 0,
             propagations= 0; 

    bool build(const std::vector<Clause>& in, int n){
        std::cerr << "[DEBUG] Solver2::build BEGIN. n_vars = " << n << std::endl;
        num_vars = n;

        for(size_t i = 0; i < in.size(); ++i) {
            if(in[i].lit.empty()) {
                std::cerr << "[DEBUG] Solver2::build: Found empty clause in input at index " << i << ". Formula UNSAT." << std::endl;
                return false;
            }
        }

        val.assign(num_vars + 1, 0);    
        phase.assign(num_vars + 1, 1);  
        reason.assign(num_vars + 1, -1);
        activity.assign(num_vars + 1, 0.0); 
        trail.clear();
        level_ofs = {0}; 

        cls.clear(); 
        cls.reserve(in.size());
        watch.assign(2 * num_vars + 1, std::vector<int>()); 

        std::cerr << "[DEBUG] Solver2::build: Initialized val, phase, reason, activity, trail, level_ofs, cls, watch." << std::endl;
        std::cerr << "[DEBUG] Solver2::build: watch vector size: " << watch.size() << std::endl;


        for(const Clause& c_in : in){
            if (c_in.lit.empty()) { 
                 std::cerr << "[DEBUG] Solver2::build: Encountered empty clause during clause processing. UNSAT." << std::endl;
                return false;
            }
            
            std::vector<int> processed_lits;
            // std::vector<bool> seen_vars(num_vars + 1, false); // Non usata, può essere rimossa se la logica di 'tautology' non la necessita
            bool tautology = false;
            for (int lit_val : c_in.lit) { // Rinominato 'lit' a 'lit_val' per evitare shadowing
                if (tautology) break;
                int var = std::abs(lit_val);
                if (var == 0 || var > num_vars) { 
                    std::cerr << "[ERROR] Solver2::build: Invalid literal " << lit_val << " in clause. Max var is " << num_vars << std::endl;
                    continue; 
                }
                // La semplificazione basata su val[var] != 0 è rischiosa qui se fatta prima che tutte le unità siano propagate.
                // La gestione delle tautologie (x V -x) e duplicati è più sicura.
                
                bool already_present = false;
                for(int plit : processed_lits) {
                    if(plit == lit_val) already_present = true;
                    if(plit == -lit_val) tautology = true; // Trovato x e -x
                }
                if (tautology) break;

                if(!already_present) {
                    processed_lits.push_back(lit_val);
                }
            }

            if (tautology) {
                std::cerr << "[DEBUG] Solver2::build: Clause ";
                for(int l_val : c_in.lit) std::cerr << l_val << " "; // Usato l_val
                std::cerr << "is a tautology. Skipping." << std::endl;
                continue; 
            }
            
            if (processed_lits.empty() && !c_in.lit.empty()) { // La clausola originale non era vuota, ma lo è diventata (es. solo duplicati che formano tautologia)
                 std::cerr << "[DEBUG] Solver2::build: Clause ";
                for(int l_val : c_in.lit) std::cerr << l_val << " "; // Usato l_val
                std::cerr << "became empty after simplification (e.g. x -x x). Potentially UNSAT if not handled by tautology rule." << std::endl;
                // Se una clausola diventa vuota e non era una tautologia, è UNSAT.
                // Ma se era (x -x), è una tautologia. Se era (x x -x), tautologia.
                // Se era (x x) e x diventa falso, allora diventa vuota.
                // Questa logica di semplificazione deve essere attenta.
                // Per ora, se diventa vuota e non era tautologia, è un problema.
                // Se processed_lits è vuota, significa che la clausola originale era vuota o una tautologia.
                // Il check iniziale for(const Clause& c: in) if(c.lit.empty()) return false; dovrebbe coprire le clausole vuote in input.
                // Se una clausola non vuota diventa vuota qui, era una tautologia.
                 return false; // Una clausola non tautologica che diventa vuota è un conflitto.
            }
             if (processed_lits.empty() && c_in.lit.empty()){
                // Questo caso è già gestito all'inizio di build.
             } else if (processed_lits.empty()){ // Era una tautologia che ha portato a una lista vuota
                 // Già gestito dal continue per tautology
             }


            int w1_idx = 0;
            int w2_idx = (processed_lits.size() > 1 ? 1 : 0); 
            
            // Assicurati che gli indici w1, w2 siano validi se processed_lits è vuota (non dovrebbe succedere qui)
            if (processed_lits.empty()) {
                 // Questo non dovrebbe accadere se le clausole vuote/tautologiche sono gestite correttamente
                std::cerr << "[DEBUG] Solver2::build: Skipping add for an effectively empty/tautological clause that wasn't caught." << std::endl;
                continue;
            }

            cls.push_back({processed_lits, w1_idx, w2_idx});
            int cid = (int)cls.size() - 1;

            std::cerr << "[DEBUG] Solver2::build: Adding clause cid=" << cid << ": ";
            for(int l_val : cls[cid].lit) std::cerr << l_val << " "; // Usato l_val
            std::cerr << " (w1=" << cls[cid].lit[w1_idx] << "@idx" << w1_idx
                      << ", w2=" << cls[cid].lit[w2_idx] << "@idx" << w2_idx << ")" << std::endl;

            add_watch(cls[cid].lit[w1_idx], cid);
            if (processed_lits.size() > 1) {
                add_watch(cls[cid].lit[w2_idx], cid);
            }


            for(int lit_val : cls[cid].lit) { // Usato lit_val
                if (std::abs(lit_val) <= num_vars && std::abs(lit_val) > 0) { 
                    activity[std::abs(lit_val)] += 1.0;
                }
            }
        }
        std::cerr << "[DEBUG] Solver2::build: Finished processing input clauses. Num processed clauses in cls: " << cls.size() << std::endl;

        std::cerr << "[DEBUG] Solver2::build: Enqueuing unit clauses from initial set (level 0)..." << std::endl;
        for(int cid = 0; cid < (int)cls.size(); ++cid) {
            if(cls[cid].lit.size() == 1) { 
                int unit_lit = cls[cid].lit[0];
                std::cerr << "[DEBUG] Solver2::build: Found initial unit clause cid=" << cid << " with literal " << unit_lit << std::endl;
                
                int var_unit = std::abs(unit_lit);
                if (var_unit == 0 || var_unit > num_vars) {
                     std::cerr << "[ERROR] Solver2::build: Invalid unit literal " << unit_lit << " in clause cid=" << cid << std::endl;
                     continue;
                }

                if (val[var_unit] == (unit_lit > 0 ? -1 : 1)) {
                    std::cerr << "[DEBUG] Solver2::build: Conflict with pre-existing assignment for unit literal " << unit_lit << ". UNSAT." << std::endl;
                    conflicts++; 
                    return false; 
                }
                if (val[var_unit] == 0) { 
                    if (!enqueue(unit_lit, cid)) { 
                        std::cerr << "[DEBUG] Solver2::build: enqueue(" << unit_lit << ", cid=" << cid << ") failed (conflict). UNSAT." << std::endl;
                        conflicts++; 
                        return false; 
                    }
                } else {
                     std::cerr << "[DEBUG] Solver2::build: Unit literal " << unit_lit << " already consistently assigned. Skipping enqueue." << std::endl;
                }
            }
        }
        std::cerr << "[DEBUG] Solver2::build: Finished enqueuing initial unit clauses." << std::endl;
        std::cerr << "[DEBUG] Solver2::build: Trail after initial enqueues: ";
        for(int l_val : trail) std::cerr << l_val << "(R" << (std::abs(l_val) > 0 && std::abs(l_val) <=num_vars ? reason[std::abs(l_val)] : -99) << ") "; // Usato l_val
        std::cerr << std::endl;


        std::cerr << "[DEBUG] Solver2::build: Calling propagate() for consistency check at level 0." << std::endl;
        int initial_conflict_cid = propagate();
        if (initial_conflict_cid != -1) {
            std::cerr << "[DEBUG] Solver2::build: propagate() at level 0 found conflict with clause cid=" << initial_conflict_cid << ". UNSAT." << std::endl;
            conflicts++;
            return false; 
        }
        std::cerr << "[DEBUG] Solver2::build: propagate() at level 0 found no conflict." << std::endl;
        std::cerr << "[DEBUG] Solver2::build END. Returning true." << std::endl;
        return true;
    }

    void add_watch(int lit_val, int cid){ // Rinominato lit a lit_val
        if (lit_val == 0) {
            std::cerr << "[ERROR] Solver2::add_watch: Attempted to add watch for literal 0 (cid=" << cid << ")" << std::endl;
            return;
        }
        int var = std::abs(lit_val);
        if (var == 0 || var > num_vars) { // Aggiunto var == 0 check
             std::cerr << "[ERROR] Solver2::add_watch: Literal " << lit_val << " (var " << var <<") out of bounds (num_vars=" << num_vars << ") for cid=" << cid << std::endl;
            return;
        }
        int idx = lit2idx(lit_val, num_vars);
        std::cerr << "[DEBUG] Solver2::add_watch: Adding watch for lit=" << lit_val << " (idx=" << idx << ") to clause cid=" << cid << std::endl;
        if (idx < 0 || idx >= watch.size()) { // Aggiunto idx < 0 check
            std::cerr << "[ERROR] Solver2::add_watch: Index " << idx << " for literal " << lit_val << " is out of bounds for watch vector (size=" << watch.size() << ")" << std::endl;
            return;
        }
        watch[idx].push_back(cid);
    }

    bool enqueue(int lit_val, int why_cid){ // Rinominato lit a lit_val
        int v = std::abs(lit_val);
        int8_t s = (lit_val > 0 ? 1 : -1);
        std::cerr << "[DEBUG] Solver2::enqueue: Attempting to enqueue lit=" << lit_val << " (var=" << v << ", sign=" << (int)s << "), reason_cid=" << why_cid;
        if (v > 0 && v <= num_vars) std::cerr << ". Current val[" << v << "]=" << (int)val[v];
        std::cerr << std::endl;


        if (v == 0 || v > num_vars) {
            std::cerr << "[ERROR] Solver2::enqueue: Invalid literal " << lit_val << " (var " << v << ")" << std::endl;
            return false; 
        }


        if(val[v] == 0){ 
            val[v] = s;
            phase[v] = s; 
            reason[v] = why_cid;
            trail.push_back(lit_val);
            propagations++;
            std::cerr << "[DEBUG] Solver2::enqueue: Successfully enqueued lit=" << lit_val << ". Trail size=" << trail.size() << ", propagations=" << propagations << std::endl;
            return true;
        } else if (val[v] == s) { 
            std::cerr << "[DEBUG] Solver2::enqueue: lit=" << lit_val << " already consistently assigned. No action." << std::endl;
            return true; 
        } else { 
            std::cerr << "[DEBUG] Solver2::enqueue: CONFLICT! lit=" << lit_val << " (var=" << v << ") already assigned to " << (int)val[v] << ". Cannot assign to " << (int)s << "." << std::endl;
            return false; 
        }
    }

    int propagate(){
        std::cerr << "[DEBUG] Solver2::propagate BEGIN. Current trail size: " << trail.size() << std::endl;
        if (!level_ofs.empty()) {
             std::cerr << "[DEBUG] Solver2::propagate: Current decision level: " << level_ofs.size() -1 << ", propagation starts from trail_offset: " << level_ofs.back() << std::endl;
        }
        
        // current_trail_pointer è l'indice nel trail del prossimo letterale le cui conseguenze devono essere propagate.
        // Inizia dall'offset dell'ultimo livello di decisione, perché i precedenti sono già stati processati.
        size_t current_trail_pointer = 0; 
        if (!level_ofs.empty()) {
            current_trail_pointer = level_ofs.back(); 
        }


        while(current_trail_pointer < trail.size()){
            int assigned_true_lit = trail[current_trail_pointer];
            int lit_false = -assigned_true_lit; 
            current_trail_pointer++; 

            std::cerr << "[DEBUG] Solver2::propagate: Processing assignments from trail. Current assigned_true_lit=" << assigned_true_lit
                      << " (so lit_false=" << lit_false << "). Trail pointer now at " << current_trail_pointer << "/" << trail.size() << std::endl;
            
            int var_abs_lit_false = std::abs(lit_false);
            if (var_abs_lit_false == 0 || var_abs_lit_false > num_vars) {
                std::cerr << "[ERROR] Solver2::propagate: Invalid var_abs_lit_false=" << var_abs_lit_false << " from lit_false=" << lit_false << std::endl;
                continue;
            }

            int watch_idx = lit2idx(lit_false, num_vars);
            if (watch_idx < 0 || watch_idx >= watch.size()) { 
                 std::cerr << "[ERROR] Solver2::propagate: Invalid watch_idx=" << watch_idx << " for lit_false=" << lit_false << std::endl;
                 continue; 
            }
            std::vector<int>& wl = watch[watch_idx]; 

            std::cerr << "[DEBUG] Solver2::propagate: lit_false=" << lit_false << " (idx=" << watch_idx << "). Watch list size: " << wl.size() << std::endl;

            size_t i = 0;
            while (i < wl.size()) {
                int cid = wl[i];
                if (cid < 0 || cid >= cls.size()){
                    std::cerr << "[ERROR] Solver2::propagate: Invalid cid=" << cid << " from watch list." << std::endl;
                    wl[i] = wl.back(); // Tentativo di recupero rimuovendo il cid problematico
                    wl.pop_back();
                    continue;
                }
                WClause &c = cls[cid]; 

                std::cerr << "[DEBUG] Solver2::propagate:  -> Checking clause cid=" << cid << ": ";
                for(int l_val : c.lit) std::cerr << l_val << " "; // Usato l_val
                std::cerr << "(w1_idx=" << c.w1 << " val=" << (c.w1 < c.lit.size() ? c.lit[c.w1] : -999) 
                          << ", w2_idx=" << c.w2 << " val=" << (c.w2 < c.lit.size() ? c.lit[c.w2] : -999) << ")" << std::endl;


                if (c.w1 < 0 || c.w1 >= c.lit.size() || c.w2 < 0 || c.w2 >= c.lit.size()) {
                    std::cerr << "[ERROR] Solver2::propagate: Invalid w1/w2 indices in clause cid=" << cid 
                              << " w1="<<c.w1 << ", w2="<<c.w2 << ", lit.size="<<c.lit.size() << std::endl;
                    i++; 
                    continue;
                }

                int current_false_watched_lit_val_in_clause; // Rinominato
                int other_watched_lit_val_in_clause; // Rinominato

                if (c.lit[c.w1] == lit_false) {
                    current_false_watched_lit_val_in_clause = c.lit[c.w1]; 
                    other_watched_lit_val_in_clause = c.lit[c.w2];
                } else if (c.lit[c.w2] == lit_false) {
                    current_false_watched_lit_val_in_clause = c.lit[c.w2]; 
                    other_watched_lit_val_in_clause = c.lit[c.w1];
                } else {
                    std::cerr << "[ERROR] Solver2::propagate: Clause cid=" << cid << " is in watch list for " << lit_false
                              << " but neither w1(" << c.lit[c.w1] << ") nor w2(" << c.lit[c.w2] << ") matches." << std::endl;
                    i++; 
                    continue;
                }
                std::cerr << "[DEBUG] Solver2::propagate:     current_false_watched=" << current_false_watched_lit_val_in_clause
                          << ", other_watched=" << other_watched_lit_val_in_clause << std::endl;


                int other_var = std::abs(other_watched_lit_val_in_clause);
                if (other_var > 0 && other_var <= num_vars && 
                    val[other_var] == (other_watched_lit_val_in_clause > 0 ? 1 : -1)) {
                    std::cerr << "[DEBUG] Solver2::propagate:     Clause cid=" << cid << " SAT by other_watched_lit=" << other_watched_lit_val_in_clause << std::endl;
                    i++; 
                    continue;
                }

                bool moved_watch = false;
                for(size_t k=0; k < c.lit.size(); ++k){
                    
                    if (k == (size_t)c.w1 || k == (size_t)c.w2) continue; // Non può essere uno dei due watch attuali


                    int L_new = c.lit[k];
                    int L_new_var = std::abs(L_new);

                    if (L_new_var > 0 && L_new_var <= num_vars && 
                        (val[L_new_var] == 0 || val[L_new_var] == (L_new > 0 ? 1 : -1))) {

                        std::cerr << "[DEBUG] Solver2::propagate:     Found new watch L_new=" << L_new << " for cid=" << cid
                                  << " to replace " << current_false_watched_lit_val_in_clause << std::endl;

                        if (c.lit[c.w1] == current_false_watched_lit_val_in_clause) {
                            c.w1 = k;
                        } else { 
                            c.w2 = k;
                        }
                        add_watch(L_new, cid);
                        moved_watch = true;
                        break; 
                    }
                }

                if(moved_watch){
                    wl[i] = wl.back(); 
                    wl.pop_back();     
                    std::cerr << "[DEBUG] Solver2::propagate:     Moved watch for cid=" << cid << ". Removed from wl of " << lit_false
                              << ". wl size now " << wl.size() << std::endl;
                    continue; 
                }

                std::cerr << "[DEBUG] Solver2::propagate:     Could not move watch for " << current_false_watched_lit_val_in_clause
                          << " in cid=" << cid << ". Checking other_watched_lit=" << other_watched_lit_val_in_clause << std::endl;

                int other_watched_var = std::abs(other_watched_lit_val_in_clause);
                if (other_watched_var == 0 || other_watched_var > num_vars) {
                     std::cerr << "[ERROR] Solver2::propagate: Invalid other_watched_lit " << other_watched_lit_val_in_clause << " in cid=" << cid << std::endl;
                     i++; continue; 
                }

                if(val[other_watched_var] == 0){
                    std::cerr << "[DEBUG] Solver2::propagate:     Clause cid=" << cid << " becomes unit. Enqueuing " << other_watched_lit_val_in_clause << std::endl;
                    if(!enqueue(other_watched_lit_val_in_clause, cid)) { 
                        std::cerr << "[DEBUG] Solver2::propagate:     CONFLICT from enqueue(" << other_watched_lit_val_in_clause << ") for cid=" << cid << ". Returning conflicting_cid=" << cid << std::endl;
                        conflicts++;
                        return cid; 
                    }
                    i++; 
                }
                else { 
                    std::cerr << "[DEBUG] Solver2::propagate:     CONFLICT! All literals false in cid=" << cid
                              << " (lit_false=" << lit_false << ", other_watched=" << other_watched_lit_val_in_clause << " is also false)."
                              << " Returning conflicting_cid=" << cid << std::endl;
                    conflicts++;
                    return cid; 
                }
            } 
        } 

        std::cerr << "[DEBUG] Solver2::propagate END. No conflict found. Returning -1." << std::endl;
        return -1; 
    }

    void bump_clause(int cid){ 
        if (cid < 0 || cid >= cls.size()) {
            std::cerr << "[ERROR] Solver2::bump_clause: Invalid cid=" << cid << std::endl;
            return;
        }
        std::cerr << "[DEBUG] Solver2::bump_clause: Bumping activity for clause cid=" << cid << std::endl;
        var_inc /= decay; 
        for(int lit_val: cls[cid].lit) { // Rinominato lit a lit_val
            int var = std::abs(lit_val);
            if (var > 0 && var <= num_vars) {
                activity[var] += var_inc;
                 std::cerr << "[DEBUG] Solver2::bump_clause:   var=" << var << " activity now " << activity[var] << std::endl;
            }
        }
        if(var_inc > 1e100){ 
            std::cerr << "[DEBUG] Solver2::bump_clause: Rescaling activities and var_inc." << std::endl;
            for(int v=1; v <= num_vars; ++v) activity[v] *= 1e-100;
            var_inc *= 1e-100;
        }
    }

    int pick_var() { 
        double max_act = -1.0;
        int best_var = 0;
        // VSIDS-like heuristic: pick unassigned variable with highest activity
        for(int v=1; v<=num_vars; ++v) {
            if(val[v]==0) { 
                if (activity[v] > max_act) { 
                    max_act = activity[v];
                    best_var = v;
                }
            }
        }
        
        if (best_var != 0) { 
             std::cerr << "[DEBUG] Solver2::pick_var: Picked var=" << best_var << " (activity=" << max_act << ")" << std::endl;
             return best_var;
        }
        // Fallback se tutte le variabili non assegnate hanno attività 0 (o se non ci sono non assegnate)
        // o se best_var non è stato aggiornato (es. tutte attività negative, non dovrebbe succedere)
        for(int v=1; v<=num_vars; ++v) { // Fallback to first unassigned
            if(val[v]==0) {
                std::cerr << "[DEBUG] Solver2::pick_var (fallback): Picked first unassigned var=" << v << std::endl;
                return v;
            }
        }
        std::cerr << "[DEBUG] Solver2::pick_var: All variables assigned. Returning 0." << std::endl;
        return 0; 
    }

    void backtrack(size_t lvl)
    {
        if(lvl + 1 >= level_ofs.size()) return;      // niente da annullare

        size_t first_idx_next_lvl = level_ofs[lvl + 1];

        while(trail.size() > first_idx_next_lvl) {   // pop di tutti i lit dei
            int v = std::abs(trail.back());          // livelli alti
            val[v]    = 0;
            reason[v] = -1;
            trail.pop_back();
        }
        level_ofs.resize(lvl + 1);                   // mantieni 0..lvl
    }

    Result search(){
        std::cerr << "[DEBUG] Solver2::search BEGIN" << std::endl;
        std::cerr << "[DEBUG] Solver2::search: Initial propagate() call." << std::endl;
        if(propagate() != -1) { // Propaga assegnazioni iniziali (livello 0)
            std::cerr << "[DEBUG] Solver2::search: Initial propagate() found conflict. Returning UNSAT." << std::endl;
            return Result::UNSAT; 
        }
        std::cerr << "[DEBUG] Solver2::search: Initial propagate() successful." << std::endl;

        while(true){
            std::cerr << "[DEBUG] Solver2::search: ----- New decision cycle -----" << std::endl;
            int var_to_decide = pick_var();

            if(var_to_decide == 0) { 
                std::cerr << "[DEBUG] Solver2::search: pick_var() returned 0. All variables assigned. Model found. Returning SAT." << std::endl;
                return Result::SAT;             
            }
            std::cerr << "[DEBUG] Solver2::search: pick_var() chose var=" << var_to_decide << std::endl;

            decisions++;
            std::cerr << "[DEBUG] Solver2::search: Decision #" << decisions << " on var=" << var_to_decide << std::endl;
            
            // Nuovo livello di decisione
            level_ofs.push_back(trail.size()); 
            size_t current_decision_level_num = level_ofs.size() - 1; // Il numero del livello attuale (es. 1, 2, ...)
            std::cerr << "[DEBUG] Solver2::search: Starting new decision level " << current_decision_level_num
                      << ". Trail offset for this level: " << level_ofs.back() << std::endl;


            int decision_lit = (phase[var_to_decide] >= 0 ? var_to_decide : -var_to_decide);
            std::cerr << "[DEBUG] Solver2::search: Enqueuing decision lit=" << decision_lit << " (phase[" << var_to_decide << "]=" << (int)phase[var_to_decide] << ")" << std::endl;
            enqueue(decision_lit , -1 ); 

            while(true){ 
                std::cerr << "[DEBUG] Solver2::search: Calling propagate() after decision/flip." << std::endl;
                int conflict_cid = propagate();

                if(conflict_cid == -1) { 
                    std::cerr << "[DEBUG] Solver2::search: propagate() successful (no conflict). Breaking to make new decision." << std::endl;
                    break; 
                }

                conflicts++; 
                std::cerr << "[DEBUG] Solver2::search: CONFLICT found by propagate(). Conflicting clause cid=" << conflict_cid
                          << ". Total conflicts=" << conflicts << std::endl;
                if (conflict_cid >=0 && conflict_cid < cls.size()) { // Check validità cid
                    bump_clause(conflict_cid); 
                } else {
                    std::cerr << "[ERROR] Solver2::search: Invalid conflict_cid=" << conflict_cid << " from propagate." << std::endl;
                    // Questo è un errore grave, potrebbe indicare un problema in propagate()
                    // Forzare UNSAT per evitare loop infiniti o crash.
                    return Result::UNSAT;
                }


                while(true){ // Loop di backtrack + flip
                    if(level_ofs.size() == 1) { // Siamo al livello 0 (level_ofs contiene solo l'offset iniziale {0})
                        std::cerr << "[DEBUG] Solver2::search: Conflict at level 0 after backtracking. Formula UNSAT." << std::endl;
                        return Result::UNSAT; 
                    }

                    size_t last_decision_level_num = level_ofs.size() - 1; // Numero del livello di decisione più recente
                    std::cerr << "[DEBUG] Solver2::search: Backtracking from level " << last_decision_level_num << std::endl;

                    // Il letterale di decisione per questo livello è trail[level_ofs[last_decision_level_num]]
                    // Assicurati che level_ofs[last_decision_level_num] sia un indice valido per trail
                    if (level_ofs[last_decision_level_num] >= trail.size() && !trail.empty()) { // Aggiunto !trail.empty()
                         std::cerr << "[ERROR] Solver2::search: Invalid trail offset for decision lit. level_ofs["<<last_decision_level_num<<"]=" 
                                   << level_ofs[last_decision_level_num] << ", trail.size="<<trail.size() << std::endl;
                         return Result::UNSAT; // Errore grave
                    }
                     if (trail.empty() && level_ofs[last_decision_level_num] > 0) { // Trail vuoto ma offset non 0
                        std::cerr << "[ERROR] Solver2::search: Trail empty but decision offset > 0." << std::endl;
                        return Result::UNSAT;
                    }
                    // Se il trail è vuoto e l'offset è 0, non ci sono decisioni da flippare a questo livello.
                    // Questo non dovrebbe accadere se level_ofs.size() > 1.

                    int decision_lit_at_this_level = 0;
                    if (!trail.empty() && level_ofs[last_decision_level_num] < trail.size()) { // Aggiunto check trail non vuoto
                        decision_lit_at_this_level = trail[level_ofs[last_decision_level_num]];
                    } else if (trail.empty() && level_ofs[last_decision_level_num] == 0 && level_ofs.size() > 1) {
                        // Questo caso è strano: livello di decisione > 0, ma trail vuoto e offset 0.
                        // Implica che la decisione non è stata messa nel trail, o trail svuotato erroneamente.
                        std::cerr << "[ERROR] Solver2::search: Inconsistent state for decision literal retrieval at level " << last_decision_level_num << std::endl;
                        return Result::UNSAT;
                    } else if (level_ofs.size() == 1) { // Siamo al livello 0, già gestito sopra
                         return Result::UNSAT;
                    }


                    int var_of_decision = std::abs(decision_lit_at_this_level);
                     if (var_of_decision == 0 && level_ofs.size() > 1) { // Se decision_lit_at_this_level era 0 e non siamo a livello 0
                        std::cerr << "[ERROR] Solver2::search: Decision literal was 0 at level " << last_decision_level_num << std::endl;
                        return Result::UNSAT; // Errore grave
                    }


                    std::cerr << "[DEBUG] Solver2::search:   Decision lit at this level was " << decision_lit_at_this_level << std::endl;

                    // Fai backtrack al livello *precedente* a last_decision_level_num.
                    // Quindi il nuovo "current level" sarà last_decision_level_num - 1.
                    backtrack(last_decision_level_num - 1); 
                                                            
                    // Ora la variabile var_of_decision (quella della decisione flippata) dovrebbe essere non assegnata.
                    // E phase[var_of_decision] dovrebbe contenere la fase che ha causato il conflitto.
                    // Vogliamo provare la fase opposta.
                    if(var_of_decision > 0 && var_of_decision <= num_vars && val[var_of_decision] == 0){ 
                        int flipped_lit = -decision_lit_at_this_level; 
                        std::cerr << "[DEBUG] Solver2::search:   Variable var=" << var_of_decision << " is now unassigned. Enqueuing flipped lit=" << flipped_lit << std::endl;
                        
                        // Stiamo "modificando" la decisione al livello (ora) corrente, che è `level_ofs.size() - 1`.
                        // L'offset del trail per questo livello è `level_ofs.back()`.
                        // enqueue aggiornerà val, phase, reason, e trail.
                        enqueue(flipped_lit, -1); 
                                                  
                        break; 
                    } else {
                        // Se la variabile non è 0 dopo il backtrack, significa che era stata assegnata a un livello ancora precedente
                        // e il flip non è possibile a questo livello. Continua a fare backtrack.
                        std::cerr << "[DEBUG] Solver2::search:   Var=" << var_of_decision << " (from lit " << decision_lit_at_this_level
                                  << ") is STILL ASSIGNED to " << (var_of_decision > 0 && var_of_decision <=num_vars ? (int)val[var_of_decision] : -99) 
                                  << " or invalid after backtrack."
                                  << " Continuing backtrack." << std::endl;
                    }
                } 
                 std::cerr << "[DEBUG] Solver2::search: Finished a backtrack/flip sequence. Continuing to propagate." << std::endl;
            } 
        } 
    }
};

#endif // SOLVER2_WATCHED_DEBUG_HPP
