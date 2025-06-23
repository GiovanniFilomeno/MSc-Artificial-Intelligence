#ifndef SOLVER2_WATCHED_DEBUG_HPP
#define SOLVER2_WATCHED_DEBUG_HPP

#include "solver.hpp" 
#include <vector>
#include <chrono>
#include <iostream> 
#include <iomanip> 

class Solver2 : public Solver {
public:
    std::string name() const override { return "Watched‑literal DPLL (Debug)"; }

    static inline auto   now() { return std::chrono::steady_clock::now(); }
    static inline double elapsed(const std::chrono::steady_clock::time_point &s){
        using namespace std::chrono;
        return duration<double,std::milli>(steady_clock::now() - s).count(); }
    
    static inline int lit2idx(int lit, int n_vars) {
        if (lit > 0) return lit;
        return n_vars - lit; 
    }


    Result solve(const std::vector<Clause>& in,int n,Stats& st) override {
        if(n == 0 && in.empty()){ 
            st.decisions     = 0;
            st.propagations  = 0;
            st.conflicts     = 0;
            st.millis        = 0.0;
            return Result::SAT;
        }

        decisions = propagations = conflicts = 0; 

        if(!build(in,n)){                 
            st.decisions     = 0;
            st.propagations  = 0;
            st.conflicts     = 0;
            st.millis        = 0.0;
            return Result::UNSAT;
        }

        auto start_time = now();
        Result res = search();
        st.millis = elapsed(start_time);
        st.decisions = decisions;
        st.propagations = propagations;

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
        num_vars = n;

        for(size_t i = 0; i < in.size(); ++i) { // no input clause must be empty
            if(in[i].lit.empty()) {
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

        for(const Clause& c_in : in){
            if (c_in.lit.empty()) { // if clause is empty --> UNSAT
                return false;
            }
            
            std::vector<int> processed_lits;
            bool tautology = false;
            for (int lit_val : c_in.lit) { 
                if (tautology) break;
                int var = std::abs(lit_val);
                if (var == 0 || var > num_vars) { //skip literals out of range
                    continue; 
                }
     
                bool already_present = false;
                for(int plit : processed_lits) { // avoid duplicates
                    if(plit == lit_val) already_present = true;
                    if(plit == -lit_val) tautology = true; 
                }
                if (tautology) break;

                if(!already_present) { //add literal if valid and not duplicated 
                    processed_lits.push_back(lit_val);
                }
            }

            if (tautology) {
                continue; 
            }
            
            if (processed_lits.empty() && !c_in.lit.empty()) {
                for(int l_val : c_in.lit) std::cerr << l_val << " ";
                 return false; 
            }
             if (processed_lits.empty() && c_in.lit.empty()){
             } else if (processed_lits.empty()){ 
             }


            int w1_idx = 0;
            int w2_idx = (processed_lits.size() > 1 ? 1 : 0); // imposta due watched literals
            
            if (processed_lits.empty()) {
                continue;
            }

            cls.push_back({processed_lits, w1_idx, w2_idx}); // add internal clause to the database cls, giving an ID
            int cid = (int)cls.size() - 1; 

            add_watch(cls[cid].lit[w1_idx], cid);
            if (processed_lits.size() > 1) {
                add_watch(cls[cid].lit[w2_idx], cid);
            } // add the watched literals in watch


            for(int lit_val : cls[cid].lit) {
                if (std::abs(lit_val) <= num_vars && std::abs(lit_val) > 0) { 
                    activity[std::abs(lit_val)] += 1.0;
                }
            } // update activity
        }

        for(int cid = 0; cid < (int)cls.size(); ++cid) { // search for unit clause
            if(cls[cid].lit.size() == 1) { 
                int unit_lit = cls[cid].lit[0];
                
                int var_unit = std::abs(unit_lit);
                if (var_unit == 0 || var_unit > num_vars) {
                     continue;
                } 

                if (val[var_unit] == (unit_lit > 0 ? -1 : 1)) {
                    conflicts++; 
                    return false; 
                } // if contraddiction found --> conflict
                if (val[var_unit] == 0) { 
                    if (!enqueue(unit_lit, cid)) { 
                        conflicts++; 
                        return false; 
                    } // if variable not assigned --> enqueue
                } else {
                    //  std::cerr << "[DEBUG] Solver2::build: Unit literal " << unit_lit << " already consistently assigned. Skipping enqueue." << std::endl;
                }
            }
        }

        int initial_conflict_cid = propagate(); //initial propagation
        if (initial_conflict_cid != -1) {
            conflicts++;
            return false; 
        }
        return true;
    }

    void add_watch(int lit_val, int cid){ 
        if (lit_val == 0) {
            return;
        }
        int var = std::abs(lit_val);
        if (var == 0 || var > num_vars) { 
            return;
        }
        int idx = lit2idx(lit_val, num_vars);
        if (idx < 0 || idx >= watch.size()) { 
            return;
        }
        watch[idx].push_back(cid);
    }

    bool enqueue(int lit_val, int why_cid){ 
        int v = std::abs(lit_val);
        int8_t s = (lit_val > 0 ? 1 : -1);

        if (v == 0 || v > num_vars) {
            return false; 
        }


        if(val[v] == 0){ // variable not assigned
            val[v] = s;
            phase[v] = s; 
            reason[v] = why_cid; //save from which clause the assignation comes
            trail.push_back(lit_val); // add lit_val to trail
            propagations++; // propagation counter
            return true;
        } else if (val[v] == s) { 
            return true; 
        } else { 
            return false; 
        }
    }

    int propagate(){
        if (!level_ofs.empty()) {
            //  std::cerr << "[DEBUG] Solver2::propagate: Current decision level: " << level_ofs.size() -1 << ", propagation starts from trail_offset: " << level_ofs.back() << std::endl;
        }
        
        size_t current_trail_pointer = 0; 
        if (!level_ofs.empty()) {
            current_trail_pointer = level_ofs.back(); 
        } // start propagation from last decision point (trail offset)


        while(current_trail_pointer < trail.size()){
            int assigned_true_lit = trail[current_trail_pointer];
            int lit_false = -assigned_true_lit; 
            current_trail_pointer++; // for each new assignation, consider the opposite literal now as false
 
            int var_abs_lit_false = std::abs(lit_false);
            if (var_abs_lit_false == 0 || var_abs_lit_false > num_vars) {
                // std::cerr << "[ERROR] Solver2::propagate: Invalid var_abs_lit_false=" << var_abs_lit_false << " from lit_false=" << lit_false << std::endl;
                continue; // skip if literal out of range
            }

            int watch_idx = lit2idx(lit_false, num_vars); // taking clause that are watching the literal now false
            if (watch_idx < 0 || watch_idx >= watch.size()) { 
                //  std::cerr << "[ERROR] Solver2::propagate: Invalid watch_idx=" << watch_idx << " for lit_false=" << lit_false << std::endl;
                 continue; 
            }
            std::vector<int>& wl = watch[watch_idx]; 

            size_t i = 0;
            while (i < wl.size()) { // loop over clause cid
                int cid = wl[i];
                if (cid < 0 || cid >= cls.size()){ // not valid clause= then remove
                    wl[i] = wl.back(); 
                    wl.pop_back();
                    continue;
                }
                WClause &c = cls[cid]; 

                if (c.w1 < 0 || c.w1 >= c.lit.size() || c.w2 < 0 || c.w2 >= c.lit.size()) {
                    i++; 
                    continue; // skip if watched index out of range
                }

                int current_false_watched_lit_val_in_clause; 
                int other_watched_lit_val_in_clause; 

                if (c.lit[c.w1] == lit_false) { // find which of the two watched literals is not false
                    current_false_watched_lit_val_in_clause = c.lit[c.w1]; 
                    other_watched_lit_val_in_clause = c.lit[c.w2];
                } else if (c.lit[c.w2] == lit_false) {
                    current_false_watched_lit_val_in_clause = c.lit[c.w2]; 
                    other_watched_lit_val_in_clause = c.lit[c.w1];
                } else {
                    i++; 
                    continue;
                }

                int other_var = std::abs(other_watched_lit_val_in_clause);
                if (other_var > 0 && other_var <= num_vars && // Case1: clause is still sat?
                    val[other_var] == (other_watched_lit_val_in_clause > 0 ? 1 : -1)) {
                    i++; 
                    continue;
                }

                bool moved_watch = false;
                for(size_t k=0; k < c.lit.size(); ++k){ // search a new literal not assigned o already satisfied to be used as new watched
                    
                    if (k == (size_t)c.w1 || k == (size_t)c.w2) continue; 


                    int L_new = c.lit[k];
                    int L_new_var = std::abs(L_new);

                    if (L_new_var > 0 && L_new_var <= num_vars && 
                        (val[L_new_var] == 0 || val[L_new_var] == (L_new > 0 ? 1 : -1))) {

                        if (c.lit[c.w1] == current_false_watched_lit_val_in_clause) { // if new found, update watched of clause
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
                    continue; 
                }

                int other_watched_var = std::abs(other_watched_lit_val_in_clause);
                if (other_watched_var == 0 || other_watched_var > num_vars) { // if I cannot move and the other watched literal is not assignet --> unit clause
                     i++; continue; 
                }

                if(val[other_watched_var] == 0){
                    if(!enqueue(other_watched_lit_val_in_clause, cid)) { 
                        conflicts++;
                        return cid; 
                    }
                    i++; 
                }
                else { 
                    conflicts++; // if both watched are false
                    return cid; 
                }
            } 
        } 

        return -1; 
    }

    void bump_clause(int cid){ 
        if (cid < 0 || cid >= cls.size()) {
            return;
        }
        var_inc /= decay; 
        for(int lit_val: cls[cid].lit) { 
            int var = std::abs(lit_val);
            if (var > 0 && var <= num_vars) {
                activity[var] += var_inc;
            }
        }
        if(var_inc > 1e100){ 
            for(int v=1; v <= num_vars; ++v) activity[v] *= 1e-100;
            var_inc *= 1e-100;
        }
    }

    int pick_var() { 
        double max_act = -1.0;
        int best_var = 0;
        for(int v=1; v<=num_vars; ++v) {
            if(val[v]==0) { 
                if (activity[v] > max_act) { 
                    max_act = activity[v];
                    best_var = v;
                }
            }
        }
        
        if (best_var != 0) { 
             return best_var;
        }
        for(int v=1; v<=num_vars; ++v) { 
            if(val[v]==0) {
                return v;
            }
        }
        return 0; 
    }

    void backtrack(size_t to_level){ 

        if (to_level >= level_ofs.size() ) { 
            return; 
        }
        
        size_t start_undo_trail_offset = level_ofs[to_level + 1]; 

        for(size_t i = trail.size(); i-- > start_undo_trail_offset; ){ 
            int lit_to_undo = trail[i];
            int v = std::abs(lit_to_undo);
            if (v > 0 && v <= num_vars) { 
                val[v] = 0;    
                reason[v] = -1; 
            }
        }
        trail.resize(start_undo_trail_offset); 
        level_ofs.resize(to_level + 1); 
        
        if (!level_ofs.empty()) {
            // std::cerr << "[DEBUG] Solver2::backtrack:   level_ofs.back() (new current trail offset for propagations) = " << level_ofs.back() << std::endl;
        }
    }

    Result search(){

        if(propagate() != -1) { 
            return Result::UNSAT; 
        }

        while(true){
            int var_to_decide = pick_var();

            if(var_to_decide == 0) { 
                return Result::SAT;             
            }

            decisions++;
            
            level_ofs.push_back(trail.size()); 
            size_t current_decision_level_num = level_ofs.size() - 1; 

            int decision_lit = (phase[var_to_decide] >= 0 ? var_to_decide : -var_to_decide);
            enqueue(decision_lit , -1 ); 

            while(true){ 
                int conflict_cid = propagate();

                if(conflict_cid == -1) { 
                    // std::cerr << "[DEBUG] Solver2::search: propagate() successful (no conflict). Breaking to make new decision." << std::endl;
                    break; 
                }

                conflicts++; 

                if (conflict_cid >=0 && conflict_cid < cls.size()) { 
                    bump_clause(conflict_cid); 
                } else {
                    return Result::UNSAT;
                }


                while(true){ 
                    if(level_ofs.size() == 1) { 
                        return Result::UNSAT; 
                    }

                    size_t last_decision_level_num = level_ofs.size() - 1; 

                    if (level_ofs[last_decision_level_num] >= trail.size() && !trail.empty()) { 
                         return Result::UNSAT; 
                    }
                     if (trail.empty() && level_ofs[last_decision_level_num] > 0) { 
                        // std::cerr << "[ERROR] Solver2::search: Trail empty but decision offset > 0." << std::endl;
                        return Result::UNSAT;
                    }

                    int decision_lit_at_this_level = 0;
                    if (!trail.empty() && level_ofs[last_decision_level_num] < trail.size()) {
                        decision_lit_at_this_level = trail[level_ofs[last_decision_level_num]];
                    } else if (trail.empty() && level_ofs[last_decision_level_num] == 0 && level_ofs.size() > 1) {
                        return Result::UNSAT;
                    } else if (level_ofs.size() == 1) { 
                         return Result::UNSAT;
                    }


                    int var_of_decision = std::abs(decision_lit_at_this_level);
                     if (var_of_decision == 0 && level_ofs.size() > 1) {
                        return Result::UNSAT; 
                    }

                    backtrack(last_decision_level_num - 1); 
                                                            
                    if(var_of_decision > 0 && var_of_decision <= num_vars && val[var_of_decision] == 0){ 
                        int flipped_lit = -decision_lit_at_this_level; 
                        enqueue(flipped_lit, -1); 
                                                  
                        break; 
                    } else {
                    }
                } 
            } 
        } 
    }
};

#endif
