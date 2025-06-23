#include "solver.hpp"
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#ifdef SAT_DEBUG
  #define DBG(x) do { x; } while(0)
#else
  #define DBG(x) do { } while(0)
#endif

using Clock = std::chrono::steady_clock;

template<class T>
static void dbg_vec(const std::vector<T>& v,const char* n){
    DBG({
        std::cerr<<n<<": { ";
        for(std::size_t i=0;i<v.size();++i)
            std::cerr<<v[i]<<(i+1==v.size()?"":" ,");
        std::cerr<<" }\n";
    });
}

// Conflict-Drien Clause Learning

class Solver3 : public Solver{
    using i8 = int8_t;
    struct WClause{ std::vector<int> lit; int w1; int w2; };

    int nvars = 0;
    std::vector<WClause> cls;            
    std::vector<std::vector<int>> watch;       
    std::vector<i8> val;            
    std::vector<int> reason;         
    std::vector<int> level_of;       
    std::vector<int> trail;         
    std::vector<int> level_beg;      

    std::vector<double> activity; double var_inc=1.0, var_decay=0.95;

    uint64_t propagations=0, decisions=0, conflicts=0;
    uint64_t next_restart=100;  double   restart_mult=1.5;

    static int lit2idx(int lit,int n){ return lit>0 ? lit : n - lit; } 
    inline bool valid_var(int v) const { return v>0 && v<=nvars; }
    inline int  cur_level()      const { return int(level_beg.size())-1; }

public:
    std::string name() const override { return "CDCL + learning"; }

    Result solve(const std::vector<Clause>& in,int n,Stats& st) override{
        reset_state();
        if(!build(in,n)) { st.millis=0; return Result::UNSAT; }

        auto start = Clock::now();
        Result res = cdcl();

        st.millis = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        st.decisions    = decisions;
        st.propagations = propagations;
        st.conflicts    = conflicts;
        return res;
    }

private:
    void reset_state(){
        cls.clear(); watch.clear(); val.clear(); reason.clear();
        level_of.clear(); trail.clear(); level_beg.clear(); activity.clear();
        propagations=decisions=conflicts=0; next_restart=100;
    }

    bool build(const std::vector<Clause>& in,int n){
        nvars=n;
        if(n==0 && !in.empty()) return false; // check: if variables are zero, but clause is not empty --> false

        val.assign(n+1,0); // everything assigned to 0 
        reason.assign(n+1,-1); // reason of assignment -1 (no responsible clause)
        level_of.assign(n+1,0); //level of decision 0 
        activity.assign(n+1,0.0);  // activity to zero
        watch.assign(n==0?1:2*n+1,{}); //watched literals: if n==0 then 1 bucket, otherwise 2n+1
        level_beg={0};

        cls.reserve(in.size());
        for(const Clause& c:in){ //loop overall the clause
            if(c.lit.empty()) return false; // empty clause is unset --> by definition
            for(int l:c.lit) if(l==0 || std::abs(l)>n) return false; // check that all literals are not zero 

            int id = int(cls.size()); // save the index of the new clause 
            int w1=0, w2=c.lit.size()>1?1:0; // if the clause has 2 literals, then 0-1, otherwise watch two times 0-0
            cls.push_back({c.lit,w1,w2}); // add the clause to the internal database of solver 
            add_watch(c.lit[w1],id); // add the clause id to the list of watched literal
            if(c.lit.size()>1) add_watch(c.lit[w2],id);
        }
        return propagate()==-1; 
    }

    void add_watch(int lit,int cid){
        if(lit==0){ DBG(std::cerr<<"[WARN] add_watch: lit 0, skip (cid="<<cid<<")\n"); return; } // warning, literal equal to zero
        int v = std::abs(lit); // extract variable associated to literal lit
        if(!valid_var(v)){ // check if it is in a valid range
            DBG(std::cerr<<"[WARN] add_watch: lit fuori range ("<<lit<<"), skip.\n");
            return; 
        }
        watch[lit2idx(lit,nvars)].push_back(cid);
    }

    int propagate(){ // BCP - Boolean Constraint Propagation
        std::size_t head=0;
        while(head<trail.size()){ // continue until i find new literal into trail to propagate
            const int falselit = -trail[head++]; // take the literal, and falsity it
            auto &wl = watch[lit2idx(falselit,nvars)]; // take the list of clause that watch that literal now false
            for(std::size_t i=0;i<wl.size();){ // loop over clause false
                const int cid = wl[i];
                WClause &c = cls[cid]; // taking ID and clause 

                if(c.lit.size()==1){ //special case, only one literal
                    int lit = c.lit[0];
                    int s   = val[std::abs(lit)];
                    if(s==0){ assign(lit,cid); ++i; continue; } //if literal is not assigned, then we can assign it
                    if(s==(lit>0?1:-1)){ ++i; continue; } // if satisfied, next clause
                    wl[i]=wl.back(); wl.pop_back(); return cid; // otherwise is false, conflitc
                }

                int *myw = (c.lit[c.w1]==falselit? &c.w1 : (c.lit[c.w2]==falselit? &c.w2 : nullptr)); // determine which one w1/w2 caused the problem (= to falselit)
                if(!myw){ ++i; continue; } //if none, skip clause
                int other = c.lit[(myw==&c.w1)? c.w2 : c.w1];
                int other_val = val[std::abs(other)]; // read the other literal (not falselit) and read the value
                if(other_val==(other>0?1:-1)){ ++i; continue; } // if other literal is already true, clause is satisfied

                bool moved=false;
                for(int k=0;k<(int)c.lit.size();++k){ if(k==c.w1||k==c.w2) continue; int L=c.lit[k]; int sv=val[std::abs(L)]; // loop over the literal into clause that is not satisfied or assigned
                    if(sv==0 || sv==(L>0?1:-1)){ 
                        *myw=k; add_watch(L,cid); // promoted to literal and updating the structure
                        wl[i]=wl.back(); wl.pop_back(); moved=true; break; // removing clase from list wl
                    }
                }
                if(moved) continue; 

                if(other_val==0){ assign(other,cid); ++i; }
                else { wl[i]=wl.back(); wl.pop_back(); return cid; }
            }
        }
        return -1; 
    }

    void assign(int lit,int cid){
        int v = std::abs(lit); // extracting variable v associated to lit
        if(!valid_var(v)){
            DBG(std::cerr<<"[WARN] assign: variabile fuori range ("<<v<<") – ignorata.\n");
            return;
        }
        const int newval = (lit>0?1:-1); // assign val: if lit > 0, then newval = 1, otherwise newval = -1
        if(val[v]!=0){ // variable already assigned
            if(val[v]==newval) return; 
            throw std::runtime_error("Assegnazione conflittuale su variabile già definita");
        }
        val[v]=newval;
        reason[v]=cid; // saving the reason for this assignment (need later in conflict analysis)
        level_of[v]=cur_level(); // register the decision level where variable v is assigned
        trail.push_back(lit); // adding literal to trail
        if(cid!=-1) ++propagations;
    }

    void bump_var(int v){
        activity[v]+=var_inc;
        if(activity[v]>1e100){ for(int i=1;i<=nvars;++i) activity[i]*=1e-100; var_inc*=1e-100; }
    }
    void bump_clause(const std::vector<int>& c){ for(int l:c) bump_var(std::abs(l)); var_inc/=var_decay; }
    int  pick_branch() const{
        int best=0; double best_a=-1.0;
        for(int v=1;v<=nvars;++v) if(val[v]==0 && activity[v]>=best_a){ best_a=activity[v]; best=v; }
        return best;
    }

    std::vector<int> analyze(int cid,int &out_bt){
        std::vector<int> learnt; learnt.reserve(8);
        std::vector<char> seen(nvars+1,0);
        int to_resolve=0;

        for(int l:cls[cid].lit){ int v=std::abs(l); if(!valid_var(v)) continue; if(seen[v]) continue; seen[v]=1; bump_var(v);
            if(level_of[v]==cur_level()) ++to_resolve; else learnt.push_back(l); }

        int idx=int(trail.size())-1; int uip=0;
        while(to_resolve>0 && idx>=0){
            int lit = trail[idx--]; int v=std::abs(lit); if(!seen[v]) continue; seen[v]=0;
            int rcid = reason[v];
            if(rcid!=-1){
                for(int q:cls[rcid].lit){ int u=std::abs(q); if(!valid_var(u)) continue; if(seen[u]) continue; seen[u]=1; bump_var(u);
                    if(level_of[u]==cur_level()) ++to_resolve; else learnt.push_back(q);
                }
            }
            --to_resolve; uip=-lit;
        }
        learnt.push_back(uip);
        bump_clause(learnt);
        int bt=0; for(std::size_t i=0;i+1<learnt.size();++i){ int v=std::abs(learnt[i]); if(valid_var(v)) bt=std::max(bt,level_of[v]); }
        out_bt=bt; return learnt;
    }

    void backtrack(int lvl){ assert(lvl<=cur_level());
        for(int i=int(trail.size())-1;i>=level_beg[lvl];--i){ int v=std::abs(trail[i]); if(valid_var(v)){ val[v]=0; reason[v]=-1; level_of[v]=0; } }
        trail.resize(level_beg[lvl]);
        level_beg.resize(lvl+1);
    }

    void add_clause(const std::vector<int>& c){ assert(!c.empty());
        cls.push_back({c,0,int(c.size())>1?1:0}); int id=int(cls.size())-1;
        add_watch(c[0],id); if(c.size()>1) add_watch(c[1],id);
    }

    Result cdcl(){
        try{
            while(true){
                if(conflicts>=next_restart){
                    backtrack(0);
                    next_restart = uint64_t(next_restart*restart_mult);
                    if(propagate()!=-1) return Result::UNSAT;
                }

                int v = pick_branch();
                if(v==0) return Result::SAT; 

                ++decisions;
                level_beg.push_back(int(trail.size()));
                assign(v,-1);

                while(true){
                    int confl = propagate();
                    if(confl==-1) break; 

                    ++conflicts;
                    if(cur_level()==0) return Result::UNSAT;

                    int bt; auto learnt = analyze(confl,bt);
                    add_clause(learnt);
                    backtrack(bt);
                    assign(learnt.back(),int(cls.size())-1);
                }
            }
        }catch(const std::runtime_error&){
            return Result::UNSAT;
        }
    }
};
