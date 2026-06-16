#include <filesystem>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include "parser.hpp"
#include "solver1_recursive.cpp"
#include "solver2_watched.cpp"
#include "solver3_cdcl.cpp"
#include "solver4_picosat.cpp"

using Clock   = std::chrono::steady_clock;
using ms      = std::chrono::milliseconds;

struct AsyncRes {
    std::atomic<bool> done{false};
    Result            res = Result::UNKNOWN;
    Stats             st;
};

int main(int argc,char** argv){
    if(argc!=2){ std::cerr<<"give path to folder\n"; return 1; }

    std::vector<std::unique_ptr<Solver>> solvers;
    solvers.emplace_back(std::make_unique<Solver1>());
    solvers.emplace_back(std::make_unique<Solver2>());
    solvers.emplace_back(std::make_unique<Solver3>());
    solvers.emplace_back(std::make_unique<SolverPico>()); 

    constexpr auto LIMIT = std::chrono::minutes(5);

    for(const auto& entry : std::filesystem::directory_iterator(argv[1])){
        if(entry.path().extension() != ".in") continue;

        std::vector<Clause> cls; int vars=0;
        if(!read_cnf(entry.path().string(), cls, vars)){
            std::cerr<<"Cannot read "<<entry.path()<<"\n"; continue;
        }
        std::cout<<"\n=== "<<entry.path().filename()<<" ("<<vars<<" vars, "
                 <<cls.size()<<" clauses) ===\n";

        for(auto& s : solvers){
            AsyncRes box;                   
            std::thread t([&cls,&box,&s,vars]{
                Stats st;                   
                box.res = s->solve(cls,vars,st);
                box.st  = st;
                box.done.store(true,std::memory_order_release);
            });

            auto start = Clock::now();
            while(!box.done.load(std::memory_order_acquire) &&
                  Clock::now()-start < LIMIT)
                std::this_thread::sleep_for(ms(100));

            if(box.done){
                std::cout<<s->name()<<": "
                         <<(box.res==Result::SAT?"SAT":box.res==Result::UNSAT?
                            "UNSAT":"??")
                         <<"  decisions="<<box.st.decisions
                         <<"  prop="     <<box.st.propagations
                         <<"  time="     <<box.st.millis<<" ms\n";
                t.join();               
            }else{
                std::cout<<s->name()<<": TIMEOUT (>5 min)\n";
                t.detach();         
            }
        }
    }
}
