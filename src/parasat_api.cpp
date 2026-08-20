#define PARASAT_BUILD_DLL
#include "../include/parasat.h"
#include "cnf_formula.hpp"
#include "gpu_portfolio_solver.hpp"
#include <string>
#include <vector>
#include <cstring>
#include <iostream>

// Internal state wrapper
struct ParaSATState {
    CNFFormula cnf;
    std::vector<int> current_clause;
    
    double timeout_sec = 10.0;
    std::string mode = "cpu+gpu";
    int cpu_threads = 0;
    int gpu_cores = 0;
    bool verbose = false;
    bool enable_hyperpath = true;
    bool enable_preflight = true;
    
    int result_status = PARASAT_UNKNOWN;
    double elapsed_ms = 0.0;
    std::vector<int> model;
    std::string winning_engine = "NONE";
    char engine_name_cstr[256] = {0};
};

extern "C" {

parasat_env parasat_init() {
    return new ParaSATState();
}

void parasat_destroy(parasat_env env) {
    if (env) {
        delete static_cast<ParaSATState*>(env);
    }
}

void parasat_set_timeout(parasat_env env, double timeout_sec) {
    if (env) static_cast<ParaSATState*>(env)->timeout_sec = timeout_sec;
}

void parasat_set_mode(parasat_env env, const char* mode) {
    if (env && mode) static_cast<ParaSATState*>(env)->mode = mode;
}

void parasat_set_threads(parasat_env env, int cpu_threads, int gpu_cores) {
    if (env) {
        static_cast<ParaSATState*>(env)->cpu_threads = cpu_threads;
        static_cast<ParaSATState*>(env)->gpu_cores = gpu_cores;
    }
}

void parasat_set_verbose(parasat_env env, int verbose) {
    if (env) static_cast<ParaSATState*>(env)->verbose = (verbose != 0);
}

void parasat_set_hyperpath(parasat_env env, int enable_hyper) {
    if (env) {
        static_cast<ParaSATState*>(env)->enable_hyperpath = (enable_hyper != 0);
        static_cast<ParaSATState*>(env)->enable_preflight = (enable_hyper != 0);
    }
}

void parasat_set_num_vars(parasat_env env, int num_vars) {
    if (env) {
        ParaSATState* state = static_cast<ParaSATState*>(env);
        state->cnf.num_vars = num_vars;
        state->cnf.pos_occur.resize(num_vars + 1);
        state->cnf.neg_occur.resize(num_vars + 1);
    }
}

void parasat_add_literal(parasat_env env, int lit) {
    if (env) {
        static_cast<ParaSATState*>(env)->current_clause.push_back(lit);
    }
}

void parasat_commit_clause(parasat_env env) {
    if (!env) return;
    ParaSATState* state = static_cast<ParaSATState*>(env);
    
    if (state->current_clause.empty()) {
        state->cnf.has_empty_clause = true;
    } else {
        // Expand num_vars dynamically if needed
        for (int lit : state->current_clause) {
            int v = std::abs(lit);
            if (v > state->cnf.num_vars) {
                state->cnf.num_vars = v;
                state->cnf.pos_occur.resize(v + 1);
                state->cnf.neg_occur.resize(v + 1);
            }
        }
        
        int c_idx = static_cast<int>(state->cnf.clauses.size());
        state->cnf.clauses.push_back(state->current_clause);
        
        if (state->current_clause.size() == 2) {
            state->cnf.num_binary_clauses++;
        }
        
        for (int lit : state->current_clause) {
            int var = std::abs(lit);
            if (lit > 0) state->cnf.pos_occur[var].push_back(c_idx);
            else state->cnf.neg_occur[var].push_back(c_idx);
        }
    }
    state->cnf.num_clauses++;
    state->current_clause.clear();
}

int parasat_solve(parasat_env env) {
    if (!env) return PARASAT_UNKNOWN;
    ParaSATState* state = static_cast<ParaSATState*>(env);
    
    ParaSATGPUPortfolio solver(
        state->cnf, 
        state->timeout_sec, 
        state->mode, 
        state->cpu_threads, 
        state->gpu_cores, 
        state->enable_preflight, 
        state->enable_hyperpath, 
        state->verbose
    );
    
    GPUPortfolioResult res = solver.solve();
    
    state->elapsed_ms = res.elapsed_ms;
    state->winning_engine = res.winning_engine;
    state->model = res.model;
    
    std::strncpy(state->engine_name_cstr, res.winning_engine.c_str(), 255);
    
    if (res.is_sat) {
        state->result_status = PARASAT_SAT;
        return PARASAT_SAT;
    } else if (res.is_unsat) {
        state->result_status = PARASAT_UNSAT;
        return PARASAT_UNSAT;
    }
    
    state->result_status = PARASAT_UNKNOWN;
    return PARASAT_UNKNOWN;
}

int parasat_get_model_val(parasat_env env, int var) {
    if (!env) return 0;
    ParaSATState* state = static_cast<ParaSATState*>(env);
    
    if (state->result_status != PARASAT_SAT) return 0;
    if (var <= 0 || var > state->cnf.num_vars) return 0;
    if (var >= static_cast<int>(state->model.size())) return 0;
    
    int val = state->model[var];
    if (val > 0) return 1;
    if (val < 0) return -1;
    return 0;
}

double parasat_get_solve_time_ms(parasat_env env) {
    if (!env) return 0.0;
    return static_cast<ParaSATState*>(env)->elapsed_ms;
}

const char* parasat_get_winning_engine(parasat_env env) {
    if (!env) return "";
    return static_cast<ParaSATState*>(env)->engine_name_cstr;
}

} // extern "C"
