#ifndef PARASAT_API_H
#define PARASAT_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #ifdef PARASAT_BUILD_DLL
        #define PARASAT_API __declspec(dllexport)
    #else
        #define PARASAT_API __declspec(dllimport)
    #endif
#else
    #define PARASAT_API __attribute__((visibility("default")))
#endif

// Status codes
#define PARASAT_UNKNOWN 0
#define PARASAT_SAT 10
#define PARASAT_UNSAT 20

typedef void* parasat_env;

// --- Initialization and Teardown ---
PARASAT_API parasat_env parasat_init();
PARASAT_API void parasat_destroy(parasat_env env);

// --- Configuration ---
PARASAT_API void parasat_set_timeout(parasat_env env, double timeout_sec);
PARASAT_API void parasat_set_mode(parasat_env env, const char* mode); // "cpu", "gpu", "cpu+gpu"
PARASAT_API void parasat_set_threads(parasat_env env, int cpu_threads, int gpu_cores);
PARASAT_API void parasat_set_verbose(parasat_env env, int verbose);
PARASAT_API void parasat_set_hyperpath(parasat_env env, int enable_hyper);

// --- Formula Construction ---
PARASAT_API void parasat_set_num_vars(parasat_env env, int num_vars);
PARASAT_API void parasat_add_literal(parasat_env env, int lit);
PARASAT_API void parasat_commit_clause(parasat_env env);

// --- Solving and Model Extraction ---
PARASAT_API int parasat_solve(parasat_env env);
PARASAT_API int parasat_get_model_val(parasat_env env, int var); // Returns 1 (True), -1 (False), or 0 (Unassigned)
PARASAT_API double parasat_get_solve_time_ms(parasat_env env);
PARASAT_API const char* parasat_get_winning_engine(parasat_env env);

#ifdef __cplusplus
}
#endif

#endif // PARASAT_API_H
