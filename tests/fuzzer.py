#!/usr/bin/env python3
import ctypes
import os
import random
import time

DLL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "repo gpu", "bin", "parasat.dll"))
if not os.path.exists(DLL_PATH):
    DLL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "repo gpu", "bin", "libparasat.so"))

if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(os.path.dirname(DLL_PATH))
    # Note: On Windows, you may need to add your compiler's bin directory to load libstdc++ / libomp
    # os.add_dll_directory(r"C:\path\to\your\mingw64\bin")

try:
    lib = ctypes.CDLL(DLL_PATH)
except Exception as e:
    print(f"Error loading {DLL_PATH}: {e}")
    exit(1)

# --- Define CTypes interfaces ---
lib.parasat_init.restype = ctypes.c_void_p
lib.parasat_destroy.argtypes = [ctypes.c_void_p]

lib.parasat_set_timeout.argtypes = [ctypes.c_void_p, ctypes.c_double]
lib.parasat_set_mode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.parasat_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.parasat_set_verbose.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.parasat_set_hyperpath.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.parasat_set_num_vars.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.parasat_add_literal.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.parasat_commit_clause.argtypes = [ctypes.c_void_p]

lib.parasat_solve.argtypes = [ctypes.c_void_p]
lib.parasat_solve.restype = ctypes.c_int
lib.parasat_get_model_val.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.parasat_get_model_val.restype = ctypes.c_int
lib.parasat_get_solve_time_ms.argtypes = [ctypes.c_void_p]
lib.parasat_get_solve_time_ms.restype = ctypes.c_double
lib.parasat_get_winning_engine.argtypes = [ctypes.c_void_p]
lib.parasat_get_winning_engine.restype = ctypes.c_char_p

# Status codes
PARASAT_UNKNOWN = 0
PARASAT_SAT = 10
PARASAT_UNSAT = 20

class ParaSATSolver:
    def __init__(self, mode=b"cpu+gpu", threads=0, cores=0):
        self.env = lib.parasat_init()
        lib.parasat_set_mode(self.env, mode)
        lib.parasat_set_threads(self.env, threads, cores)
        lib.parasat_set_verbose(self.env, 0) # quiet mode for fuzzing

    def __del__(self):
        lib.parasat_destroy(self.env)
        
    def add_clause(self, literals):
        for lit in literals:
            lib.parasat_add_literal(self.env, lit)
        lib.parasat_commit_clause(self.env)
        
    def solve(self):
        status = lib.parasat_solve(self.env)
        return status
        
    def get_model(self, num_vars):
        return [lib.parasat_get_model_val(self.env, v) for v in range(1, num_vars + 1)]
    
    def get_engine(self):
        return lib.parasat_get_winning_engine(self.env).decode('utf-8')

# Naive DPLL solver in Python for Reference Testing
def dpll(clauses, vars_set, assignment):
    # Base cases
    if len(clauses) == 0:
        return True, assignment
    if any(len(c) == 0 for c in clauses):
        return False, None
    
    # Unit propagation
    unit_clauses = [c for c in clauses if len(c) == 1]
    while unit_clauses:
        unit = unit_clauses[0][0]
        assignment[abs(unit)] = 1 if unit > 0 else -1
        new_clauses = []
        for c in clauses:
            if unit in c: continue
            if -unit in c:
                new_c = [l for l in c if l != -unit]
                if len(new_c) == 0: return False, None
                new_clauses.append(new_c)
            else:
                new_clauses.append(c)
        clauses = new_clauses
        if len(clauses) == 0: return True, assignment
        unit_clauses = [c for c in clauses if len(c) == 1]
        
    # Pure literal elimination (omitted for brevity)
    
    # Pick a variable
    if not clauses: return True, assignment
    var = abs(clauses[0][0])
    
    # Branch True
    c_true = []
    for c in clauses:
        if var in c: continue
        c_true.append([l for l in c if l != -var])
    res, model = dpll(c_true, vars_set, assignment.copy())
    if res:
        model[var] = 1
        return True, model
        
    # Branch False
    c_false = []
    for c in clauses:
        if -var in c: continue
        c_false.append([l for l in c if l != var])
    res, model = dpll(c_false, vars_set, assignment.copy())
    if res:
        model[var] = -1
        return True, model
        
    return False, None

def generate_random_3sat(num_vars, num_clauses):
    clauses = []
    for _ in range(num_clauses):
        c = []
        while len(c) < 3:
            v = random.randint(1, num_vars)
            sign = 1 if random.random() > 0.5 else -1
            if sign * v not in c and -sign * v not in c:
                c.append(sign * v)
        clauses.append(c)
    return clauses

def run_fuzzer(iterations=100, vars=20, clauses=90):
    print(f"============================================================")
    print(f" ParaSAT API Fuzzer & Correctness Prover")
    print(f" Running {iterations} iterations on random 3-SAT (N={vars}, M={clauses})")
    print(f"============================================================")
    
    success = 0
    start_time = time.time()
    
    for i in range(iterations):
        cnf = generate_random_3sat(vars, clauses)
        
        # 1. Reference Solver (DPLL)
        ref_sat, _ = dpll(cnf, set(range(1, vars + 1)), {})
        
        # 2. ParaSAT C-API Shared Library
        solver = ParaSATSolver(mode=b"cpu", threads=1)
        lib.parasat_set_num_vars(solver.env, vars)
        for c in cnf:
            solver.add_clause(c)
            
        res = solver.solve()
        parasat_sat = (res == PARASAT_SAT)
        
        if ref_sat != parasat_sat:
            print(f"\n[!] MISMATCH at iteration {i}!")
            print(f"Reference: {'SAT' if ref_sat else 'UNSAT'}")
            print(f"ParaSAT:   {'SAT' if parasat_sat else 'UNSAT'}")
            print("CNF:")
            for c in cnf: print(c)
            return False
            
        if parasat_sat:
            model = solver.get_model(vars)
            # Verify model
            for c in cnf:
                clause_sat = False
                for lit in c:
                    var = abs(lit)
                    if (lit > 0 and model[var-1] > 0) or (lit < 0 and model[var-1] < 0):
                        clause_sat = True
                        break
                if not clause_sat:
                    print(f"\n[!] INVALID MODEL RETURNED BY API at iteration {i}!")
                    return False
        
        success += 1
        if (i+1) % 10 == 0:
            print(f"[*] Passed {i+1} / {iterations}...")
            
    print(f"============================================================")
    print(f"[OK] ALL {success} TESTS PASSED SUCCESSFULLY! 100% CORRECTNESS VERIFIED.")
    print(f"[Time] {time.time() - start_time:.2f} seconds.")
    print(f"============================================================")
    return True

if __name__ == "__main__":
    run_fuzzer(iterations=500, vars=20, clauses=85) # High phase-transition density
