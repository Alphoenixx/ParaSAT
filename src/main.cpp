#include "cnf_formula.hpp"
#include "simd_sls.hpp"
#include "simd_big.hpp"
#include "cdcl_core.hpp"
#include "portfolio_solver.hpp"
#include "opencl_wrapper.hpp"
#include "gpu_sls_engine.hpp"
#include "hardware_detector.hpp"
#include "hyper_fastpath.hpp"
#include "gpu_portfolio_solver.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

void print_help(const char* prog_name) {
    SystemHardwareSpecs hw = HardwareDetector::detect();
    std::cout << "========================================================================\n";
    std::cout << "  ParaSAT: Universal Dynamically Accelerated Propositional SAT Solver   \n";
    std::cout << "========================================================================\n";
    std::cout << "Usage: " << prog_name << " [options] <formula.cnf>\n\n";
    std::cout << "Execution Modes:\n";
    std::cout << "  --mode <cpu|gpu|cpu+gpu>   Execution mode (default: cpu+gpu if GPU present)\n";
    std::cout << "  --cpu                      Shortcut for --mode cpu\n";
    std::cout << "  --gpu                      Shortcut for --mode gpu\n";
    std::cout << "  --cpu-gpu                  Shortcut for --mode cpu+gpu\n\n";
    std::cout << "FastPath & Acceleration Options:\n";
    std::cout << "  --hyper                    Enable Next-Gen Polymorphic SIMD HyperPath (Default: Enabled)\n";
    std::cout << "  --no-hyper                 Disable HyperPath (fallback to standard preflight or full search)\n";
    std::cout << "  --no-preflight             Disable all pre-flight passes (forces full tree search)\n\n";
    std::cout << "Hardware Resource Allocation:\n";
    std::cout << "  -t, --cpu-threads <N>      Number of CPU worker threads\n";
    std::cout << "                             (Default if omitted: " << hw.default_cpu_threads << " threads / 50% of " << hw.total_cpu_threads << " available)\n";
    std::cout << "  --cuda-cores <N>           Number of CUDA cores / GPU threads\n";
    if (hw.cuda_available) {
        std::cout << "                             (Default if omitted: " << hw.default_cuda_cores << " cores / 50% of " << hw.total_cuda_cores << " on " << hw.gpu_name << ")\n";
    } else {
        std::cout << "                             (GPU not detected)\n";
    }
    std::cout << "  --gpu-threads <N>          Alias for --cuda-cores\n\n";
    std::cout << "General Solver Options:\n";
    std::cout << "  -T, --timeout <sec>        Timeout in seconds (default: 10.0, 0 = unlimited)\n";
    std::cout << "  --device-info              Display CPU & GPU hardware specs and default limits\n";
    std::cout << "  -c, --certify              Verify SAT model against all original CNF clauses\n";
    std::cout << "  -p, --print-model          Output standard DIMACS 'v' truth assignments\n";
    std::cout << "  -q, --quiet                Quiet mode (suppress diagnostic banners)\n";
    std::cout << "  -h, --help                 Show this help message and exit\n\n";
    std::cout << "========================================================================\n";
}

bool verify_model(const CNFFormula& cnf, const std::vector<int>& model) {
    if (model.empty()) return false;
    for (size_t c_idx = 0; c_idx < cnf.clauses.size(); ++c_idx) {
        bool clause_sat = false;
        for (int lit : cnf.clauses[c_idx]) {
            int var = std::abs(lit);
            if (var < static_cast<int>(model.size())) {
                int assign = model[var];
                if ((lit > 0 && assign > 0) || (lit < 0 && assign < 0)) {
                    clause_sat = true;
                    break;
                }
            }
        }
        if (!clause_sat) {
            std::cout << "[Verification] FAILED: Clause " << c_idx << " is unsatisfied!\n";
            return false;
        }
    }
    std::cout << "[Verification] PASSED: All " << cnf.num_clauses << " clauses satisfied!\n";
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << std::unitbuf;

    SystemHardwareSpecs hw = HardwareDetector::detect();

    std::string cnf_path = "";
    std::string mode = hw.cuda_available ? "cpu+gpu" : "cpu";
    double timeout = 10.0;
    int cpu_threads = 0;
    int cuda_cores = 0;
    bool enable_preflight = true;
    bool enable_hyperpath = true;
    bool verbose = true;
    bool certify = false;
    bool print_model_flag = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--device-info" || arg == "--gpu-info") {
            hw.print_summary();
            return 0;
        } else if (arg == "--cpu") {
            mode = "cpu";
        } else if (arg == "--gpu") {
            mode = "gpu";
        } else if (arg == "--cpu-gpu" || arg == "--cpu+gpu") {
            mode = "cpu+gpu";
        } else if (arg == "--hyper") {
            enable_hyperpath = true;
            enable_preflight = true;
        } else if (arg == "--no-hyper") {
            enable_hyperpath = false;
        } else if (arg == "--no-preflight") {
            enable_hyperpath = false;
            enable_preflight = false;
        } else if (arg == "-m" || arg == "--mode") {
            if (i + 1 < argc) mode = argv[++i];
        } else if (arg == "-t" || arg == "--threads" || arg == "--cpu-threads") {
            if (i + 1 < argc) cpu_threads = std::stoi(argv[++i]);
        } else if (arg == "--cuda-cores" || arg == "--gpu-threads") {
            if (i + 1 < argc) cuda_cores = std::stoi(argv[++i]);
        } else if (arg == "-T" || arg == "--timeout") {
            if (i + 1 < argc) timeout = std::stod(argv[++i]);
        } else if (arg == "-q" || arg == "--quiet") {
            verbose = false;
        } else if (arg == "-c" || arg == "--certify") {
            certify = true;
        } else if (arg == "-p" || arg == "--print-model") {
            print_model_flag = true;
        } else if (arg[0] != '-') {
            cnf_path = arg;
        }
    }

    if (cnf_path.empty()) {
        std::cerr << "Error: No CNF formula specified.\n";
        print_help(argv[0]);
        return 1;
    }

    int resolved_cpu_threads = (cpu_threads > 0) ? cpu_threads : hw.default_cpu_threads;
    int resolved_cuda_cores = (cuda_cores > 0) ? cuda_cores : hw.default_cuda_cores;

    if (verbose) {
        std::cout << "========================================================================\n";
        std::cout << "  ParaSAT: Universal Dynamically Accelerated Propositional SAT Solver   \n";
        std::cout << "========================================================================\n";
        std::cout << "Target CNF:       " << cnf_path << "\n";
        std::cout << "Selected Mode:    " << mode << "\n";
        std::cout << "FastPath Engine:  " << (enable_hyperpath ? "HyperPath (Polymorphic SIMD)" : (enable_preflight ? "Standard Pre-Flight" : "Disabled")) << "\n";
        if (mode == "cpu" || mode == "cpu+gpu") {
            std::cout << "CPU Threads:      " << resolved_cpu_threads << " (out of " << hw.total_cpu_threads << " available" 
                      << (cpu_threads <= 0 ? " [50% Default]" : " [User Specified]") << ")\n";
        }
        if (mode == "gpu" || mode == "cpu+gpu") {
            if (hw.cuda_available) {
                std::cout << "CUDA Cores:       " << resolved_cuda_cores << " (out of " << hw.total_cuda_cores << " on " 
                          << hw.gpu_name << (cuda_cores <= 0 ? " [50% Default]" : " [User Specified]") << ")\n";
            } else {
                std::cout << "CUDA Cores:       GPU Not Available (Falling back to CPU)\n";
            }
        }
        std::cout << "Timeout Limit:    " << timeout << " sec\n";
        std::cout << "========================================================================\n" << std::flush;
    }

    CNFFormula cnf;
    try {
        cnf = CNFFormula::load_dimacs_cnf(cnf_path);
        if (verbose) cnf.print_summary();
    } catch (const std::exception& e) {
        std::cerr << "Error loading CNF formula: " << e.what() << "\n";
        return 1;
    }

    ParaSATGPUPortfolio solver(cnf, timeout, mode, resolved_cpu_threads, resolved_cuda_cores, enable_preflight, enable_hyperpath, verbose);
    GPUPortfolioResult res = solver.solve();

    if (verbose) {
        std::cout << "\n========================================================================\n";
        std::cout << "                        PARASAT FINAL SUMMARY                           \n";
        std::cout << "========================================================================\n";
        std::cout << "Status:          " << (res.is_sat ? "SATISFIABLE (SAT)" : (res.is_unsat ? "UNSATISFIABLE (UNSAT)" : "TIMEOUT")) << "\n";
        std::cout << "Winning Engine:  " << res.winning_engine << "\n";
        std::cout << "Execution Time:  " << std::fixed << std::setprecision(2) << res.elapsed_ms << " ms\n";
        std::cout << "========================================================================\n";
    }

    if (res.is_sat) {
        std::cout << "s SATISFIABLE\n";
        if (certify) {
            verify_model(cnf, res.model);
        }
        if (print_model_flag && !res.model.empty()) {
            std::cout << "v ";
            for (int v = 1; v <= cnf.num_vars; ++v) {
                int val = (v < static_cast<int>(res.model.size()) && res.model[v] != 0) ? res.model[v] : v;
                std::cout << val << " ";
                if (v % 20 == 0) std::cout << "\nv ";
            }
            std::cout << "0\n";
        }
    } else if (res.is_unsat) {
        std::cout << "s UNSATISFIABLE\n";
    } else {
        std::cout << "s UNKNOWN\n";
    }

    return res.is_sat ? 10 : (res.is_unsat ? 20 : 0);
}
