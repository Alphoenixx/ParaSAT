#pragma once

#include "cnf_formula.hpp"
#include "inprocess_engine.hpp"
#include "simd_sls.hpp"
#include "simd_big.hpp"
#include "cdcl_core.hpp"
#include "shared_clause_pool.hpp"
#include "opencl_wrapper.hpp"
#include "gpu_sls_engine.hpp"
#include "hardware_detector.hpp"
#include "hyper_fastpath.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>

struct GPUPortfolioResult {
    bool is_sat = false;
    bool is_unsat = false;
    std::string winning_engine;
    double elapsed_ms = 0.0;
    std::vector<int> model;
};

class ParaSATGPUPortfolio {
private:
    const CNFFormula& initial_cnf;
    double time_limit_sec;
    std::string mode; // "cpu", "gpu", "cpu+gpu"
    int num_cpu_threads;
    int num_cuda_cores;
    bool enable_preflight;
    bool enable_hyperpath;
    bool verbose;

public:
    ParaSATGPUPortfolio(const CNFFormula& formula, double time_sec = 10.0, const std::string& run_mode = "cpu+gpu", int cpu_threads = 0, int cuda_cores = 0, bool preflight = true, bool enable_hyper = true, bool verb = true)
        : initial_cnf(formula), time_limit_sec(time_sec), mode(run_mode), enable_preflight(preflight), enable_hyperpath(enable_hyper), verbose(verb) {
        
        SystemHardwareSpecs hw = HardwareDetector::detect();

        if (cpu_threads <= 0) {
            num_cpu_threads = hw.default_cpu_threads;
        } else {
            num_cpu_threads = cpu_threads;
        }

        if (cuda_cores <= 0) {
            num_cuda_cores = hw.default_cuda_cores;
        } else {
            num_cuda_cores = cuda_cores;
        }

        if (num_cuda_cores % 32 != 0) {
            num_cuda_cores = std::max(32, ((num_cuda_cores + 31) / 32) * 32);
        }
    }

    GPUPortfolioResult solve() {
        auto start = std::chrono::high_resolution_clock::now();
        GPUPortfolioResult res;

        // 1. Trivial Empty Clause Refutation
        if (initial_cnf.has_empty_clause) {
            res.is_unsat = true;
            res.winning_engine = "Trivial Empty Clause Contradiction";
            auto end = std::chrono::high_resolution_clock::now();
            res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (verbose) {
                std::cout << "[ParaSAT] Instant UNSAT certified (Empty Clause Present) in " << res.elapsed_ms << " ms\n" << std::flush;
            }
            return res;
        }

        // 2. Layer 1: Zero-Overhead Micro-Pass (Single-Core JW-CDCL for < 500 conflicts, 0 OpenMP overhead)
        if (enable_preflight && mode != "gpu") {
            CDCLCore micropass(initial_cnf, 0.05, nullptr, 0, 500, false);
            CDCLResult micro_res = micropass.solve();
            if (micro_res.is_sat) {
                auto end = std::chrono::high_resolution_clock::now();
                res.is_sat = true;
                res.winning_engine = "ParaSAT-MicroPass (Jeroslow-Wang CDCL)";
                res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                res.model = micro_res.model;
                if (verbose) {
                    std::cout << "[ParaSAT-Adaptive] Solved in Micro-Pass in " << res.elapsed_ms << " ms\n" << std::flush;
                }
                return res;
            } else if (micro_res.is_unsat) {
                auto end = std::chrono::high_resolution_clock::now();
                res.is_unsat = true;
                res.winning_engine = "ParaSAT-MicroPass 1-UIP Resolution";
                res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                if (verbose) {
                    std::cout << "[ParaSAT-Adaptive] Proved UNSAT in Micro-Pass in " << res.elapsed_ms << " ms\n" << std::flush;
                }
                return res;
            }
        }

        // 3. Layer 2: Next-Gen Polymorphic SIMD HyperPath (Multi-Stream Speculation + SIMD-BIG)
        if (enable_hyperpath && mode != "gpu") {
            HyperPathEngine hyper_engine(initial_cnf, num_cpu_threads, false);
            HyperPathResult hp_res = hyper_engine.solve();
            if (hp_res.is_sat) {
                auto end = std::chrono::high_resolution_clock::now();
                res.is_sat = true;
                res.winning_engine = hp_res.engine_name;
                res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                res.model = hp_res.model;
                if (verbose) {
                    std::cout << "[ParaSAT-Adaptive] Solved in HyperPath in " << res.elapsed_ms << " ms (" << res.winning_engine << ")\n" << std::flush;
                }
                return res;
            } else if (hp_res.is_unsat) {
                auto end = std::chrono::high_resolution_clock::now();
                res.is_unsat = true;
                res.winning_engine = hp_res.engine_name;
                res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                if (verbose) {
                    std::cout << "[ParaSAT-Adaptive] Proved UNSAT in HyperPath in " << res.elapsed_ms << " ms (" << res.winning_engine << ")\n" << std::flush;
                }
                return res;
            }
        }

        // 4. Layer 3: Deep In-Processing Pipeline (Bounded Variable Elimination & BCP Simplification)
        CNFFormula cnf = initial_cnf;
        InprocessEngine inproc(cnf, verbose);
        if (mode != "gpu") {
            InprocessResult inproc_res = inproc.run_inprocessing(0);
            if (inproc_res.proved_unsat) {
                auto end = std::chrono::high_resolution_clock::now();
                res.is_unsat = true;
                res.winning_engine = "ParaSAT-Inprocess (BVE/BCP Contradiction)";
                res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                if (verbose) {
                    std::cout << "[ParaSAT] Proved UNSAT in In-Processing in " << res.elapsed_ms << " ms\n" << std::flush;
                }
                return res;
            }
        }

        // 5. Layer 4: Deep Multi-Core CPU + GPU Massive Heterogeneous Portfolio
        alignas(64) std::atomic<bool> global_done{false};
        alignas(64) std::atomic<bool> global_is_sat{false};
        alignas(64) std::atomic<bool> global_is_unsat{false};
        std::string winner_name = "NONE";
        std::vector<int> final_model(cnf.num_vars + 1, 0);

        SharedClausePool shared_pool;
        OpenCLContext ocl;

        bool run_gpu = (mode == "gpu" || mode == "cpu+gpu") && ocl.available;
        bool run_cpu = (mode == "cpu" || mode == "cpu+gpu");

        if (verbose) {
            std::cout << "========================================================================\n";
            std::cout << "  ParaSAT Deep Engine Mode: [" << mode << "]\n";
            if (run_cpu) {
                std::cout << "  -> Active CPU Threads:  " << num_cpu_threads << "\n";
            }
            if (run_gpu) {
                std::cout << "  -> Active CUDA Cores:   " << num_cuda_cores << " [" << ocl.device_name << "]\n";
            }
            std::cout << "========================================================================\n" << std::flush;
        }

        // Launch GPU SLS worker in background thread if GPU mode active
        std::thread gpu_worker_thread;
        if (run_gpu && cnf.num_vars <= 2048) {
            gpu_worker_thread = std::thread([&]() {
                GPUSLSEngine gpu_engine(cnf, ocl, num_cuda_cores, 256, time_limit_sec, false);
                GPUSATResult gpu_res = gpu_engine.solve(&global_done);

                if (gpu_res.is_sat && !global_done.load(std::memory_order_relaxed)) {
                    global_is_sat.store(true, std::memory_order_release);
                    global_done.store(true, std::memory_order_release);
                    #pragma omp critical
                    {
                        winner_name = "NVIDIA-GPU SLS (Thread " + std::to_string(gpu_res.winning_gpu_thread) + " on " + ocl.device_name + ")";
                        final_model = gpu_res.model;
                    }
                }
            });
        }

        // Launch CPU multi-core CDCL + AVX2 SLS workers if CPU mode active
        if (run_cpu) {
            int num_sls_threads = (num_cpu_threads >= 2) ? std::max(1, num_cpu_threads / 3) : 0;

            #pragma omp parallel num_threads(num_cpu_threads)
            {
                int tid = omp_get_thread_num();

                if (tid < num_sls_threads) {
                    double thread_noise = 0.45 + (tid * 0.04);
                    SIMDSLS sls_worker(cnf, 10000000, time_limit_sec, 1, thread_noise, false);
                    SATResult sls_res = sls_worker.solve_single(tid, &global_done);

                    if (sls_res.is_sat && !global_done.load(std::memory_order_relaxed)) {
                        global_is_sat.store(true, std::memory_order_release);
                        global_done.store(true, std::memory_order_release);
                        #pragma omp critical
                        {
                            winner_name = "CPU AVX2-SLS (Thread " + std::to_string(tid) + ")";
                            final_model = sls_res.model;
                        }
                    }
                } else {
                    int cdcl_idx = tid - num_sls_threads;
                    int default_ph = (cdcl_idx % 3 == 0) ? 0 : ((cdcl_idx % 3 == 1) ? -1 : 1);
                    CDCLCore cdcl_worker(cnf, time_limit_sec, &shared_pool, default_ph, 2000000000, false);
                    CDCLResult cdcl_res = cdcl_worker.solve(&global_done);

                    if (cdcl_res.is_sat && !global_done.load(std::memory_order_relaxed)) {
                        global_is_sat.store(true, std::memory_order_release);
                        global_done.store(true, std::memory_order_release);
                        #pragma omp critical
                        {
                            winner_name = "CPU ParaSAT-CDCL (Thread " + std::to_string(tid) + ")";
                            final_model = cdcl_res.model;
                        }
                    } else if (cdcl_res.is_unsat && !global_done.load(std::memory_order_relaxed)) {
                        global_is_unsat.store(true, std::memory_order_release);
                        global_done.store(true, std::memory_order_release);
                        #pragma omp critical
                        {
                            winner_name = "CPU ParaSAT-CDCL 1-UIP Resolution (Thread " + std::to_string(tid) + ")";
                        }
                    }
                }
            }
        }

        if (gpu_worker_thread.joinable()) {
            gpu_worker_thread.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        res.is_sat = global_is_sat.load();
        res.is_unsat = global_is_unsat.load();
        res.winning_engine = winner_name;
        res.model = final_model;
        if (res.is_sat) {
            inproc.reconstruct_model(res.model);
        }

        if (verbose) {
            std::cout << "[ParaSAT] Finished in " << res.elapsed_ms << " ms | Result: "
                      << (res.is_sat ? "SAT" : (res.is_unsat ? "UNSAT" : "TIMEOUT"))
                      << " | Winner: " << res.winning_engine << "\n" << std::flush;
        }

        return res;
    }
};
