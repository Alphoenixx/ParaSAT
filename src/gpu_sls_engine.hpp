#pragma once

#include "cnf_formula.hpp"
#include "opencl_wrapper.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <cmath>

struct GPUSATResult {
    bool is_sat = false;
    long long total_flips = 0;
    double elapsed_ms = 0.0;
    double flips_per_sec = 0.0;
    int winning_gpu_thread = -1;
    std::vector<int> model;
};

class GPUSLSEngine {
private:
    const CNFFormula& cnf;
    OpenCLContext& ocl;
    int num_gpu_workers;
    int work_group_size;
    double time_limit_sec;
    bool verbose;

    // OpenCL program and kernel handles
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    // GPU Buffers
    cl_mem d_clauses = nullptr;
    cl_mem d_offsets = nullptr;
    cl_mem d_sizes = nullptr;
    cl_mem d_solved = nullptr;
    cl_mem d_model = nullptr;
    cl_mem d_flips = nullptr;

    // OpenCL C GPU Kernel Source
    const char* kernel_src = R"CLC(
        // Fast PRNG: 32-bit xorshift
        inline uint xorshift32(uint* state) {
            uint x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            return x;
        }

        // Massively Parallel GPU ProbSAT Stochastic Local Search Kernel
        __kernel void gpu_probsat_search(
            __global const int* clauses_flat,
            __global const int* clause_offsets,
            __global const int* clause_sizes,
            const int num_vars,
            const int num_clauses,
            const int max_flips_per_step,
            const uint base_seed,
            __global volatile int* global_solved,
            __global int* out_model,
            __global long* out_total_flips
        ) {
            int gid = get_global_id(0);
            uint rng_state = base_seed + (uint)gid * 1664525u + 1013904223u;
            if (rng_state == 0) rng_state = 123456789u;

            // Thread-local variable truth assignments: packed bit array or char array
            // Up to 4096 variables supported per GPU walker
            #define MAX_VARS_GPU 2048
            char local_assign[MAX_VARS_GPU + 1];
            if (num_vars > MAX_VARS_GPU) return;

            // Random initial assignment
            for (int v = 1; v <= num_vars; ++v) {
                local_assign[v] = (xorshift32(&rng_state) & 1) ? 1 : -1;
            }

            int flips = 0;
            while (flips < max_flips_per_step) {
                flips++;

                // Check global early exit flag
                if (*global_solved != 0) return;

                // 1. Evaluate clause satisfaction and collect unsatisfied clause candidates
                int unsat_count = 0;
                int chosen_unsat_c = -1;

                for (int c = 0; c < num_clauses; ++c) {
                    int offset = clause_offsets[c];
                    int sz = clause_sizes[c];
                    bool sat = false;

                    for (int i = 0; i < sz; ++i) {
                        int lit = clauses_flat[offset + i];
                        int var = lit > 0 ? lit : -lit;
                        char val = local_assign[var];
                        if ((lit > 0 && val == 1) || (lit < 0 && val == -1)) {
                            sat = true;
                            break;
                        }
                    }

                    if (!sat) {
                        unsat_count++;
                        // Reservoir sampling to select random unsatisfied clause in O(1) space
                        if (chosen_unsat_c == -1 || (xorshift32(&rng_state) % (uint)unsat_count) == 0) {
                            chosen_unsat_c = c;
                        }
                    }
                }

                // If all clauses satisfied, solution found!
                if (unsat_count == 0) {
                    if (atomic_xchg(global_solved, 1) == 0) {
                        for (int v = 1; v <= num_vars; ++v) {
                            out_model[v] = (local_assign[v] == 1) ? v : -v;
                        }
                        out_model[0] = gid; // Record winning GPU thread ID
                    }
                    return;
                }

                // 2. Select variable from chosen unsatisfied clause to flip
                int c_offset = clause_offsets[chosen_unsat_c];
                int c_sz = clause_sizes[chosen_unsat_c];
                int best_flip_var = 0;
                int min_break = 999999;

                for (int i = 0; i < c_sz; ++i) {
                    int cand_lit = clauses_flat[c_offset + i];
                    int cand_var = cand_lit > 0 ? cand_lit : -cand_lit;

                    // Calculate break count: number of satisfied clauses that would become unsatisfied
                    int breaks = 0;
                    for (int c = 0; c < num_clauses; ++c) {
                        int off = clause_offsets[c];
                        int sz = clause_sizes[c];
                        int true_lits = 0;
                        bool cand_var_is_true = false;

                        for (int j = 0; j < sz; ++j) {
                            int l = clauses_flat[off + j];
                            int v = l > 0 ? l : -l;
                            char val = local_assign[v];
                            if ((l > 0 && val == 1) || (l < 0 && val == -1)) {
                                true_lits++;
                                if (v == cand_var) cand_var_is_true = true;
                            }
                        }

                        if (true_lits == 1 && cand_var_is_true) {
                            breaks++;
                        }
                    }

                    if (breaks < min_break) {
                        min_break = breaks;
                        best_flip_var = cand_var;
                        if (breaks == 0) break; // Zero-break greedy shortcut
                    }
                }

                // Noise probability fallback
                if (best_flip_var == 0 || (xorshift32(&rng_state) % 100) < 45) {
                    int r_idx = xorshift32(&rng_state) % (uint)c_sz;
                    int r_lit = clauses_flat[c_offset + r_idx];
                    best_flip_var = r_lit > 0 ? r_lit : -r_lit;
                }

                // Flip chosen variable
                local_assign[best_flip_var] = -local_assign[best_flip_var];
            }
        }
    )CLC";

public:
    GPUSLSEngine(const CNFFormula& formula, OpenCLContext& context, int workers = 4096, int work_group = 256, double time_sec = 10.0, bool verb = true)
        : cnf(formula), ocl(context), num_gpu_workers(workers), work_group_size(work_group), time_limit_sec(time_sec), verbose(verb) {
        if (!ocl.available) return;

        // Build flat clause representation for GPU coalesced memory
        std::vector<int> clauses_flat;
        std::vector<int> clause_offsets;
        std::vector<int> clause_sizes;
        clauses_flat.reserve(cnf.num_clauses * 4);
        clause_offsets.reserve(cnf.num_clauses);
        clause_sizes.reserve(cnf.num_clauses);

        for (const auto& c : cnf.clauses) {
            clause_offsets.push_back(static_cast<int>(clauses_flat.size()));
            clause_sizes.push_back(static_cast<int>(c.size()));
            for (int lit : c) {
                clauses_flat.push_back(lit);
            }
        }

        cl_int err = 0;
        size_t src_len = strlen(kernel_src);
        program = ocl.clCreateProgramWithSource(ocl.context, 1, &kernel_src, &src_len, &err);
        if (err != CL_SUCCESS) {
            if (verbose) std::cerr << "[GPU-SLS] Failed to create OpenCL program: " << err << "\n";
            return;
        }

        err = ocl.clBuildProgram(program, 1, &ocl.device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[8192];
            ocl.clGetProgramBuildInfo(program, ocl.device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            if (verbose) std::cerr << "[GPU-SLS] Build Log:\n" << log << "\n";
            return;
        }

        kernel = ocl.clCreateKernel(program, "gpu_probsat_search", &err);
        if (err != CL_SUCCESS) return;

        // Allocate Device Buffers
        d_clauses = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, clauses_flat.size() * sizeof(int), clauses_flat.data(), &err);
        d_offsets = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, clause_offsets.size() * sizeof(int), clause_offsets.data(), &err);
        d_sizes = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, clause_sizes.size() * sizeof(int), clause_sizes.data(), &err);
        d_solved = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(int), nullptr, &err);
        d_model = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, (cnf.num_vars + 2) * sizeof(int), nullptr, &err);
        d_flips = ocl.clCreateBuffer(ocl.context, CL_MEM_READ_WRITE, sizeof(long long), nullptr, &err);
    }

    ~GPUSLSEngine() {
        if (d_clauses && ocl.clReleaseMemObject) ocl.clReleaseMemObject(d_clauses);
        if (d_offsets && ocl.clReleaseMemObject) ocl.clReleaseMemObject(d_offsets);
        if (d_sizes && ocl.clReleaseMemObject) ocl.clReleaseMemObject(d_sizes);
        if (d_solved && ocl.clReleaseMemObject) ocl.clReleaseMemObject(d_solved);
        if (d_model && ocl.clReleaseMemObject) ocl.clReleaseMemObject(d_model);
        if (d_flips && ocl.clReleaseMemObject) ocl.clReleaseMemObject(d_flips);
        if (kernel && ocl.clReleaseKernel) ocl.clReleaseKernel(kernel);
        if (program && ocl.clReleaseProgram) ocl.clReleaseProgram(program);
    }

    GPUSATResult solve(std::atomic<bool>* external_term_flag = nullptr) {
        auto start = std::chrono::high_resolution_clock::now();
        GPUSATResult res;

        if (!ocl.available || !kernel) {
            if (verbose) std::cout << "[GPU-SLS] OpenCL GPU engine not available.\n";
            return res;
        }

        int h_solved = 0;
        long long h_flips = 0;
        std::vector<int> h_model(cnf.num_vars + 2, 0);

        ocl.clEnqueueWriteBuffer(ocl.queue, d_solved, CL_TRUE, 0, sizeof(int), &h_solved, 0, nullptr, nullptr);
        ocl.clEnqueueWriteBuffer(ocl.queue, d_flips, CL_TRUE, 0, sizeof(long long), &h_flips, 0, nullptr, nullptr);

        int nv = cnf.num_vars;
        int nc = cnf.num_clauses;
        int flips_per_step = 250;

        ocl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_clauses);
        ocl.clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_offsets);
        ocl.clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_sizes);
        ocl.clSetKernelArg(kernel, 3, sizeof(int), &nv);
        ocl.clSetKernelArg(kernel, 4, sizeof(int), &nc);
        ocl.clSetKernelArg(kernel, 5, sizeof(int), &flips_per_step);
        ocl.clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_solved);
        ocl.clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_model);
        ocl.clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_flips);

        size_t global_work_size = num_gpu_workers;
        size_t local_work_size = work_group_size;

        if (verbose) {
            std::cout << "[ParaSAT-GPU] Launching " << global_work_size << " GPU Threads (" 
                      << ocl.device_name << " | " << ocl.compute_units << " SMs)\n" << std::flush;
        }

        uint32_t seed_step = 42;
        long long total_evaluated_flips = 0;

        while (true) {
            if (external_term_flag && external_term_flag->load(std::memory_order_relaxed)) {
                break;
            }

            seed_step += 1337;
            ocl.clSetKernelArg(kernel, 6, sizeof(uint32_t), &seed_step);

            ocl.clEnqueueNDRangeKernel(ocl.queue, kernel, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, nullptr);
            ocl.clFinish(ocl.queue);

            total_evaluated_flips += static_cast<long long>(global_work_size) * flips_per_step;

            ocl.clEnqueueReadBuffer(ocl.queue, d_solved, CL_TRUE, 0, sizeof(int), &h_solved, 0, nullptr, nullptr);
            if (h_solved != 0) {
                res.is_sat = true;
                ocl.clEnqueueReadBuffer(ocl.queue, d_model, CL_TRUE, 0, (cnf.num_vars + 2) * sizeof(int), h_model.data(), 0, nullptr, nullptr);
                res.winning_gpu_thread = h_model[0];
                res.model.resize(cnf.num_vars + 1);
                for (int v = 1; v <= cnf.num_vars; ++v) {
                    res.model[v] = h_model[v];
                }
                break;
            }

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now - start).count();
            if (elapsed_sec > time_limit_sec) {
                break;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        res.total_flips = total_evaluated_flips;
        res.flips_per_sec = (res.elapsed_ms > 0) ? (total_evaluated_flips * 1000.0 / res.elapsed_ms) : 0.0;

        if (verbose && res.is_sat) {
            std::cout << "[ParaSAT-GPU] Solved by GPU Thread " << res.winning_gpu_thread 
                      << " in " << res.elapsed_ms << " ms (" 
                      << std::fixed << std::setprecision(2) << (res.flips_per_sec / 1e6) << " Million Flips/sec)\n" << std::flush;
        }

        return res;
    }
};
