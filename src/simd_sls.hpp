#pragma once

#include "cnf_formula.hpp"
#include "simd_types.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <atomic>
#include <algorithm>
#include <cmath>

struct SATResult {
    bool is_sat = false;
    long long flips = 0;
    double elapsed_ms = 0.0;
    double flips_per_sec = 0.0;
    std::vector<int> model;
};

class SIMDSLS {
private:
    const CNFFormula& cnf;
    int max_flips;
    double time_limit_sec;
    int num_threads;
    double noise_prob;
    bool verbose;

public:
    SIMDSLS(const CNFFormula& formula, int flips = 20000000, double time_sec = 5.0, int threads = 16, double noise = 0.57, bool verb = true)
        : cnf(formula), max_flips(flips), time_limit_sec(time_sec), num_threads(threads), noise_prob(noise), verbose(verb) {}

    // Direct single-threaded execution with O(1) critical-variable caching for peak flip throughput
    SATResult solve_single(int seed_offset, std::atomic<bool>* term_flag) {
        auto start_time = std::chrono::high_resolution_clock::now();
        int nv = cnf.num_vars;
        int nc = cnf.num_clauses;

        std::mt19937 rng(42 + seed_offset * 1337);
        std::uniform_real_distribution<double> dist_01(0.0, 1.0);

        std::vector<uint8_t> assign(nv + 1);
        for (int v = 1; v <= nv; ++v) {
            assign[v] = rng() % 2;
        }

        std::vector<int> num_true_lits(nc, 0);
        std::vector<int> crit_var(nc, 0); // Holds the critical variable when num_true_lits == 1
        std::vector<int> unsat_clauses;
        std::vector<int> unsat_pos_in_list(nc, -1);
        unsat_clauses.reserve(nc);

        for (int c = 0; c < nc; ++c) {
            int true_count = 0;
            int last_true_var = 0;
            for (int lit : cnf.clauses[c]) {
                int var = std::abs(lit);
                bool val = (assign[var] == 1);
                if ((lit > 0 && val) || (lit < 0 && !val)) {
                    true_count++;
                    last_true_var = var;
                }
            }
            num_true_lits[c] = true_count;
            if (true_count == 0) {
                unsat_pos_in_list[c] = static_cast<int>(unsat_clauses.size());
                unsat_clauses.push_back(c);
            } else if (true_count == 1) {
                crit_var[c] = last_true_var;
            }
        }

        std::vector<int> break_count(nv + 1, 0);
        for (int c = 0; c < nc; ++c) {
            if (num_true_lits[c] == 1) {
                break_count[crit_var[c]]++;
            }
        }

        const double cb = 2.5;
        auto score_fn = [&](int b) -> double {
            return std::pow(static_cast<double>(b) + 0.1, -cb);
        };

        long long flips = 0;
        bool solved = false;
        std::vector<int> model(nv + 1, 0);
        std::vector<double> weights;

        while (flips < max_flips) {
            flips++;

            if ((flips & 0x7F) == 0 && term_flag && term_flag->load(std::memory_order_relaxed)) {
                break;
            }

            if (unsat_clauses.empty()) {
                solved = true;
                for (int v = 1; v <= nv; ++v) {
                    model[v] = (assign[v] == 1) ? v : -v;
                }
                break;
            }

            int unsat_idx = rng() % unsat_clauses.size();
            int chosen_c = unsat_clauses[unsat_idx];
            const auto& clause = cnf.clauses[chosen_c];

            int best_var = -1;
            int zero_break_var = -1;
            int sz = static_cast<int>(clause.size());
            if (weights.size() < static_cast<size_t>(sz)) {
                weights.resize(sz + 16, 0.0);
            }
            double total_weight = 0.0;

            for (int i = 0; i < sz; ++i) {
                int var = std::abs(clause[i]);
                int brk = break_count[var];
                if (brk == 0) {
                    zero_break_var = var;
                    break;
                }
                double w = score_fn(brk);
                weights[i] = w;
                total_weight += w;
            }

            if (zero_break_var != -1) {
                best_var = zero_break_var;
            } else if (dist_01(rng) < noise_prob || total_weight <= 0.0) {
                best_var = std::abs(clause[rng() % sz]);
            } else {
                double r = dist_01(rng) * total_weight;
                double cum = 0.0;
                for (int i = 0; i < sz; ++i) {
                    cum += weights[i];
                    if (r <= cum || i == sz - 1) {
                        best_var = std::abs(clause[i]);
                        break;
                    }
                }
            }

            if (best_var == -1) {
                best_var = std::abs(clause[rng() % sz]);
            }

            bool old_val = (assign[best_var] == 1);
            bool new_val = !old_val;
            assign[best_var] = new_val ? 1 : 0;

            const auto& became_true_clauses = new_val ? cnf.pos_occur[best_var] : cnf.neg_occur[best_var];
            const auto& became_false_clauses = new_val ? cnf.neg_occur[best_var] : cnf.pos_occur[best_var];

            // 1. Clauses where literal became TRUE
            for (int c : became_true_clauses) {
                int old_cnt = num_true_lits[c];
                num_true_lits[c] = old_cnt + 1;

                if (old_cnt == 0) {
                    int pos = unsat_pos_in_list[c];
                    int last_c = unsat_clauses.back();
                    unsat_clauses[pos] = last_c;
                    unsat_pos_in_list[last_c] = pos;
                    unsat_clauses.pop_back();
                    unsat_pos_in_list[c] = -1;

                    crit_var[c] = best_var;
                    break_count[best_var]++;
                } else if (old_cnt == 1) {
                    // O(1) update: previously critical variable is no longer critical
                    break_count[crit_var[c]]--;
                }
            }

            // 2. Clauses where literal became FALSE
            for (int c : became_false_clauses) {
                int old_cnt = num_true_lits[c];
                num_true_lits[c] = old_cnt - 1;

                if (old_cnt == 1) {
                    unsat_pos_in_list[c] = static_cast<int>(unsat_clauses.size());
                    unsat_clauses.push_back(c);
                    break_count[best_var]--;
                    crit_var[c] = 0;
                } else if (old_cnt == 2) {
                    // Find remaining single true literal
                    for (int lit : cnf.clauses[c]) {
                        int var = std::abs(lit);
                        if (var != best_var) {
                            bool v_val = (assign[var] == 1);
                            if ((lit > 0 && v_val) || (lit < 0 && !v_val)) {
                                crit_var[c] = var;
                                break_count[var]++;
                                break;
                            }
                        }
                    }
                }
            }

            if (__builtin_expect((flips & 0x1FFFF) == 0, 0)) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
                if (elapsed_sec > time_limit_sec) {
                    break;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        SATResult res;
        res.is_sat = solved;
        res.flips = flips;
        res.elapsed_ms = total_ms;
        res.flips_per_sec = (total_ms > 0) ? (flips * 1000.0 / total_ms) : 0.0;
        res.model = model;

        return res;
    }

    SATResult solve() {
        alignas(64) std::atomic<bool> global_solved{false};
        return solve_single(0, &global_solved);
    }
};
