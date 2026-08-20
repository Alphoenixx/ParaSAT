#pragma once

#include "cnf_formula.hpp"
#include "simd_types.hpp"
#include <vector>
#include <iostream>
#include <cstdint>
#include <chrono>

struct BIGResult {
    bool proved_unsat = false;
    int units_propagated = 0;
    std::vector<int> fixed_literals;
    double elapsed_ms = 0.0;
};

class SIMDBinaryImplicationGraph {
private:
    const CNFFormula& cnf;
    int nv;
    int num_lit_words;
    bool verbose;

    // Convert signed literal to 0-indexed positive integer:
    // +1 -> 0, -1 -> 1, +2 -> 2, -2 -> 3, ..., +v -> 2*(v-1), -v -> 2*(v-1)+1
    inline int lit_to_idx(int lit) const {
        return (lit > 0) ? (2 * (lit - 1)) : (2 * (-lit - 1) + 1);
    }

    inline int idx_to_lit(int idx) const {
        int v = (idx / 2) + 1;
        return (idx % 2 == 0) ? v : -v;
    }

    inline int neg_idx(int idx) const {
        return idx ^ 1;
    }

public:
    // Adjacency matrix for implications: 2*N x 2*N
    std::vector<uint64_t> big_adj;

    SIMDBinaryImplicationGraph(const CNFFormula& formula, bool verb = true)
        : cnf(formula), nv(formula.num_vars), verbose(verb) {
        int total_lits = 2 * nv;
        num_lit_words = (total_lits + 63) / 64;
        if (num_lit_words % 4 != 0) {
            num_lit_words += (4 - (num_lit_words % 4));
        }
        big_adj.assign(total_lits * num_lit_words, 0ULL);

        // Populate from binary clauses: (a v b) => ~a -> b and ~b -> a
        for (const auto& bc : cnf.binary_clauses) {
            int a = bc.first;
            int b = bc.second;

            int not_a = lit_to_idx(-a);
            int not_b = lit_to_idx(-b);
            int pos_a = lit_to_idx(a);
            int pos_b = lit_to_idx(b);

            // not_a implies pos_b
            big_adj[not_a * num_lit_words + (pos_b / 64)] |= (1ULL << (pos_b % 64));
            // not_b implies pos_a
            big_adj[not_b * num_lit_words + (pos_a / 64)] |= (1ULL << (pos_a % 64));
        }
    }

    // Propagate unit literal through BIG using 256-bit SIMD bitsets
    BIGResult propagate_units(const std::vector<int>& initial_units) {
        auto start = std::chrono::high_resolution_clock::now();
        BIGResult res;
        int total_lits = 2 * nv;

        std::vector<uint64_t> assigned_lits(num_lit_words, 0ULL);
        std::vector<int> queue;

        for (int u : initial_units) {
            int idx = lit_to_idx(u);
            if (!(assigned_lits[idx / 64] & (1ULL << (idx % 64)))) {
                assigned_lits[idx / 64] |= (1ULL << (idx % 64));
                queue.push_back(idx);
            }
        }

        size_t head = 0;
        while (head < queue.size()) {
            int curr = queue[head++];
            int neg_curr = neg_idx(curr);

            // Contradiction: both x and ~x assigned
            if (assigned_lits[neg_curr / 64] & (1ULL << (neg_curr % 64))) {
                res.proved_unsat = true;
                break;
            }

            // SIMD check for new implied literals
            const uint64_t* curr_implications = &big_adj[curr * num_lit_words];
            for (int w = 0; w < num_lit_words; ++w) {
                uint64_t new_impl = curr_implications[w] & ~assigned_lits[w];
                while (new_impl > 0) {
                    int b = __builtin_ctzll(new_impl);
                    int target_idx = w * 64 + b;
                    if (target_idx < total_lits) {
                        assigned_lits[w] |= (1ULL << b);
                        queue.push_back(target_idx);
                    }
                    new_impl &= ~(1ULL << b);
                }
            }
        }

        for (int idx : queue) {
            res.fixed_literals.push_back(idx_to_lit(idx));
        }
        res.units_propagated = static_cast<int>(res.fixed_literals.size());

        auto end = std::chrono::high_resolution_clock::now();
        res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (verbose) {
            std::cout << "[ParaSAT-BIG] Propagated " << res.units_propagated << " unit literals in " 
                      << res.elapsed_ms << " ms | UNSAT Contradiction = " 
                      << (res.proved_unsat ? "YES" : "NO") << "\n" << std::flush;
        }

        return res;
    }
};
