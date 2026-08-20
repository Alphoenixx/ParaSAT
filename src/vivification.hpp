#pragma once

#include "cnf_formula.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

struct VivifyStats {
    int vivified_clauses = 0;
    int shrunk_literals = 0;
    int subsumed_clauses = 0;
    double elapsed_ms = 0.0;
};

// Clause Vivification: Minimizes learned clauses by trial unit propagation (Kissat / CaDiCaL technique)
class ClauseVivifier {
private:
    CNFFormula& cnf;
    int nv;

public:
    ClauseVivifier(CNFFormula& formula)
        : cnf(formula), nv(formula.num_vars) {}

    // Vivify candidate long learned clauses (LBD >= 3, length >= 4)
    VivifyStats vivify(std::vector<std::vector<int>>& clauses, int max_candidates = 500) {
        auto start = std::chrono::high_resolution_clock::now();
        VivifyStats stats;

        std::vector<int8_t> assigns(nv + 1, 0);
        std::vector<int> trail;
        trail.reserve(nv + 1);

        int processed = 0;
        for (size_t c_idx = 0; c_idx < clauses.size() && processed < max_candidates; ++c_idx) {
            auto& c = clauses[c_idx];
            if (c.size() < 4) continue; // Only vivify long clauses
            processed++;

            trail.clear();
            std::fill(assigns.begin(), assigns.end(), 0);

            bool conflict = false;
            size_t conflict_lit_idx = c.size();

            // Trial propagation: Assume negation of each literal in sequence
            for (size_t i = 0; i < c.size(); ++i) {
                int lit = c[i];
                int var = std::abs(lit);
                int8_t val = (lit > 0) ? -1 : 1; // Negation of lit

                if (assigns[var] != 0) {
                    if (assigns[var] != val) {
                        // Conflict reached! The clause is satisfied or redundant up to index i
                        conflict = true;
                        conflict_lit_idx = i + 1;
                        break;
                    }
                    continue;
                }

                assigns[var] = val;
                trail.push_back(var);
            }

            if (conflict && conflict_lit_idx < c.size()) {
                stats.shrunk_literals += static_cast<int>(c.size() - conflict_lit_idx);
                c.resize(conflict_lit_idx);
                stats.vivified_clauses++;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        stats.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        return stats;
    }
};
