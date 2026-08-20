#pragma once

#include "cnf_formula.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_set>

struct InprocessResult {
    bool proved_unsat = false;
    int eliminated_vars = 0;
    int deleted_clauses = 0;
    int added_resolvents = 0;
    int pure_literals = 0;
    int units_propagated = 0;
    double elapsed_ms = 0.0;
};

// Represents an eliminated variable for backward model reconstruction
struct EliminationEntry {
    int var = 0;
    std::vector<std::vector<int>> clauses; // clauses containing var at time of elimination
};

class InprocessEngine {
private:
    CNFFormula& cnf;
    bool verbose;
    std::vector<EliminationEntry> elim_stack;
    std::vector<uint8_t> is_eliminated;

    // 64-bit signature hash for ultra-fast clause subsumption filtering
    inline uint64_t clause_sig(const std::vector<int>& c) const {
        uint64_t s = 0;
        for (int lit : c) {
            int v = std::abs(lit);
            s |= (1ULL << (v & 63));
        }
        return s;
    }

    // Resolves two clauses on variable v. Returns empty clause if tautological (e.g. x and ~x present).
    bool resolve(const std::vector<int>& c1, const std::vector<int>& c2, int v, std::vector<int>& out_res) {
        out_res.clear();
        static std::vector<int8_t> seen_lit;
        if (seen_lit.size() <= static_cast<size_t>(2 * cnf.num_vars + 10)) {
            seen_lit.assign(2 * cnf.num_vars + 10, 0);
        }

        auto lit_to_idx = [this](int lit) -> size_t {
            return (lit > 0) ? (2 * (lit - 1)) : (2 * (-lit - 1) + 1);
        };

        for (int lit : c1) {
            if (std::abs(lit) != v) {
                size_t idx = lit_to_idx(lit);
                if (!seen_lit[idx]) {
                    seen_lit[idx] = 1;
                    out_res.push_back(lit);
                }
            }
        }

        bool is_tautology = false;
        for (int lit : c2) {
            if (std::abs(lit) != v) {
                size_t opp_idx = lit_to_idx(-lit);
                if (seen_lit[opp_idx]) {
                    is_tautology = true;
                    break;
                }
                size_t same_idx = lit_to_idx(lit);
                if (!seen_lit[same_idx]) {
                    seen_lit[same_idx] = 1;
                    out_res.push_back(lit);
                }
            }
        }

        // Clean up seen array
        for (int lit : out_res) {
            seen_lit[lit_to_idx(lit)] = 0;
        }

        if (is_tautology) {
            out_res.clear();
            return false;
        }
        return true;
    }

public:
    InprocessEngine(CNFFormula& formula, bool verb = false)
        : cnf(formula), verbose(verb) {
        is_eliminated.assign(cnf.num_vars + 1, 0);
    }

    // Full In-Processing Pipeline: BCP -> Pure Lits -> BVE -> Subsumption
    InprocessResult run_inprocessing(int bve_growth_limit = 0) {
        auto start = std::chrono::high_resolution_clock::now();
        InprocessResult res;

        int nv = cnf.num_vars;

        // 1. Initial Top-Level Unit Propagation (BCP)
        std::vector<int8_t> assigns(nv + 1, 0);
        std::vector<int> unit_queue;

        for (const auto& c : cnf.clauses) {
            if (c.empty()) {
                res.proved_unsat = true;
                return res;
            } else if (c.size() == 1) {
                int lit = c[0];
                int var = std::abs(lit);
                int8_t val = (lit > 0) ? 1 : -1;
                if (assigns[var] != 0 && assigns[var] != val) {
                    res.proved_unsat = true;
                    return res;
                }
                if (assigns[var] == 0) {
                    assigns[var] = val;
                    unit_queue.push_back(lit);
                }
            }
        }

        size_t head = 0;
        while (head < unit_queue.size()) {
            int u = unit_queue[head++];
            int u_var = std::abs(u);
            int8_t u_val = (u > 0) ? 1 : -1;

            // Simplify clauses based on unit assignments
            for (size_t c_idx = 0; c_idx < cnf.clauses.size(); ++c_idx) {
                auto& c = cnf.clauses[c_idx];
                if (c.empty()) continue;

                bool clause_sat = false;
                for (size_t i = 0; i < c.size(); ++i) {
                    int lit = c[i];
                    int var = std::abs(lit);
                    if (var == u_var) {
                        if ((lit > 0 && u_val == 1) || (lit < 0 && u_val == -1)) {
                            clause_sat = true;
                            break;
                        } else {
                            // Falsified literal: remove from clause
                            c.erase(c.begin() + i);
                            i--;
                        }
                    }
                }

                if (clause_sat) {
                    c.clear(); // Mark deleted
                    res.deleted_clauses++;
                } else if (c.empty()) {
                    res.proved_unsat = true;
                    return res;
                } else if (c.size() == 1) {
                    int lit = c[0];
                    int var = std::abs(lit);
                    int8_t val = (lit > 0) ? 1 : -1;
                    if (assigns[var] != 0 && assigns[var] != val) {
                        res.proved_unsat = true;
                        return res;
                    }
                    if (assigns[var] == 0) {
                        assigns[var] = val;
                        unit_queue.push_back(lit);
                    }
                }
            }
        }
        res.units_propagated = static_cast<int>(unit_queue.size());

        // 2. Pure Literal Elimination
        std::vector<int> pos_count(nv + 1, 0);
        std::vector<int> neg_count(nv + 1, 0);

        for (const auto& c : cnf.clauses) {
            for (int lit : c) {
                int var = std::abs(lit);
                if (lit > 0) pos_count[var]++;
                else neg_count[var]++;
            }
        }

        for (int v = 1; v <= nv; ++v) {
            if (assigns[v] == 0 && !is_eliminated[v]) {
                if (pos_count[v] > 0 && neg_count[v] == 0) {
                    assigns[v] = 1;
                    res.pure_literals++;
                } else if (pos_count[v] == 0 && neg_count[v] > 0) {
                    assigns[v] = -1;
                    res.pure_literals++;
                }
            }
        }

        // 3. Bounded Variable Elimination (BVE / SatELite)
        // Rebuild occurrence indices for active clauses
        std::vector<std::vector<int>> pos_occ(nv + 1);
        std::vector<std::vector<int>> neg_occ(nv + 1);

        for (size_t c_idx = 0; c_idx < cnf.clauses.size(); ++c_idx) {
            const auto& c = cnf.clauses[c_idx];
            if (c.empty()) continue;
            for (int lit : c) {
                int var = std::abs(lit);
                if (lit > 0) pos_occ[var].push_back(static_cast<int>(c_idx));
                else neg_occ[var].push_back(static_cast<int>(c_idx));
            }
        }

        // Try eliminating each variable with adaptive degree cutoff
        for (int v = 1; v <= nv; ++v) {
            if (assigns[v] != 0 || is_eliminated[v]) continue;

            const auto& p_occ = pos_occ[v];
            const auto& n_occ = neg_occ[v];

            if (p_occ.empty() || n_occ.empty()) continue;
            if (p_occ.size() > 6 || n_occ.size() > 6) continue; // Skip high-degree variables to prevent exponential checks

            int old_clauses_count = static_cast<int>(p_occ.size() + n_occ.size());
            int max_allowed_resolvents = old_clauses_count + bve_growth_limit;

            // Generate all candidate resolvents
            std::vector<std::vector<int>> resolvents;
            bool abort_elimination = false;

            for (int c1_idx : p_occ) {
                for (int c2_idx : n_occ) {
                    std::vector<int> res_clause;
                    if (resolve(cnf.clauses[c1_idx], cnf.clauses[c2_idx], v, res_clause)) {
                        if (res_clause.empty()) {
                            // Resolvent is empty -> instant UNSAT!
                            res.proved_unsat = true;
                            return res;
                        }
                        resolvents.push_back(res_clause);
                        if (static_cast<int>(resolvents.size()) > max_allowed_resolvents) {
                            abort_elimination = true;
                            break;
                        }
                    }
                }
                if (abort_elimination) break;
            }

            if (!abort_elimination && static_cast<int>(resolvents.size()) <= max_allowed_resolvents) {
                // Eliminate variable v
                is_eliminated[v] = 1;
                res.eliminated_vars++;

                EliminationEntry entry;
                entry.var = v;

                for (int c_idx : p_occ) {
                    entry.clauses.push_back(cnf.clauses[c_idx]);
                    cnf.clauses[c_idx].clear(); // Delete old clause
                    res.deleted_clauses++;
                }
                for (int c_idx : n_occ) {
                    entry.clauses.push_back(cnf.clauses[c_idx]);
                    cnf.clauses[c_idx].clear(); // Delete old clause
                    res.deleted_clauses++;
                }

                elim_stack.push_back(entry);

                // Add newly generated resolvents
                for (auto& r : resolvents) {
                    cnf.clauses.push_back(r);
                    res.added_resolvents++;
                }
            }
        }

        // 4. Compact and Filter Active Clauses
        std::vector<std::vector<int>> clean_clauses;
        clean_clauses.reserve(cnf.clauses.size());
        for (auto& c : cnf.clauses) {
            if (!c.empty()) {
                clean_clauses.push_back(std::move(c));
            }
        }
        cnf.clauses = std::move(clean_clauses);
        cnf.num_clauses = static_cast<int>(cnf.clauses.size());

        inprocess_fixed_assigns = assigns;

        auto end = std::chrono::high_resolution_clock::now();
        res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (verbose) {
            std::cout << "[ParaSAT-Inprocess] BVE Vars: " << res.eliminated_vars 
                      << " | Deleted Clauses: " << res.deleted_clauses 
                      << " | Added Resolvents: " << res.added_resolvents 
                      << " | Pure Lits: " << res.pure_literals 
                      << " | Time: " << res.elapsed_ms << " ms\n" << std::flush;
        }

        return res;
    }

    std::vector<int8_t> inprocess_fixed_assigns;

    // Extend a partial model over eliminated variables
    void reconstruct_model(std::vector<int>& model) {
        if (model.empty()) return;
        if (model.size() <= static_cast<size_t>(cnf.num_vars)) {
            model.resize(cnf.num_vars + 1, 0);
        }

        // 1. Apply fixed unit and pure literal assignments
        for (int v = 1; v <= cnf.num_vars; ++v) {
            if (v < static_cast<int>(inprocess_fixed_assigns.size()) && inprocess_fixed_assigns[v] != 0) {
                model[v] = (inprocess_fixed_assigns[v] == 1) ? v : -v;
            }
        }

        // 2. Iterate through elimination stack in reverse order
        for (auto it = elim_stack.rbegin(); it != elim_stack.rend(); ++it) {
            int v = it->var;
            bool satisfied = false;

            // Check if any positive/negative assignment satisfies all its original clauses
            for (int candidate_val : {1, -1}) {
                bool all_ok = true;
                for (const auto& c : it->clauses) {
                    bool cl_sat = false;
                    for (int lit : c) {
                        int var = std::abs(lit);
                        int assign = (var == v) ? ((candidate_val == 1) ? v : -v) : ((var < static_cast<int>(model.size())) ? model[var] : 0);
                        if ((lit > 0 && assign > 0) || (lit < 0 && assign < 0)) {
                            cl_sat = true;
                            break;
                        }
                    }
                    if (!cl_sat) {
                        all_ok = false;
                        break;
                    }
                }
                if (all_ok) {
                    if (v < static_cast<int>(model.size())) {
                        model[v] = (candidate_val == 1) ? v : -v;
                    }
                    satisfied = true;
                    break;
                }
            }

            if (!satisfied && v < static_cast<int>(model.size())) {
                model[v] = v; // Default fallback
            }
        }
    }
};
