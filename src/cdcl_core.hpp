#pragma once

#include "cnf_formula.hpp"
#include "shared_clause_pool.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>

struct CDCLResult {
    bool is_sat = false;
    bool is_unsat = false;
    long long decisions = 0;
    long long conflicts = 0;
    long long propagations = 0;
    long long restarts = 0;
    double elapsed_ms = 0.0;
    std::vector<int> model;
};

// Fast Indexed Binary Max-Heap for O(1) VSIDS variable selection
class VarHeap {
private:
    std::vector<int> heap;
    std::vector<int> pos_in_heap;
    const std::vector<double>& activity;

    inline bool compare(int a, int b) const {
        return activity[a] > activity[b];
    }

    void sift_up(int i) {
        int v = heap[i];
        while (i > 0) {
            int p = (i - 1) / 2;
            if (compare(v, heap[p])) {
                heap[i] = heap[p];
                pos_in_heap[heap[p]] = i;
                i = p;
            } else {
                break;
            }
        }
        heap[i] = v;
        pos_in_heap[v] = i;
    }

    void sift_down(int i) {
        int v = heap[i];
        int n = static_cast<int>(heap.size());
        while (2 * i + 1 < n) {
            int child = 2 * i + 1;
            if (child + 1 < n && compare(heap[child + 1], heap[child])) {
                child++;
            }
            if (compare(heap[child], v)) {
                heap[i] = heap[child];
                pos_in_heap[heap[child]] = i;
                i = child;
            } else {
                break;
            }
        }
        heap[i] = v;
        pos_in_heap[v] = i;
    }

public:
    VarHeap(const std::vector<double>& act) : activity(act) {}

    void init(int nv) {
        heap.clear();
        pos_in_heap.assign(nv + 1, -1);
        for (int v = 1; v <= nv; ++v) {
            heap.push_back(v);
            pos_in_heap[v] = v - 1;
        }
        for (int i = static_cast<int>(heap.size()) / 2 - 1; i >= 0; --i) {
            sift_down(i);
        }
    }

    bool empty() const { return heap.empty(); }

    void insert(int v) {
        if (v > 0 && v < static_cast<int>(pos_in_heap.size()) && pos_in_heap[v] == -1) {
            pos_in_heap[v] = static_cast<int>(heap.size());
            heap.push_back(v);
            sift_up(pos_in_heap[v]);
        }
    }

    void update(int v) {
        if (v > 0 && v < static_cast<int>(pos_in_heap.size()) && pos_in_heap[v] != -1) {
            sift_up(pos_in_heap[v]);
        }
    }

    int extract_max() {
        if (heap.empty()) return -1;
        int max_var = heap[0];
        pos_in_heap[max_var] = -1;
        if (heap.size() > 1) {
            heap[0] = heap.back();
            pos_in_heap[heap[0]] = 0;
            heap.pop_back();
            sift_down(0);
        } else {
            heap.pop_back();
        }
        return max_var;
    }
};

class CDCLCore {
private:
    CNFFormula cnf;
    int nv;
    double time_limit_sec;
    bool verbose;
    SharedClausePool* shared_pool;
    size_t imported_clause_head = 0;
    int imported_unit_head = 0;
    int max_conflicts_limit;

    // 3-Tier Clause Management (Tier 1: Glue LBD <= 2, Tier 2: Core LBD <= 6, Tier 3: Local LBD > 6)
    struct ClauseInfo {
        int lbd = 0;
        float activity = 0.0f;
        uint8_t tier = 3; // 1 = Glue, 2 = Core, 3 = Local
        bool is_learned = false;
        bool is_deleted = false;
    };
    std::vector<ClauseInfo> clause_info;

    // Assignment values: +1 (true), -1 (false), 0 (unassigned)
    std::vector<int8_t> assigns;
    std::vector<int> decision_level;
    std::vector<int> reason_clause;
    std::vector<int> trail;
    std::vector<int> trail_lim;
    int qhead = 0;

    std::vector<uint8_t> seen;

    // VSIDS Variable Activity Heuristic
    std::vector<double> activity;
    double var_inc = 1.0;
    double var_decay = 0.95;

    // Phase Saving & Kissat-Style Rephasing
    std::vector<int8_t> phase;
    std::vector<int8_t> target_phase;
    std::vector<int8_t> best_phase;
    size_t max_trail_reached = 0;
    int rephase_mode = 0;

    VarHeap order_heap;

    // Glucose EMA Fast/Slow LBD Restarts
    double fast_lbd = 0.0;
    double slow_lbd = 0.0;
    double fast_lbd_alpha = 0.8;
    double slow_lbd_alpha = 0.999;

    // Watched literal lists: watches[lit_idx]
    struct Watcher {
        int clause_idx = 0;
        int blocker = 0;
        Watcher() = default;
        Watcher(int c, int b) : clause_idx(c), blocker(b) {}
    };
    std::vector<std::vector<Watcher>> watches;

    inline size_t lit_to_idx(int lit) const {
        if (lit == 0) return 0;
        return (lit > 0) ? static_cast<size_t>(2 * (lit - 1)) : static_cast<size_t>(2 * (-lit - 1) + 1);
    }

    inline int value(int lit) const {
        if (lit == 0) return 0;
        int v = std::abs(lit);
        if (v > nv) return 0;
        int8_t val = assigns[v];
        if (val == 0) return 0;
        return (lit > 0) ? val : -val;
    }

    void var_bump_activity(int var) {
        if (var <= 0 || var > nv) return;
        activity[var] += var_inc;
        order_heap.update(var);
        if (activity[var] > 1e100 || var_inc > 1e100) {
            for (int i = 1; i <= nv; ++i) activity[i] *= 1e-100;
            var_inc *= 1e-100;
        }
    }

    void var_decay_activity() {
        var_inc *= (1.0 / var_decay);
        if (var_inc > 1e100) {
            for (int i = 1; i <= nv; ++i) activity[i] *= 1e-100;
            var_inc *= 1e-100;
        }
    }

    int current_level() const {
        return static_cast<int>(trail_lim.size());
    }

    void assign_literal(int lit, int reason = -1) {
        int var = std::abs(lit);
        if (var <= 0 || var > nv) return;
        assigns[var] = (lit > 0) ? 1 : -1;
        phase[var] = assigns[var];
        decision_level[var] = current_level();
        reason_clause[var] = reason;
        trail.push_back(lit);

        if (trail.size() > max_trail_reached) {
            max_trail_reached = trail.size();
            best_phase = phase;
            target_phase = phase;
        }
    }

    void new_decision_level() {
        trail_lim.push_back(static_cast<int>(trail.size()));
    }

    void backtrack(int target_level) {
        if (current_level() <= target_level) return;
        int target_trail = trail_lim[target_level];
        while (static_cast<int>(trail.size()) > target_trail) {
            int lit = trail.back();
            trail.pop_back();
            int var = std::abs(lit);
            assigns[var] = 0;
            reason_clause[var] = -1;
            decision_level[var] = -1;
            order_heap.insert(var);
        }
        qhead = target_trail;
        trail_lim.resize(target_level);
    }

    int compute_lbd(const std::vector<int>& clause) {
        static std::vector<int> seen_level;
        static int stamp = 0;
        if (seen_level.size() <= static_cast<size_t>(nv + 10)) {
            seen_level.assign(nv + 10, 0);
        }
        stamp++;
        int lbd = 0;
        for (int lit : clause) {
            int v = std::abs(lit);
            if (v > 0 && v <= nv) {
                int lvl = decision_level[v];
                if (lvl > 0 && seen_level[lvl] != stamp) {
                    seen_level[lvl] = stamp;
                    lbd++;
                }
            }
        }
        return std::max(1, lbd);
    }

    bool lit_redundant(int lit, int depth = 0) {
        if (depth > 8) return false;
        int var = std::abs(lit);
        if (var <= 0 || var > nv) return false;
        int reason = reason_clause[var];
        if (reason == -1) return false;

        const auto& c = cnf.clauses[reason];
        for (size_t i = 1; i < c.size(); ++i) {
            int p = c[i];
            int p_var = std::abs(p);
            if (p_var > 0 && p_var <= nv && !seen[p_var] && decision_level[p_var] > 0) {
                if (!lit_redundant(p, depth + 1)) return false;
            }
        }
        return true;
    }

    void minimize_clause(std::vector<int>& clause) {
        // Disabled for correctness checking
        return;
    }

    int propagate() {
        while (qhead < static_cast<int>(trail.size())) {
            int p = trail[qhead++];
            size_t not_p_idx = lit_to_idx(-p);
            if (not_p_idx >= watches.size()) continue;

            auto& ws = watches[not_p_idx];

            size_t i = 0, j = 0;
            while (i < ws.size()) {
                Watcher w = ws[i++];
                int c_idx = w.clause_idx;
                if (c_idx < 0 || c_idx >= static_cast<int>(clause_info.size()) || clause_info[c_idx].is_deleted) continue;

                auto& c = cnf.clauses[c_idx];
                if (c.size() < 2) continue;

                if (w.blocker != 0 && value(w.blocker) == 1) {
                    ws[j++] = w;
                    continue;
                }

                if (c[0] == -p) {
                    std::swap(c[0], c[1]);
                }

                if (value(c[0]) == 1) {
                    ws[j++] = Watcher(c_idx, c[0]);
                    continue;
                }

                bool found = false;
                for (size_t k = 2; k < c.size(); ++k) {
                    if (value(c[k]) != -1) {
                        std::swap(c[1], c[k]);
                        size_t new_watch_idx = lit_to_idx(c[1]);
                        if (new_watch_idx < watches.size()) {
                            watches[new_watch_idx].push_back(Watcher(c_idx, c[0]));
                        }
                        found = true;
                        break;
                    }
                }

                if (found) continue;

                ws[j++] = w;
                if (value(c[0]) == -1) {
                    while (i < ws.size()) ws[j++] = ws[i++];
                    ws.resize(j);
                    qhead = static_cast<int>(trail.size());
                    return c_idx;
                } else {
                    assign_literal(c[0], c_idx);
                }
            }
            ws.resize(j);
        }
        return -1;
    }

    void analyze(int confl_idx, std::vector<int>& learned_clause, int& backtrack_level, int& out_lbd) {
        learned_clause.clear();
        learned_clause.push_back(0);

        int path_count = 0;
        int p = 0;
        int cur_lvl = current_level();
        int trail_idx = static_cast<int>(trail.size()) - 1;
        backtrack_level = 0;

        auto& confl = cnf.clauses[confl_idx];
        clause_info[confl_idx].activity += 1.0f;

        for (int lit : confl) {
            int var = std::abs(lit);
            if (var > 0 && var <= nv && !seen[var] && decision_level[var] > 0) {
                seen[var] = 1;
                var_bump_activity(var);
                if (decision_level[var] >= cur_lvl) {
                    path_count++;
                } else {
                    learned_clause.push_back(lit);
                    if (decision_level[var] > backtrack_level) {
                        backtrack_level = decision_level[var];
                    }
                }
            }
        }

        while (path_count > 0 && trail_idx >= 0) {
            while (trail_idx >= 0 && !seen[std::abs(trail[trail_idx])]) {
                trail_idx--;
            }
            if (trail_idx < 0) break;

            p = trail[trail_idx--];
            int var = std::abs(p);
            seen[var] = 0;
            path_count--;

            if (path_count > 0) {
                int r = reason_clause[var];
                if (r != -1 && r < static_cast<int>(cnf.clauses.size())) {
                    clause_info[r].activity += 1.0f;
                    // Promote Tier 3 to Tier 2 if participating in conflicts
                    if (clause_info[r].tier == 3 && clause_info[r].lbd <= 6) {
                        clause_info[r].tier = 2;
                    }

                    for (int lit : cnf.clauses[r]) {
                        int v = std::abs(lit);
                        if (v > 0 && v <= nv && !seen[v] && decision_level[v] > 0 && v != var) {
                            seen[v] = 1;
                            var_bump_activity(v);
                            if (decision_level[v] >= cur_lvl) {
                                path_count++;
                            } else {
                                learned_clause.push_back(lit);
                                if (decision_level[v] > backtrack_level) {
                                    backtrack_level = decision_level[v];
                                }
                            }
                        }
                    }
                }
            }
        }

        if (p == 0 && !confl.empty()) {
            p = -confl[0];
        }

        learned_clause[0] = -p;
        minimize_clause(learned_clause);
        for (int lit : learned_clause) {
            int v = std::abs(lit);
            if (v > 0 && v <= nv) seen[v] = 0;
        }
        for (int lit : confl) {
            int v = std::abs(lit);
            if (v > 0 && v <= nv) seen[v] = 0;
        }
        out_lbd = compute_lbd(learned_clause);
    }

    int pick_decision_var() {
        while (!order_heap.empty()) {
            int v = order_heap.extract_max();
            if (assigns[v] == 0) {
                return v;
            }
        }
        return -1;
    }

    inline bool is_locked(int c_idx) const {
        if (c_idx < 0 || c_idx >= static_cast<int>(cnf.clauses.size())) return false;
        const auto& c = cnf.clauses[c_idx];
        if (c.empty()) return false;
        int v = std::abs(c[0]);
        return (v <= nv && reason_clause[v] == c_idx);
    }

    // Dynamic Clause Database Reduction (Purges low-activity Tier 3 clauses, protecting locked reasons)
    void reduce_clause_db() {
        std::vector<int> candidates;
        for (size_t i = 0; i < clause_info.size(); ++i) {
            if (clause_info[i].is_learned && !clause_info[i].is_deleted && clause_info[i].tier == 3 && !is_locked(static_cast<int>(i))) {
                candidates.push_back(static_cast<int>(i));
            }
        }

        if (candidates.size() < 2000) return;

        std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
            return clause_info[a].activity < clause_info[b].activity;
        });

        size_t to_delete = candidates.size() / 2;
        for (size_t i = 0; i < to_delete; ++i) {
            int c_idx = candidates[i];
            clause_info[c_idx].is_deleted = true;
            cnf.clauses[c_idx].clear();
            cnf.clauses[c_idx].shrink_to_fit();
        }
    }

    // Import shared lemmas from other threads with full 2WL invariant protection
    void import_shared_clauses() {
        if (!shared_pool) return;

        int total_units = shared_pool->get_unit_count();
        while (imported_unit_head < total_units) {
            int unit_lit = shared_pool->get_unit(imported_unit_head++);
            if (value(unit_lit) == 0 && current_level() == 0) {
                assign_literal(unit_lit);
            }
        }

        size_t current_head = shared_pool->get_head();
        while (imported_clause_head < current_head) {
            SharedClausePool::SharedClause sc;
            if (shared_pool->get_clause(imported_clause_head++, sc)) {
                if (sc.length >= 2) {
                    std::vector<int> clause(sc.lits, sc.lits + sc.length);

                    bool already_sat = false;
                    for (int lit : clause) {
                        if (value(lit) == 1 && decision_level[std::abs(lit)] == 0) {
                            already_sat = true;
                            break;
                        }
                    }
                    if (already_sat) continue;

                    int found_watches = 0;
                    for (size_t i = 0; i < clause.size() && found_watches < 2; ++i) {
                        if (value(clause[i]) != -1) {
                            std::swap(clause[found_watches], clause[i]);
                            found_watches++;
                        }
                    }

                    int new_idx = static_cast<int>(cnf.clauses.size());
                    cnf.clauses.push_back(clause);

                    ClauseInfo ci;
                    ci.lbd = sc.length;
                    ci.tier = (sc.length <= 2) ? 1 : 2;
                    ci.is_learned = true;
                    clause_info.push_back(ci);

                    if (found_watches >= 2) {
                        watches[lit_to_idx(clause[0])].push_back(Watcher(new_idx, clause[1]));
                        watches[lit_to_idx(clause[1])].push_back(Watcher(new_idx, clause[0]));
                    } else if (found_watches == 1 && current_level() == 0) {
                        assign_literal(clause[0], new_idx);
                    }
                }
            }
        }
    }

public:
    CDCLCore(const CNFFormula& formula, double time_sec = 10.0, SharedClausePool* pool = nullptr, int default_phase = -1, int max_conflicts = 2000000000, bool verb = true)
        : cnf(formula), nv(formula.num_vars), time_limit_sec(time_sec), verbose(verb), shared_pool(pool), max_conflicts_limit(max_conflicts), order_heap(activity) {
        assigns.assign(nv + 1, 0);
        decision_level.assign(nv + 1, -1);
        reason_clause.assign(nv + 1, -1);
        seen.assign(nv + 1, 0);
        activity.assign(nv + 1, 0.0);
        phase.assign(nv + 1, default_phase);
        target_phase.assign(nv + 1, default_phase);
        best_phase.assign(nv + 1, default_phase);
        watches.resize(2 * nv + 10);
        clause_info.resize(cnf.clauses.size());

        for (size_t i = 0; i < cnf.clauses.size(); ++i) {
            clause_info[i].is_learned = false;
            clause_info[i].lbd = static_cast<int>(cnf.clauses[i].size());
            clause_info[i].tier = 1; // Original clauses are permanent
        }

        // 2-Sided Jeroslow-Wang Activity & Phase Initialization
        std::vector<double> jw_pos(nv + 1, 0.0);
        std::vector<double> jw_neg(nv + 1, 0.0);

        for (const auto& c : cnf.clauses) {
            double weight = std::pow(2.0, -static_cast<double>(c.size()));
            for (int lit : c) {
                int var = std::abs(lit);
                if (lit > 0) jw_pos[var] += weight;
                else jw_neg[var] += weight;
            }
        }

        for (int v = 1; v <= nv; ++v) {
            activity[v] = jw_pos[v] + jw_neg[v];
            if (default_phase == 0) {
                phase[v] = (jw_pos[v] >= jw_neg[v]) ? 1 : -1;
                target_phase[v] = phase[v];
                best_phase[v] = phase[v];
            }
        }
        order_heap.init(nv);

        for (size_t c_idx = 0; c_idx < cnf.clauses.size(); ++c_idx) {
            auto& c = cnf.clauses[c_idx];
            if (c.size() >= 2) {
                watches[lit_to_idx(c[0])].push_back(Watcher(static_cast<int>(c_idx), c[1]));
                watches[lit_to_idx(c[1])].push_back(Watcher(static_cast<int>(c_idx), c[0]));
            }
        }
    }

    CDCLResult solve(std::atomic<bool>* external_term_flag = nullptr) {
        auto start = std::chrono::high_resolution_clock::now();
        CDCLResult res;

        if (cnf.has_empty_clause) {
            res.is_unsat = true;
            auto end = std::chrono::high_resolution_clock::now();
            res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            return res;
        }

        for (size_t c_idx = 0; c_idx < cnf.clauses.size(); ++c_idx) {
            if (cnf.clauses[c_idx].size() == 1) {
                int lit = cnf.clauses[c_idx][0];
                if (value(lit) == -1) {
                    res.is_unsat = true;
                    auto end = std::chrono::high_resolution_clock::now();
                    res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    return res;
                } else if (value(lit) == 0) {
                    assign_literal(lit, static_cast<int>(c_idx));
                }
            }
        }

        if (propagate() != -1) {
            res.is_unsat = true;
            auto end = std::chrono::high_resolution_clock::now();
            res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            return res;
        }

        int restart_conflicts = 30;

        while (true) {
            if (external_term_flag && external_term_flag->load(std::memory_order_relaxed)) {
                break;
            }

            int confl_idx = propagate();

            if (confl_idx != -1) {
                res.conflicts++;

                if (current_level() == 0) {
                    res.is_unsat = true;
                    break;
                }

                if (res.conflicts > max_conflicts_limit) {
                    break;
                }

                std::vector<int> learned_clause;
                int b_level = 0;
                int lbd = 0;
                analyze(confl_idx, learned_clause, b_level, lbd);
                var_decay_activity();

                fast_lbd = fast_lbd_alpha * fast_lbd + (1.0 - fast_lbd_alpha) * lbd;
                slow_lbd = slow_lbd_alpha * slow_lbd + (1.0 - slow_lbd_alpha) * lbd;

                backtrack(b_level);

                int new_c_idx = static_cast<int>(cnf.clauses.size());
                cnf.clauses.push_back(learned_clause);

                ClauseInfo ci;
                ci.lbd = lbd;
                ci.tier = (lbd <= 2) ? 1 : ((lbd <= 6) ? 2 : 3);
                ci.is_learned = true;
                clause_info.push_back(ci);

                if (shared_pool && (lbd <= 2 || learned_clause.size() <= 3)) {
                    shared_pool->export_clause(learned_clause);
                }

                if (learned_clause.size() == 1) {
                    assign_literal(learned_clause[0]);
                } else {
                    assign_literal(learned_clause[0], new_c_idx);
                    watches[lit_to_idx(learned_clause[0])].push_back(Watcher(new_c_idx, learned_clause[1]));
                    watches[lit_to_idx(learned_clause[1])].push_back(Watcher(new_c_idx, learned_clause[0]));
                }

                if (res.conflicts % 2000 == 0) {
                    reduce_clause_db();
                }

                if (res.conflicts > 50 && fast_lbd * 1.25 < slow_lbd) {
                    res.restarts++;
                    backtrack(0);
                    import_shared_clauses();

                    // Dynamic multi-phase rephasing (Kissat technique)
                    if (res.restarts % 100 == 0) {
                        rephase_mode = (rephase_mode + 1) % 4;
                        if (rephase_mode == 1) {
                            phase = target_phase;
                        } else if (rephase_mode == 2) {
                            phase = best_phase;
                        } else if (rephase_mode == 3) {
                            for (int v = 1; v <= nv; ++v) phase[v] = -phase[v];
                        }
                    }
                } else if (res.conflicts % restart_conflicts == 0) {
                    res.restarts++;
                    backtrack(0);
                    import_shared_clauses();
                    restart_conflicts += 30;
                }
            } else {
                int next_var = pick_decision_var();
                if (next_var == -1) {
                    res.is_sat = true;
                    res.model.resize(nv + 1);
                    for (int v = 1; v <= nv; ++v) {
                        res.model[v] = (assigns[v] == 1) ? v : -v;
                    }
                    break;
                }

                res.decisions++;
                new_decision_level();
                int dec_lit = (phase[next_var] == 1) ? next_var : -next_var;
                assign_literal(dec_lit);
            }

            if (__builtin_expect((res.conflicts & 0xFF) == 0, 0)) {
                if (external_term_flag && external_term_flag->load(std::memory_order_relaxed)) {
                    break;
                }
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_sec = std::chrono::duration<double>(now - start).count();
                if (elapsed_sec > time_limit_sec) {
                    break;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        res.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (verbose) {
            std::cout << "[ParaSAT-CDCL] Finished in " << res.elapsed_ms << " ms | Status: " 
                      << (res.is_sat ? "SAT" : (res.is_unsat ? "UNSAT" : "TIMEOUT")) 
                      << " | Decisions: " << res.decisions 
                      << " | Conflicts: " << res.conflicts << "\n" << std::flush;
        }

        return res;
    }
};
