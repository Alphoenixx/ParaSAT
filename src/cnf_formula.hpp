#pragma once

#include "simd_types.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <iomanip>

struct CNFFormula {
    int num_vars = 0;
    int num_clauses = 0;
    int num_clause_words = 0; // Number of uint64_t words per clause bitset row

    // Standard representation
    std::vector<std::vector<int>> clauses;
    std::vector<int> clause_lengths;

    // Binary clause indices for fast 2-SAT / BIG propagation
    std::vector<std::pair<int, int>> binary_clauses;
    int num_binary_clauses = 0;
    int num_unit_clauses = 0;

    // Literal occurrence lists: positive and negative
    std::vector<std::vector<int>> pos_occur; // clauses where +v appears
    std::vector<std::vector<int>> neg_occur; // clauses where -v appears

    // 256-bit AVX2-aligned bitset matrices for Stochastic Local Search (SLS)
    // var_pos_clause_bitsets[v * num_clause_words + (c / 64)] & (1ULL << (c % 64))
    std::vector<uint64_t> var_pos_clause_bitsets;
    std::vector<uint64_t> var_neg_clause_bitsets;

    bool has_empty_clause = false;

    static CNFFormula load_dimacs_cnf(const std::string& filepath) {
        CNFFormula cnf;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CNF file: " + filepath);
        }

        std::string line;
        bool header_found = false;
        std::vector<int> current_clause;

        while (std::getline(file, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            size_t first_non_space = line.find_first_not_of(" \t");
            if (first_non_space == std::string::npos) continue;
            std::string trimmed = line.substr(first_non_space);
            if (trimmed.empty()) continue;

            if (trimmed[0] == 'c') {
                continue; // Comment line
            } else if (trimmed[0] == 'p') {
                std::istringstream iss(trimmed);
                std::string p_str, format;
                int nv = 0, nc = 0;
                iss >> p_str >> format >> nv >> nc;
                cnf.num_vars = nv;
                cnf.num_clauses = nc;

                cnf.clauses.reserve(nc);
                cnf.clause_lengths.reserve(nc);
                cnf.pos_occur.resize(nv + 1);
                cnf.neg_occur.resize(nv + 1);

                header_found = true;
            } else {
                if (!header_found) {
                    cnf.num_vars = 0;
                    cnf.num_clauses = 0;
                }
                std::istringstream iss(trimmed);
                int lit = 0;
                while (iss >> lit) {
                    if (lit == 0) {
                        if (current_clause.empty()) {
                            // Explicit empty clause: formula is trivially UNSAT!
                            cnf.has_empty_clause = true;
                            cnf.clauses.push_back({});
                            cnf.clause_lengths.push_back(0);
                        } else {
                            int c_idx = static_cast<int>(cnf.clauses.size());
                            cnf.clauses.push_back(current_clause);
                            cnf.clause_lengths.push_back(static_cast<int>(current_clause.size()));

                            if (current_clause.size() == 1) {
                                cnf.num_unit_clauses++;
                            } else if (current_clause.size() == 2) {
                                cnf.binary_clauses.emplace_back(current_clause[0], current_clause[1]);
                                cnf.num_binary_clauses++;
                            }

                            for (int l : current_clause) {
                                int var = std::abs(l);
                                if (var >= static_cast<int>(cnf.pos_occur.size())) {
                                    cnf.pos_occur.resize(var + 1);
                                    cnf.neg_occur.resize(var + 1);
                                }
                                if (var > cnf.num_vars) cnf.num_vars = var;

                                if (l > 0) {
                                    cnf.pos_occur[var].push_back(c_idx);
                                } else {
                                    cnf.neg_occur[var].push_back(c_idx);
                                }
                            }
                            current_clause.clear();
                        }
                    } else {
                        current_clause.push_back(lit);
                    }
                }
            }
        }

        // Clamp to actual clauses parsed
        cnf.num_clauses = static_cast<int>(cnf.clauses.size());

        cnf.num_clause_words = (cnf.num_clauses + 63) / 64;
        if (cnf.num_clause_words % 4 != 0) {
            cnf.num_clause_words += (4 - (cnf.num_clause_words % 4));
        }

        // Build AVX2-aligned clause bitset matrices
        int total_words = (cnf.num_vars + 1) * cnf.num_clause_words;
        cnf.var_pos_clause_bitsets.assign(total_words, 0ULL);
        cnf.var_neg_clause_bitsets.assign(total_words, 0ULL);

        for (int v = 1; v <= cnf.num_vars; ++v) {
            for (int c_idx : cnf.pos_occur[v]) {
                int word = c_idx / 64;
                int bit = c_idx % 64;
                cnf.var_pos_clause_bitsets[v * cnf.num_clause_words + word] |= (1ULL << bit);
            }
            for (int c_idx : cnf.neg_occur[v]) {
                int word = c_idx / 64;
                int bit = c_idx % 64;
                cnf.var_neg_clause_bitsets[v * cnf.num_clause_words + word] |= (1ULL << bit);
            }
        }

        return cnf;
    }

    void print_summary() const {
        double binary_pct = (num_clauses > 0) ? (100.0 * num_binary_clauses / num_clauses) : 0.0;
        double ratio = (num_vars > 0) ? (static_cast<double>(num_clauses) / num_vars) : 0.0;
        std::cout << "[CNF Parser] Vars: " << num_vars 
                  << " | Clauses: " << num_clauses 
                  << " | Clause/Var Ratio: " << std::fixed << std::setprecision(2) << ratio
                  << " | Binary Clauses: " << num_binary_clauses << " (" << binary_pct << "%)"
                  << " | Unit Clauses: " << num_unit_clauses
                  << " | Words/Var: " << num_clause_words << "\n" << std::flush;
    }
};
