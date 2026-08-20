#pragma once

#include <vector>
#include <atomic>
#include <cstring>
#include <mutex>

// Thread-safe seqlock-protected circular buffer for sharing short learned clauses (LBD <= 2 or size <= 4)
class SharedClausePool {
public:
    static constexpr size_t POOL_CAPACITY = 8192;
    static constexpr size_t MAX_SHARED_CLAUSE_LEN = 4;

    struct SharedClause {
        int length = 0;
        int lits[MAX_SHARED_CLAUSE_LEN];
    };

    struct Slot {
        alignas(64) std::atomic<uint32_t> seq{0};
        SharedClause data;
    };

private:
    Slot buffer[POOL_CAPACITY];
    alignas(64) std::atomic<size_t> write_head{0};
    alignas(64) std::atomic<int> num_units{0};
    int unit_buffer[4096];
    std::mutex unit_mutex;

public:
    SharedClausePool() {
        for (size_t i = 0; i < POOL_CAPACITY; ++i) {
            buffer[i].seq.store(0, std::memory_order_relaxed);
            buffer[i].data.length = 0;
        }
    }

    void export_unit(int lit) {
        std::lock_guard<std::mutex> lock(unit_mutex);
        int idx = num_units.load(std::memory_order_relaxed);
        if (idx < 4096) {
            unit_buffer[idx] = lit;
            num_units.store(idx + 1, std::memory_order_release);
        }
    }

    void export_clause(const std::vector<int>& clause) {
        if (clause.empty() || clause.size() > MAX_SHARED_CLAUSE_LEN) return;
        if (clause.size() == 1) {
            export_unit(clause[0]);
            return;
        }

        size_t idx = write_head.fetch_add(1, std::memory_order_relaxed) % POOL_CAPACITY;
        Slot& slot = buffer[idx];

        uint32_t seq = slot.seq.load(std::memory_order_relaxed);
        slot.seq.store(seq + 1, std::memory_order_acquire); // Odd sequence = write in progress

        slot.data.length = static_cast<int>(clause.size());
        for (size_t i = 0; i < clause.size(); ++i) {
            slot.data.lits[i] = clause[i];
        }

        slot.seq.store(seq + 2, std::memory_order_release); // Even sequence = write complete
    }

    size_t get_head() const {
        return write_head.load(std::memory_order_acquire);
    }

    int get_unit_count() const {
        return num_units.load(std::memory_order_acquire);
    }

    int get_unit(int idx) const {
        return unit_buffer[idx];
    }

    bool get_clause(size_t index, SharedClause& out) const {
        size_t actual_idx = index % POOL_CAPACITY;
        const Slot& slot = buffer[actual_idx];

        for (int retries = 0; retries < 3; ++retries) {
            uint32_t seq1 = slot.seq.load(std::memory_order_acquire);
            if (seq1 & 1) continue; // Write in progress, retry

            out = slot.data;

            uint32_t seq2 = slot.seq.load(std::memory_order_acquire);
            if (seq1 == seq2 && out.length > 0) {
                return true; // Clean, non-torn read
            }
        }
        return false;
    }
};
