// ─────────────────────────────────────────────────────────────────────
// BoundedArrowPool — 500MB Safety Cap Arrow Memory Allocator
//
// Subclasses arrow::MemoryPool to enforce a strict upper bound on
// total Arrow allocations. Prevents malformed/malicious Arrow IPC
// payloads from exhausting workstation RAM.
//
// Thread-safe: uses std::atomic<int64_t> for allocation tracking.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <arrow/memory_pool.h>
#include <atomic>
#include <cstdint>

namespace algae::models {

class BoundedArrowPool : public arrow::MemoryPool {
public:
    /// @param limit_bytes Maximum total allocation (default: 500MB)
    explicit BoundedArrowPool(int64_t limit_bytes = 500 * 1024 * 1024)
        : m_limit(limit_bytes)
        , m_allocated(0)
        , m_peak(0)
        , m_backend(arrow::default_memory_pool())
    {}

    arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
        // Atomic pre-add to prevent TOCTOU races under concurrent allocation
        int64_t prev = m_allocated.fetch_add(size, std::memory_order_relaxed);
        if (prev + size > m_limit) {
            m_allocated.fetch_sub(size, std::memory_order_relaxed);
            return arrow::Status::OutOfMemory(
                "BoundedArrowPool: safety threshold exceeded (",
                std::to_string(prev + size), " > ", std::to_string(m_limit), " bytes)"
            );
        }

        arrow::Status status = m_backend->Allocate(size, alignment, out);
        if (!status.ok()) {
            m_allocated.fetch_sub(size, std::memory_order_relaxed);
            return status;
        }

        m_num_allocs.fetch_add(1, std::memory_order_relaxed);

        // Track peak for diagnostics
        int64_t current = prev + size;
        int64_t old_peak = m_peak.load(std::memory_order_relaxed);
        while (current > old_peak &&
               !m_peak.compare_exchange_weak(old_peak, current, std::memory_order_relaxed)) {}

        return arrow::Status::OK();
    }

    void Free(uint8_t* buffer, int64_t size, int64_t alignment) override {
        m_backend->Free(buffer, size, alignment);
        m_allocated.fetch_sub(size, std::memory_order_relaxed);
    }

    arrow::Status Reallocate(int64_t old_size, int64_t new_size,
                             int64_t alignment, uint8_t** ptr) override {
        int64_t delta = new_size - old_size;
        if (delta > 0) {
            int64_t prev = m_allocated.fetch_add(delta, std::memory_order_relaxed);
            if (prev + delta > m_limit) {
                m_allocated.fetch_sub(delta, std::memory_order_relaxed);
                return arrow::Status::OutOfMemory(
                    "BoundedArrowPool: safety threshold exceeded during reallocation"
                );
            }
        }

        arrow::Status status = m_backend->Reallocate(old_size, new_size, alignment, ptr);
        if (!status.ok()) {
            if (delta > 0) m_allocated.fetch_sub(delta, std::memory_order_relaxed);
            return status;
        }

        if (delta < 0) {
            m_allocated.fetch_add(delta, std::memory_order_relaxed); // delta is negative
        }

        return arrow::Status::OK();
    }

    int64_t bytes_allocated() const override {
        return m_allocated.load(std::memory_order_relaxed);
    }

    int64_t total_bytes_allocated() const override {
        return m_allocated.load(std::memory_order_relaxed);
    }

    int64_t num_allocations() const override {
        return m_num_allocs.load(std::memory_order_relaxed);
    }

    int64_t max_memory() const override { return m_limit; }

    /// Peak allocation observed (for SRE diagnostics)
    int64_t peak_allocated() const { return m_peak.load(std::memory_order_relaxed); }

    std::string backend_name() const override { return "BoundedArrowPool(500MB)"; }

private:
    const int64_t m_limit;
    std::atomic<int64_t> m_allocated;
    std::atomic<int64_t> m_peak;
    std::atomic<int64_t> m_num_allocs{0};
    arrow::MemoryPool* m_backend;
};

} // namespace algae::models
