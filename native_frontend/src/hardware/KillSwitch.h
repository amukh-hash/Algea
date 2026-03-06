// ─────────────────────────────────────────────────────────────────────
// KillSwitch — Out-Of-Band shared memory halt mechanism (C++ side)
//
// Writes to a boost::interprocess shared memory block and signals
// the backend daemon via POSIX SIGUSR1 (Linux) or named event 
// (Windows) for deterministic, network-independent circuit breaking.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

namespace algae::hardware {

/// Shared memory layout — must EXACTLY match the Python backend's struct.
/// 64-byte aligned for cache-line isolation.
struct alignas(64) SharedControlBlock {
    std::atomic<uint64_t> sequence_id{0};
    std::atomic<uint32_t> halt_mask{0};   // Bitmask: (1 << sleeve_id)
    char reason[52]{};
};

/// Out-Of-Band kill switch that bypasses TCP networking.
///
/// When the PM activates "Halt Sleeve N" in the UI, this writes
/// atomically to shared memory and fires a signal to the backend PID,
/// guaranteeing sub-microsecond circuit breaking even if the network
/// stack is congested.
class KillSwitch {
public:
    explicit KillSwitch(const std::string& shm_name = "AlgaeControlPlane");
    ~KillSwitch();

    // Non-copyable
    KillSwitch(const KillSwitch&) = delete;
    KillSwitch& operator=(const KillSwitch&) = delete;

    /// Open or create the shared memory segment
    bool initialize();

    /// Halt a specific sleeve
    void haltSleeve(uint32_t sleeve_id, const std::string& reason);

    /// Resume a specific sleeve
    void resumeSleeve(uint32_t sleeve_id);

    /// Resume all sleeves
    void resumeAll();

    /// Check if a sleeve is halted
    [[nodiscard]] bool isSleeveHalted(uint32_t sleeve_id) const;

    /// Get the current halt mask
    [[nodiscard]] uint32_t haltMask() const;

    /// Set the backend process ID for signaling
    void setBackendPid(int pid) { m_backend_pid = pid; }

private:
    void signalBackend();

    std::string m_shm_name;
    SharedControlBlock* m_ctrl_block = nullptr;
    int m_backend_pid = 0;

    // Platform-specific handles
#ifdef _WIN32
    void* m_mapping_handle = nullptr;
#else
    int m_shm_fd = -1;
#endif
};

} // namespace algae::hardware
