// ─────────────────────────────────────────────────────────────────────
// KillSwitch — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "KillSwitch.h"

#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace algae::hardware {

KillSwitch::KillSwitch(const std::string& shm_name)
    : m_shm_name(shm_name)
{
}

KillSwitch::~KillSwitch() {
#ifdef _WIN32
    if (m_ctrl_block) {
        UnmapViewOfFile(m_ctrl_block);
        m_ctrl_block = nullptr;
    }
    if (m_mapping_handle) {
        CloseHandle(m_mapping_handle);
        m_mapping_handle = nullptr;
    }
#else
    if (m_ctrl_block && m_ctrl_block != MAP_FAILED) {
        munmap(m_ctrl_block, sizeof(SharedControlBlock));
        m_ctrl_block = nullptr;
    }
    if (m_shm_fd >= 0) {
        close(m_shm_fd);
        m_shm_fd = -1;
    }
#endif
}

bool KillSwitch::initialize() {
#ifdef _WIN32
    // Windows: CreateFileMapping with a named section
    m_mapping_handle = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        nullptr,
        PAGE_READWRITE,
        0,
        sizeof(SharedControlBlock),
        m_shm_name.c_str()
    );
    if (!m_mapping_handle) {
        std::cerr << "KillSwitch: CreateFileMapping failed: " << GetLastError() << "\n";
        return false;
    }

    m_ctrl_block = static_cast<SharedControlBlock*>(
        MapViewOfFile(m_mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedControlBlock))
    );
    if (!m_ctrl_block) {
        std::cerr << "KillSwitch: MapViewOfFile failed: " << GetLastError() << "\n";
        CloseHandle(m_mapping_handle);
        m_mapping_handle = nullptr;
        return false;
    }
#else
    // POSIX: shm_open + mmap
    m_shm_fd = shm_open(m_shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (m_shm_fd < 0) {
        perror("KillSwitch: shm_open failed");
        return false;
    }

    if (ftruncate(m_shm_fd, sizeof(SharedControlBlock)) < 0) {
        perror("KillSwitch: ftruncate failed");
        close(m_shm_fd);
        m_shm_fd = -1;
        return false;
    }

    void* addr = mmap(
        nullptr, sizeof(SharedControlBlock),
        PROT_READ | PROT_WRITE, MAP_SHARED,
        m_shm_fd, 0
    );
    if (addr == MAP_FAILED) {
        perror("KillSwitch: mmap failed");
        close(m_shm_fd);
        m_shm_fd = -1;
        return false;
    }

    m_ctrl_block = static_cast<SharedControlBlock*>(addr);
#endif

    // Zero-initialize the block
    m_ctrl_block->sequence_id.store(0, std::memory_order_relaxed);
    m_ctrl_block->halt_mask.store(0, std::memory_order_relaxed);
    std::memset(m_ctrl_block->reason, 0, sizeof(m_ctrl_block->reason));

    return true;
}

void KillSwitch::haltSleeve(uint32_t sleeve_id, const std::string& reason) {
    if (!m_ctrl_block) return;

    // 1. Write reason string
    std::memset(m_ctrl_block->reason, 0, sizeof(m_ctrl_block->reason));
    std::strncpy(m_ctrl_block->reason, reason.c_str(), 63);

    // 2. Set halt bit atomically
    m_ctrl_block->halt_mask.fetch_or(1u << sleeve_id, std::memory_order_release);

    // 3. Increment sequence (triggers backend check)
    m_ctrl_block->sequence_id.fetch_add(1, std::memory_order_release);

    // 4. Signal backend process
    signalBackend();
}

void KillSwitch::resumeSleeve(uint32_t sleeve_id) {
    if (!m_ctrl_block) return;

    m_ctrl_block->halt_mask.fetch_and(~(1u << sleeve_id), std::memory_order_release);
    m_ctrl_block->sequence_id.fetch_add(1, std::memory_order_release);
    signalBackend();
}

void KillSwitch::resumeAll() {
    if (!m_ctrl_block) return;

    m_ctrl_block->halt_mask.store(0, std::memory_order_release);
    std::memset(m_ctrl_block->reason, 0, sizeof(m_ctrl_block->reason));
    m_ctrl_block->sequence_id.fetch_add(1, std::memory_order_release);
    signalBackend();
}

bool KillSwitch::isSleeveHalted(uint32_t sleeve_id) const {
    if (!m_ctrl_block) return false;
    return (m_ctrl_block->halt_mask.load(std::memory_order_acquire) & (1u << sleeve_id)) != 0;
}

uint32_t KillSwitch::haltMask() const {
    if (!m_ctrl_block) return 0;
    return m_ctrl_block->halt_mask.load(std::memory_order_acquire);
}

void KillSwitch::signalBackend() {
    if (m_backend_pid <= 0) return;

#ifdef _WIN32
    // Windows: Use a named event to signal the backend
    // The backend creates an event named "AlgaeKillSwitchEvent"
    HANDLE hEvent = OpenEventA(EVENT_MODIFY_STATE, FALSE, "AlgaeKillSwitchEvent");
    if (hEvent) {
        SetEvent(hEvent);
        CloseHandle(hEvent);
    }
#else
    // POSIX: SIGUSR1 directly to backend PID
    kill(m_backend_pid, SIGUSR1);
#endif
}

} // namespace algae::hardware
