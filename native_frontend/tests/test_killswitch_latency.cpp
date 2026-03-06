// ─────────────────────────────────────────────────────────────────────
// test_killswitch_latency.cpp — Microsecond latency validation suite
//
// Validates the OOB kill switch SLA (<50μs) by:
//   1. Basic functional tests (halt, resume, shared state)
//   2. Profiled shared memory write latency (10K iterations)
//   3. On Linux: fork()-based IPC proof with SIGUSR1 context switch
//
// The fork() test spawns a child process simulating the Python backend,
// measures the round-trip time of shm write + POSIX signal delivery,
// and asserts the total duration is < 50μs.
// ─────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "hardware/KillSwitch.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#ifdef __linux__
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif

using algae::hardware::KillSwitch;
using algae::hardware::SharedControlBlock;

TEST(KillSwitch, InitializesClean) {
    KillSwitch ks("AlgaeTestCtrl_Init");
    ASSERT_TRUE(ks.initialize());
    
    EXPECT_EQ(ks.haltMask(), 0u);
    EXPECT_FALSE(ks.isSleeveHalted(0));
    EXPECT_FALSE(ks.isSleeveHalted(4));
}

TEST(KillSwitch, HaltAndResumeSleeveCorrectly) {
    KillSwitch ks("AlgaeTestCtrl_Halt");
    ASSERT_TRUE(ks.initialize());
    
    // Halt sleeve 3
    ks.haltSleeve(3, "Test halt");
    
    EXPECT_TRUE(ks.isSleeveHalted(3));
    EXPECT_FALSE(ks.isSleeveHalted(0));
    EXPECT_FALSE(ks.isSleeveHalted(4));
    EXPECT_EQ(ks.haltMask(), 1u << 3);
    
    // Halt sleeve 5 as well
    ks.haltSleeve(5, "Another halt");
    
    EXPECT_TRUE(ks.isSleeveHalted(3));
    EXPECT_TRUE(ks.isSleeveHalted(5));
    EXPECT_EQ(ks.haltMask(), (1u << 3) | (1u << 5));
    
    // Resume sleeve 3
    ks.resumeSleeve(3);
    
    EXPECT_FALSE(ks.isSleeveHalted(3));
    EXPECT_TRUE(ks.isSleeveHalted(5));
}

TEST(KillSwitch, ResumeAllClearsEverything) {
    KillSwitch ks("AlgaeTestCtrl_ResumeAll");
    ASSERT_TRUE(ks.initialize());
    
    ks.haltSleeve(0, "halt 0");
    ks.haltSleeve(2, "halt 2");
    ks.haltSleeve(4, "halt 4");
    
    EXPECT_EQ(ks.haltMask(), (1u << 0) | (1u << 2) | (1u << 4));
    
    ks.resumeAll();
    
    EXPECT_EQ(ks.haltMask(), 0u);
    EXPECT_FALSE(ks.isSleeveHalted(0));
    EXPECT_FALSE(ks.isSleeveHalted(2));
    EXPECT_FALSE(ks.isSleeveHalted(4));
}

TEST(KillSwitch, SharedMemoryWriteLatency) {
    KillSwitch ks("AlgaeTestCtrl_Latency");
    ASSERT_TRUE(ks.initialize());

    // Warm up the cache line
    ks.haltSleeve(0, "warmup");
    ks.resumeAll();

    // Measure write latency over 10K iterations
    constexpr int iterations = 10'000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        ks.haltSleeve(4, "latency_test");
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double avgNs = static_cast<double>(elapsed.count()) / iterations;

    // MATHEMATICAL PROOF: Shared memory atomic write must be << 1μs
    // Allow up to 50μs to account for CI variability
    EXPECT_LT(avgNs, 50'000.0)
        << "Average kill switch write latency: " << avgNs << " ns"
        << " (" << (avgNs / 1000.0) << " μs)";

    std::cout << "[LATENCY] Kill switch write: " << avgNs << " ns"
              << " (" << (avgNs / 1000.0) << " μs)" << std::endl;
}

TEST(KillSwitch, TwoInstancesSeeSharedState) {
    // Simulates the C++ UI writing and the Python backend reading
    const char* shmName = "AlgaeTestCtrl_Shared";

    KillSwitch writer(shmName);
    KillSwitch reader(shmName);
    
    ASSERT_TRUE(writer.initialize());
    ASSERT_TRUE(reader.initialize());
    
    // Writer halts sleeve 2
    writer.haltSleeve(2, "shared_test");

    // Reader should see it (same shared memory)
    EXPECT_TRUE(reader.isSleeveHalted(2));
    EXPECT_EQ(reader.haltMask(), 1u << 2);
    
    // Writer resumes
    writer.resumeAll();
    EXPECT_FALSE(reader.isSleeveHalted(2));
}

#ifdef __linux__
// ── Fork-Based IPC Latency Proof (Linux Only) ────────────────────
// Spawns a child process simulating the Python backend listener.
// Measures the total round-trip time: shm write + SIGUSR1 delivery
// + child spin-wait ack. Must complete in < 50μs.

static std::atomic<bool> g_signal_received{false};
static void handle_sigusr1(int) { g_signal_received.store(true, std::memory_order_release); }

TEST(KillSwitch, ForkBasedMicrosecondLatencyProof) {
    const char* shmName = "AlgaeTestCtrl_Fork";
    
    // Clean up any previous shared memory
    KillSwitch cleanup(shmName);
    cleanup.initialize();
    cleanup.resumeAll();

    pid_t pid = fork();

    if (pid == 0) {
        // --- CHILD PROCESS (Simulating Python Backend) ---
        std::signal(SIGUSR1, handle_sigusr1);
        
        KillSwitch child_ks(shmName);
        child_ks.initialize();
        
        // Spin-wait for the hardware interrupt
        while (!g_signal_received.load(std::memory_order_acquire)) {}
        
        // Verify the memory mapped value was updated correctly
        // 16 = 0b00010000 (Sleeve 4)
        if (child_ks.isSleeveHalted(4)) {
            _exit(0);
        }
        _exit(1); 
    } 
    else {
        ASSERT_GT(pid, 0) << "fork() failed";

        // --- PARENT PROCESS (Simulating C++ Native UI) ---
        KillSwitch parent_ks(shmName);
        ASSERT_TRUE(parent_ks.initialize());
        
        // Allow child process to register signal handlers
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Fire OOB Kill Command
        parent_ks.haltSleeve(4, "Fork Latency SLA Test");
        
        // Send SIGUSR1 to child
        kill(pid, SIGUSR1);

        // 2. Wait for child to exit successfully
        int status;
        waitpid(pid, &status, 0);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();

        EXPECT_TRUE(WIFEXITED(status));
        EXPECT_EQ(WEXITSTATUS(status), 0);
        
        // 3. MATHEMATICAL PROOF: Total context switch time must be under 50μs
        EXPECT_LT(duration_us, 50) 
            << "Fork IPC round-trip: " << duration_us << " μs";
        
        std::cout << "[LATENCY] Fork IPC round-trip: " << duration_us 
                  << " μs" << std::endl;
    }
}
#endif
