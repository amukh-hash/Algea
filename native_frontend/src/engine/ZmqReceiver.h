// ─────────────────────────────────────────────────────────────────────
// ZmqReceiver — High-frequency ingestion thread
//
// Runs on a pinned CPU core, receives ZMQ multipart messages, 
// deserializes Protobuf, and enqueues into SPSC bounded ring buffer
// with Drop-Head backpressure policy.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <functional>

#include <readerwriterqueue/readerwriterqueue.h>

namespace algae::engine {

/// Deserialized message wrapper for the SPSC queue
struct IngestMessage {
    uint64_t sequence_id = 0;
    std::string topic;           // e.g. "telemetry.metric", "control.snapshot"
    std::vector<uint8_t> payload; // Raw Protobuf bytes (deferred deserialization)
};

/// ZeroMQ ingestion thread with bounded SPSC queue and CPU affinity pinning.
///
/// The consumer (UiSynchronizer) drains the queue at 60 Hz via QTimer.
/// If the queue fills during a market volatility spike, the oldest message
/// is dropped (Drop-Head policy) and m_data_loss_flag is set atomically.
class ZmqReceiver {
public:
    static constexpr size_t QUEUE_CAPACITY = 10'000;

    explicit ZmqReceiver(
        const std::string& event_endpoint = "tcp://127.0.0.1:5556",
        const std::string& grid_endpoint  = "tcp://127.0.0.1:5557",
        int cpu_core = 2
    );
    ~ZmqReceiver();

    // Non-copyable, non-movable
    ZmqReceiver(const ZmqReceiver&) = delete;
    ZmqReceiver& operator=(const ZmqReceiver&) = delete;

    /// Start the ingestion thread
    void start();

    /// Request stop and join the thread
    void stop();

    /// Try to dequeue a message (called by UiSynchronizer at 60 Hz)
    [[nodiscard]] bool tryDequeue(IngestMessage& out);

    /// Check and clear the data loss flag
    [[nodiscard]] bool checkAndClearDataLoss();

    /// True if the ingestion thread is running
    [[nodiscard]] bool isRunning() const { return m_running.load(std::memory_order_relaxed); }

    /// Current queue size (approximate)
    [[nodiscard]] size_t queueSize() const { return m_queue.size_approx(); }

    /// Set minimum sequence ID to process (anything older dropped immediately)
    void setWatermark(uint64_t w) { m_watermark.store(w, std::memory_order_relaxed); }

    /// Set CurveZMQ server public key (bootstrapped from REST /api/control/state)
    void setCurveServerKey(const std::string& key) { m_curve_server_key = key; }

private:
    void run(std::stop_token stoken);
    void pinToCore(int core);

    std::string m_event_endpoint;
    std::string m_grid_endpoint;
    int m_cpu_core;

    moodycamel::ReaderWriterQueue<IngestMessage> m_queue{QUEUE_CAPACITY};
    std::atomic<bool> m_data_loss_flag{false};
    std::atomic<bool> m_running{false};
    std::atomic<uint64_t> m_watermark{0};
    bool m_curve_enabled{false};
    std::string m_curve_server_key;
    std::jthread m_thread;
    int m_failed_handshakes{0};
    static constexpr int MAX_HANDSHAKE_FAILURES = 3;
};

} // namespace algae::engine
