// ─────────────────────────────────────────────────────────────────────
// ZmqReceiver — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "ZmqReceiver.h"

#include <zmq.hpp>
#include <iostream>

#ifdef __linux__
#include <pthread.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif
#include <bit>
#include <cstring>

namespace algae::engine {

ZmqReceiver::ZmqReceiver(
    const std::string& event_endpoint,
    const std::string& grid_endpoint,
    int cpu_core
)
    : m_event_endpoint(event_endpoint)
    , m_grid_endpoint(grid_endpoint)
    , m_cpu_core(cpu_core)
    , m_curve_enabled(false)
{
    // Check environment toggle for CurveZMQ encryption
    const char* env = std::getenv("ALGAE_ZMQ_CURVE_ENABLED");
    if (env && std::string(env) == "1") {
        m_curve_enabled = true;
    }
}

ZmqReceiver::~ZmqReceiver() {
    stop();
}

void ZmqReceiver::start() {
    if (m_running.load()) return;
    m_running.store(true);
    m_thread = std::jthread([this](std::stop_token st) { run(st); });
}

void ZmqReceiver::stop() {
    if (!m_running.load()) return;
    m_running.store(false);
    m_thread.request_stop();
    if (m_thread.joinable()) {
        m_thread.join();
    }
    // §3.2: CurveZMQ teardown — sockets are closed inside run()
    // when stop_token is triggered. The jthread guarantees join.
}

bool ZmqReceiver::tryDequeue(IngestMessage& out) {
    return m_queue.try_dequeue(out);
}

bool ZmqReceiver::checkAndClearDataLoss() {
    return m_data_loss_flag.exchange(false, std::memory_order_relaxed);
}

void ZmqReceiver::pinToCore(int core) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
#ifdef _WIN32
    DWORD_PTR mask = static_cast<DWORD_PTR>(1) << core;
    SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
}

void ZmqReceiver::run(std::stop_token stoken) {
    pinToCore(m_cpu_core);

    zmq::context_t ctx{1};

    // ── CurveZMQ Encryption (environment-gated) ──────────────────────
    // Generate ephemeral client keypair per-session if encryption enabled.
    // Server public key is bootstrapped via REST /api/control/state.
    char client_public[41] = {}, client_secret[41] = {};
    bool apply_curve = m_curve_enabled && !m_curve_server_key.empty();
    if (apply_curve) {
        zmq_curve_keypair(client_public, client_secret);
    }

    // Event channel subscriber
    zmq::socket_t event_sub{ctx, zmq::socket_type::sub};
    event_sub.set(zmq::sockopt::subscribe, "");
    event_sub.set(zmq::sockopt::rcvhwm, 0);
    event_sub.set(zmq::sockopt::linger, 0);
    event_sub.set(zmq::sockopt::rcvbuf, 16 * 1024 * 1024);  // 16MB OS TCP buffer
    event_sub.set(zmq::sockopt::tcp_keepalive, 1);
    event_sub.set(zmq::sockopt::tcp_keepalive_idle, 30);
    event_sub.set(zmq::sockopt::tcp_keepalive_intvl, 10);
    event_sub.set(zmq::sockopt::reconnect_ivl, 100);
    event_sub.set(zmq::sockopt::reconnect_ivl_max, 5000);
    if (apply_curve) {
        event_sub.set(zmq::sockopt::curve_serverkey, m_curve_server_key);
        event_sub.set(zmq::sockopt::curve_publickey, std::string(client_public));
        event_sub.set(zmq::sockopt::curve_secretkey, std::string(client_secret));
    }
    event_sub.connect(m_event_endpoint);

    // Grid channel subscriber
    zmq::socket_t grid_sub{ctx, zmq::socket_type::sub};
    grid_sub.set(zmq::sockopt::subscribe, "");
    grid_sub.set(zmq::sockopt::rcvhwm, 0);
    grid_sub.set(zmq::sockopt::linger, 0);
    grid_sub.set(zmq::sockopt::rcvbuf, 16 * 1024 * 1024);  // 16MB OS TCP buffer
    grid_sub.set(zmq::sockopt::tcp_keepalive, 1);
    grid_sub.set(zmq::sockopt::tcp_keepalive_idle, 30);
    grid_sub.set(zmq::sockopt::tcp_keepalive_intvl, 10);
    grid_sub.set(zmq::sockopt::reconnect_ivl, 100);
    grid_sub.set(zmq::sockopt::reconnect_ivl_max, 5000);
    if (apply_curve) {
        grid_sub.set(zmq::sockopt::curve_serverkey, m_curve_server_key);
        grid_sub.set(zmq::sockopt::curve_publickey, std::string(client_public));
        grid_sub.set(zmq::sockopt::curve_secretkey, std::string(client_secret));
    }
    grid_sub.connect(m_grid_endpoint);

    // Poll both sockets with 1ms timeout
    zmq::pollitem_t items[] = {
        { event_sub.handle(), 0, ZMQ_POLLIN, 0 },
        { grid_sub.handle(),  0, ZMQ_POLLIN, 0 },
    };

    while (!stoken.stop_requested()) {
        zmq::poll(items, 2, std::chrono::milliseconds{1});

        // ── Event Channel ──────────────────────────────────────────
        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t topic_msg, seq_msg, payload_msg;
            auto r1 = event_sub.recv(topic_msg, zmq::recv_flags::dontwait);
            auto r2 = event_sub.recv(seq_msg, zmq::recv_flags::dontwait);
            auto r3 = event_sub.recv(payload_msg, zmq::recv_flags::dontwait);

            if (r1 && r2 && r3) {
                IngestMessage msg;
                msg.topic.assign(
                    static_cast<const char*>(topic_msg.data()),
                    topic_msg.size()
                );

                // Fast-Path Sequence Extraction (8-byte prefix)
                if (seq_msg.size() < 8) continue;
                
                uint64_t seq_be;
                std::memcpy(&seq_be, seq_msg.data(), sizeof(uint64_t));
                
                // C++23 endianness check and byteswap
                uint64_t seq_id = std::endian::native == std::endian::little 
                                  ? std::byteswap(seq_be) : seq_be;

                if (seq_id <= m_watermark.load(std::memory_order_relaxed)) continue;
                msg.sequence_id = seq_id;

                msg.payload.assign(
                    static_cast<const uint8_t*>(payload_msg.data()),
                    static_cast<const uint8_t*>(payload_msg.data()) + payload_msg.size()
                );

                // 3. Drop-Head Backpressure Execution
                if (!m_queue.try_enqueue(std::move(msg))) {
                    IngestMessage dropped;
                    m_queue.try_dequeue(dropped);    // Eject oldest tick
                    m_queue.try_enqueue(std::move(msg)); // Insert newest tick
                    
                    // Signal UI via atomic flag (memory_order_release ensures visibility)
                    m_data_loss_flag.store(true, std::memory_order_release);
                }
            }
        }

        // ── Grid Channel (Arrow IPC) ───────────────────────────────
        if (items[1].revents & ZMQ_POLLIN) {
            zmq::message_t topic_msg, seq_msg, payload_msg;
            auto r1 = grid_sub.recv(topic_msg, zmq::recv_flags::dontwait);
            auto r2 = grid_sub.recv(seq_msg, zmq::recv_flags::dontwait);
            auto r3 = grid_sub.recv(payload_msg, zmq::recv_flags::dontwait);

            if (r1 && r2 && r3) {
                IngestMessage msg;
                msg.topic = "grid." + std::string(
                    static_cast<const char*>(topic_msg.data()),
                    topic_msg.size()
                );

                // Fast-Path Sequence Extraction (8-byte prefix)
                if (seq_msg.size() < 8) continue;
                
                uint64_t seq_be;
                std::memcpy(&seq_be, seq_msg.data(), sizeof(uint64_t));
                
                // C++23 endianness check and byteswap
                uint64_t seq_id = std::endian::native == std::endian::little 
                                  ? std::byteswap(seq_be) : seq_be;

                if (seq_id <= m_watermark.load(std::memory_order_relaxed)) continue;
                msg.sequence_id = seq_id;

                msg.payload.assign(
                    static_cast<const uint8_t*>(payload_msg.data()),
                    static_cast<const uint8_t*>(payload_msg.data()) + payload_msg.size()
                );

                // 3. Drop-Head Backpressure Execution
                if (!m_queue.try_enqueue(std::move(msg))) {
                    IngestMessage dropped;
                    m_queue.try_dequeue(dropped);
                    m_queue.try_enqueue(std::move(msg));
                    m_data_loss_flag.store(true, std::memory_order_release);
                }
            }
        }
    }

    // §3.2: Deterministic CurveZMQ socket teardown.
    // Close sockets explicitly before zmq::context_t destructor
    // to prevent ETERM deadlocks during active Curve25519 handshakes.
    event_sub.set(zmq::sockopt::linger, 0);
    grid_sub.set(zmq::sockopt::linger, 0);
    event_sub.close();
    grid_sub.close();
}

} // namespace algae::engine
