// ─────────────────────────────────────────────────────────────────────
// StateReconciler — Sequence watermarking for REST ↔ ZMQ gap resolution
//
// Ensures live ZMQ data is applied only AFTER the historical REST
// snapshot is fully loaded, preventing race conditions during startup.
//
// Protocol:
//   1. Initialize ZMQ socket. Buffer incoming ticks without rendering.
//   2. Dispatch QNetworkAccessManager GET to /api/control/state.
//   3. Extract snapshot_sequence_id from the REST response.
//   4. Iterate the ZMQ buffer. Discard any tick where
//      tick.sequence_id <= snapshot_sequence_id. Flush remainder.
//   5. Switch to live mode — all subsequent ticks rendered immediately.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QJsonDocument>

#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>

#include "ZmqReceiver.h"  // IngestMessage full definition needed for std::deque

namespace algae::engine {

/// State reconciliation watermarker that bridges REST bootstrap
/// with live ZMQ streaming, preventing duplicate or stale data.
class StateReconciler : public QObject {
    Q_OBJECT

    Q_PROPERTY(bool reconciled READ isReconciled NOTIFY reconciliationComplete)
    Q_PROPERTY(bool partitioned READ isPartitioned NOTIFY partitionDetected)
    Q_PROPERTY(int pendingBufferSize READ pendingBufferSize NOTIFY bufferSizeChanged)

public:
    using PayloadCallback = std::function<void(const IngestMessage&)>;

    explicit StateReconciler(QObject* parent = nullptr);

    /// Set the callback for flushing reconciled messages to GlobalStore
    void setFlushCallback(PayloadCallback cb) { m_flushCallback = std::move(cb); }

    /// Buffer an incoming ZMQ message during pre-reconciliation phase
    /// Returns true if the message was buffered (still reconciling),
    /// false if we're in live mode (caller should process immediately).
    bool bufferOrPassthrough(const IngestMessage& msg);

    /// Called when the REST bootstrap completes.
    /// Extracts the snapshot_sequence_id and flushes buffered messages
    /// that have sequence_id > snapshot_sequence_id.
    void onBootstrapComplete(uint64_t snapshotSequenceId);

    /// Convenience: parse snapshot_sequence_id from REST JSON response
    void onBootstrapComplete(const QJsonDocument& restResponse);

    bool isReconciled() const { return m_reconciled; }
    bool isPartitioned() const { return m_partitioned; }
    int pendingBufferSize() const;

    /// §3.1: Invalidate state upon network partition detection.
    /// Resets to pre-reconciliation mode, blocks all ZMQ processing
    /// until a fresh REST bootstrap completes.
    void invalidateOnPartition();

Q_SIGNALS:
    void reconciliationComplete();
    void partitionDetected();
    void partitionRecovered();
    void bufferSizeChanged();
    void messagesDropped(int count);

private:
    void flushBuffer(uint64_t watermark);

    mutable std::mutex m_mutex;
    std::deque<IngestMessage> m_buffer;
    PayloadCallback m_flushCallback;
    uint64_t m_snapshotSeqId = 0;
    bool m_reconciled = false;
    bool m_partitioned = false;
};

} // namespace algae::engine
