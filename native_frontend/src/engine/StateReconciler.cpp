// ─────────────────────────────────────────────────────────────────────
// StateReconciler — Implementation
//
// Sequence watermark engine utilizing C++20 <ranges> for deterministic
// filtering of stale ZMQ ticks against the REST bootstrap watermark.
// ─────────────────────────────────────────────────────────────────────
#include "StateReconciler.h"
#include "ZmqReceiver.h"
#include "GlobalStore.h"

#include <QDebug>
#include <QJsonObject>
#include <ranges>
#include <algorithm>

namespace algae::engine {

StateReconciler::StateReconciler(QObject* parent)
    : QObject(parent)
{
}

bool StateReconciler::bufferOrPassthrough(const IngestMessage& msg) {
    std::lock_guard lock(m_mutex);

    // §3.1: During network partition, drop all ZMQ messages silently.
    // The UI is displaying the NETWORK PARTITION modal; accepting
    // stale or partial ticks would corrupt the displayed state.
    if (m_partitioned) {
        return true; // Buffered = true (caller should NOT process)
    }

    if (m_reconciled) {
        // Live mode — fast-path execution, caller processes immediately
        return false;
    }

    // Hold ticks in buffer while REST bootstrap is inflight
    m_buffer.push_back(msg);
    Q_EMIT bufferSizeChanged();
    return true;
}

void StateReconciler::onBootstrapComplete(uint64_t snapshotSequenceId) {
    std::lock_guard lock(m_mutex);

    m_snapshotSeqId = snapshotSequenceId;

    // §3.1: If recovering from partition, log the recovery event
    if (m_partitioned) {
        qInfo() << "StateReconciler: PARTITION RECOVERED — re-bootstrapping from seq_id ="
                << m_snapshotSeqId;
        m_partitioned = false;
        Q_EMIT partitionRecovered();
    }

    qInfo() << "StateReconciler: bootstrap complete, watermark seq_id ="
            << m_snapshotSeqId << ", buffer size =" << m_buffer.size();

    flushBuffer(snapshotSequenceId);

    m_reconciled = true;

    // Open the ZMQ fast-path lock-free
    Q_EMIT reconciliationComplete();
}

void StateReconciler::onBootstrapComplete(const QJsonDocument& restResponse) {
    uint64_t seqId = 0;

    if (restResponse.isObject()) {
        auto obj = restResponse.object();
        if (obj.contains("snapshot_sequence_id")) {
            seqId = static_cast<uint64_t>(obj["snapshot_sequence_id"].toDouble());
        }
        else if (obj.contains("state") && obj["state"].isObject()) {
            auto state = obj["state"].toObject();
            if (state.contains("sequence_id")) {
                seqId = static_cast<uint64_t>(state["sequence_id"].toDouble());
            }
        }
        else if (obj.contains("sequence_id")) {
            seqId = static_cast<uint64_t>(obj["sequence_id"].toDouble());
        }
    }

    onBootstrapComplete(seqId);
}

int StateReconciler::pendingBufferSize() const {
    std::lock_guard lock(m_mutex);
    return static_cast<int>(m_buffer.size());
}

void StateReconciler::invalidateOnPartition() {
    std::lock_guard lock(m_mutex);

    if (m_partitioned) return; // Already in partition state

    qWarning() << "StateReconciler: NETWORK PARTITION DETECTED — "
               << "invalidating all state, blocking ZMQ processing until "
               << "fresh REST bootstrap completes";

    m_partitioned = true;
    m_reconciled = false;

    // Purge all buffered ticks — they are from before the partition
    // and applying them on reconnection would corrupt the grid
    m_buffer.clear();
    m_buffer.shrink_to_fit();

    Q_EMIT partitionDetected();
    Q_EMIT bufferSizeChanged();
}

void StateReconciler::flushBuffer(uint64_t watermark) {
    // Must be called under m_mutex lock

    int dropped = 0;
    int flushed = 0;

    // C++20 Ranges: Filter out stale ticks from the buffer that were
    // included in the REST snapshot. Only ticks with sequence_id > watermark
    // are forwarded to GlobalStore.
    auto valid_ticks = m_buffer | std::views::filter([watermark](const auto& msg) {
        return msg.sequence_id > watermark;
    });

    for (const auto& msg : valid_ticks) {
        if (m_flushCallback) {
            m_flushCallback(msg);
        }
        ++flushed;
    }

    dropped = static_cast<int>(m_buffer.size()) - flushed;

    if (dropped > 0) {
        qInfo() << "StateReconciler: dropped" << dropped
                << "stale messages (seq_id <=" << watermark << ")";
        Q_EMIT messagesDropped(dropped);
    }

    qInfo() << "StateReconciler: flushed" << flushed << "messages to live state";

    m_buffer.clear();
    m_buffer.shrink_to_fit();

    Q_EMIT bufferSizeChanged();
}

} // namespace algae::engine
