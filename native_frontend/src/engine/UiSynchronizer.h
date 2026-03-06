// ─────────────────────────────────────────────────────────────────────
// UiSynchronizer — Frame-paced queue drain via QTimer
//
// Decouples the ZMQ ingestion thread from the Qt main thread.
// Drains the SPSC queue at 60 Hz (16ms) using QTimer, batching all
// accumulated messages per frame into GlobalStore.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QTimer>

namespace algae::engine {

class ZmqReceiver;
class GlobalStore;
class StateReconciler;

/// Frame-paced synchronizer that drains the ingestion queue at ~60 FPS.
///
/// Connected to the Qt Event Loop via QTimer — NOT busy-polling.
/// Each timer tick drains all accumulated messages and routes them
/// through GlobalStore for deserialization and state update.
class UiSynchronizer : public QObject {
    Q_OBJECT

public:
    /// @param receiver     The ZMQ ingestion thread (producer)
    /// @param store        The global state store (consumer)
    /// @param reconciler   The watermark engine for REST↔ZMQ consistency
    /// @param parent       Qt parent object
    explicit UiSynchronizer(
        ZmqReceiver* receiver,
        GlobalStore* store,
        StateReconciler* reconciler = nullptr,
        QObject* parent = nullptr
    );

    ~UiSynchronizer() override = default;

    /// Start the frame timer
    void start();

    /// Stop the frame timer
    void stop();

Q_SIGNALS:
    /// Emitted when the data loss flag is detected
    void dataLossDetected();

    /// Emitted each frame with the count of messages drained
    void frameDrained(int messageCount);

private Q_SLOTS:
    void drainQueue();

private:
    ZmqReceiver* m_receiver;
    GlobalStore* m_store;
    StateReconciler* m_reconciler;
    QTimer* m_timer;
};

} // namespace algae::engine
