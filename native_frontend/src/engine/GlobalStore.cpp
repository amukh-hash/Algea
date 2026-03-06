// ─────────────────────────────────────────────────────────────────────
// GlobalStore — Implementation (RCU)
//
// Routes deserialized payloads from the ingestion queue. Mutations
// are applied via commitStateMutation() which performs an atomic
// pointer swap of the ApplicationState struct, eliminating tearing.
// The QML V4 engine only sees NOTIFY signals when commitFrameUpdates()
// fires at 60Hz.
// ─────────────────────────────────────────────────────────────────────
#include "GlobalStore.h"
#include "ZmqReceiver.h"
#include "../models/ArrowBuilderWorker.h"

#include <QDebug>

// Generated Protobuf headers (from CMake protobuf_generate_cpp)
#include "telemetry.pb.h"
#include "control.pb.h"
#include "portfolio.pb.h"

namespace algae::engine {

GlobalStore::GlobalStore(QObject* parent)
    : QObject(parent)
{
}

void GlobalStore::routePayload(const IngestMessage& msg) {
    // Update sequence watermark via RCU
    commitStateMutation([&](ApplicationState& s) {
        if (msg.sequence_id > s.last_sequence_id) {
            s.last_sequence_id = msg.sequence_id;
        }
    });

    // Route by topic prefix — uses static_cast to generated C++ class
    // NEVER Protobuf Reflection (Blind Spot 3 mitigation)
    if (msg.topic.starts_with("telemetry.")) {
        handleTelemetryEvent(msg);
    } else if (msg.topic.starts_with("control.")) {
        handleControlSnapshot(msg);
    } else if (msg.topic.starts_with("grid.")) {
        handleGridData(msg);
    } else {
        qDebug() << "Unknown topic:" << QString::fromStdString(msg.topic);
    }
}

// ── Handlers ───────────────────────────────────────────────────────

void GlobalStore::handleTelemetryEvent(const IngestMessage& msg) {
    ::algae::streaming::StreamEnvelope envelope;
    if (!envelope.ParseFromArray(msg.payload.data(), 
                                  static_cast<int>(msg.payload.size()))) {
        qWarning() << "Failed to parse StreamEnvelope from topic:" 
                    << QString::fromStdString(msg.topic);
        return;
    }

    switch (envelope.payload_case()) {
        case ::algae::streaming::StreamEnvelope::kTick: {
            const auto& tick = envelope.tick();
            Q_EMIT tickReceived(
                QString::fromStdString(tick.symbol()),
                tick.bid(),
                tick.ask()
            );
            break;
        }

        case ::algae::streaming::StreamEnvelope::kMetric: {
            const auto& metric = envelope.metric();
            const auto& key = metric.key();
            if (key == "stat_arb_correlation") {
                // Silent atomic update — no signal emission here
                silentUpdateStatArbCorr(metric.value());
            }
            break;
        }

        case ::algae::streaming::StreamEnvelope::kAlert: {
            const auto& alert = envelope.alert();
            bool isInhibited = !alert.root_cause_id().empty();
            Q_EMIT alertReceived(
                QString::fromStdString(alert.id()),
                isInhibited
            );
            break;
        }

        default:
            break;
    }
}

void GlobalStore::handleControlSnapshot(const IngestMessage& msg) {
    ::algae::control::ControlSnapshot snapshot;
    if (!snapshot.ParseFromArray(msg.payload.data(),
                                  static_cast<int>(msg.payload.size()))) {
        qWarning() << "Failed to parse ControlSnapshot";
        return;
    }

    // Batch all control state mutations into a single RCU swap
    commitStateMutation([&](ApplicationState& s) {
        if (snapshot.has_state()) {
            const auto& state = snapshot.state();
            s.system_paused = state.paused();
            s.execution_mode = QString::fromStdString(state.execution_mode());
            s.vol_regime_override = QString::fromStdString(state.vol_regime_override());
        }
        if (snapshot.has_broker()) {
            s.broker_connected = snapshot.broker().connected();
        }
        if (snapshot.has_calendar()) {
            s.current_session = QString::fromStdString(snapshot.calendar().current_session());
        }
    });

    // Signals are NOT emitted here — they fire via commitFrameUpdates()
    // at 60Hz in the UiSynchronizer, preventing V4 engine stalls.
}

void GlobalStore::handleGridData(const IngestMessage& msg) {
    // Route Arrow IPC payload to ArrowBuilderWorker for schema-hardened
    // deserialization and scene graph hydration.
    // Topic examples: "grid.positions", "chart.kronos_fan", "chart.sankey_alloc"
    if (msg.payload.empty()) return;

    if (m_arrow_worker) {
        m_arrow_worker->processIpcPayload(
            msg.topic,
            msg.payload.data(),
            msg.payload.size()
        );
    }
}

} // namespace algae::engine
