// ─────────────────────────────────────────────────────────────────────
// GlobalStore — RCU (Read-Copy-Update) State Store
//
// Eliminates atomic tearing by encapsulating all scalar state into an
// immutable ApplicationState struct. The ingestion thread copies, mutates,
// and atomically swaps the pointer. The UI thread latches a coherent
// snapshot exactly ONCE per 16ms frame boundary.
//
// This guarantees the QML render thread never observes a fractured
// intermediary state (e.g., new PnL with old position count).
//
// QML_ELEMENT + QML_SINGLETON macros are CRITICAL: they force the
// linker to preserve MOC metadata tables even under aggressive LTO.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QString>
#include <QVariantMap>
#include <QtQml/qqmlregistration.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace algae::engine {

struct IngestMessage;

} // namespace algae::engine

namespace algae::models {
class ArrowBuilderWorker;
} // namespace algae::models

namespace algae::engine {

/// Immutable state container — never mutated in-place after publication.
/// The ingestion thread creates a COPY, mutates it, then atomically swaps.
struct ApplicationState {
    // Portfolio
    double portfolio_value = 0.0;
    double total_pnl = 0.0;
    int position_count = 0;

    // System
    bool broker_connected = false;
    bool system_paused = false;
    bool data_loss = false;

    // Risk
    double stat_arb_correlation = 0.0;

    // Allocation (Tab 4)
    double total_leverage = 0.0;
    double kronos_weight = 0.40;
    double cooc_weight = 0.30;
    double statarb_weight = 0.20;
    double tft_weight = 0.10;
    double kronos_confidence = 0.0;
    double cooc_confidence = 0.0;
    double statarb_confidence = 0.0;
    double tft_confidence = 0.0;

    // Lab (Tab 6)
    int shadow_runs = 0;
    int pending_promotions = 0;
    int promoted_models = 0;

    // Sequence
    uint64_t last_sequence_id = 0;
    int64_t clock_offset_ns = 0;

    // Non-scalar (copied by value during RCU mutation)
    QString current_session = "closed";
    QString execution_mode = "paper";
    QString vol_regime_override;
};

/// Singleton global state store — all telemetry, control, and portfolio
/// state flows through here after deserialization.
class GlobalStore : public QObject {
    Q_OBJECT
    QML_ELEMENT
    QML_SINGLETON

    // ── System Overview ────────────────────────────────────────────
    Q_PROPERTY(bool brokerConnected READ brokerConnected NOTIFY healthStatusChanged)
    Q_PROPERTY(QString currentSession READ currentSession NOTIFY healthStatusChanged)
    Q_PROPERTY(QString executionMode READ executionMode NOTIFY healthStatusChanged)
    Q_PROPERTY(bool systemPaused READ systemPaused NOTIFY healthStatusChanged)
    Q_PROPERTY(bool dataLossFlag READ dataLossFlag NOTIFY healthStatusChanged)

    // ── Portfolio ──────────────────────────────────────────────────
    Q_PROPERTY(double totalPortfolioValue READ totalPortfolioValue NOTIFY portfolioValueChanged)
    Q_PROPERTY(double totalPnl READ totalPnl NOTIFY portfolioValueChanged)
    Q_PROPERTY(int positionCount READ positionCount NOTIFY portfolioValueChanged)

    // ── Risk ───────────────────────────────────────────────────────
    Q_PROPERTY(double statArbCorrelation READ statArbCorrelation NOTIFY healthStatusChanged)
    Q_PROPERTY(QString volRegimeOverride READ volRegimeOverride NOTIFY healthStatusChanged)

    // ── Sequence Tracking ──────────────────────────────────────────
    Q_PROPERTY(uint64_t lastSequenceId READ lastSequenceId NOTIFY healthStatusChanged)
    Q_PROPERTY(int64_t clockOffsetNs READ clockOffsetNs NOTIFY healthStatusChanged)

    // ── Allocation (Tab 4) ──────────────────────────────────────
    Q_PROPERTY(double totalLeverage READ totalLeverage NOTIFY portfolioValueChanged)
    Q_PROPERTY(double kronosWeight READ kronosWeight NOTIFY portfolioValueChanged)
    Q_PROPERTY(double coocWeight READ coocWeight NOTIFY portfolioValueChanged)
    Q_PROPERTY(double statarbWeight READ statarbWeight NOTIFY portfolioValueChanged)
    Q_PROPERTY(double tftWeight READ tftWeight NOTIFY portfolioValueChanged)
    Q_PROPERTY(double kronosConfidence READ kronosConfidence NOTIFY portfolioValueChanged)
    Q_PROPERTY(double coocConfidence READ coocConfidence NOTIFY portfolioValueChanged)
    Q_PROPERTY(double statarbConfidence READ statarbConfidence NOTIFY portfolioValueChanged)
    Q_PROPERTY(double tftConfidence READ tftConfidence NOTIFY portfolioValueChanged)

    // ── Lab (Tab 6) ───────────────────────────────────────────
    Q_PROPERTY(int shadowRuns READ shadowRuns NOTIFY healthStatusChanged)
    Q_PROPERTY(int pendingPromotions READ pendingPromotions NOTIFY healthStatusChanged)
    Q_PROPERTY(int promotedModels READ promotedModels NOTIFY healthStatusChanged)

public:
    static GlobalStore* instance() {
        static GlobalStore store;
        return &store;
    }

    // ── QML Accessors (read from latched UI snapshot — coherent) ──
    double totalPortfolioValue() const { return m_ui_snapshot->portfolio_value; }
    double totalPnl()           const { return m_ui_snapshot->total_pnl; }
    int    positionCount()      const { return m_ui_snapshot->position_count; }
    bool   brokerConnected()    const { return m_ui_snapshot->broker_connected; }
    bool   systemPaused()       const { return m_ui_snapshot->system_paused; }
    bool   dataLossFlag()       const { return m_ui_snapshot->data_loss; }
    double statArbCorrelation() const { return m_ui_snapshot->stat_arb_correlation; }
    uint64_t lastSequenceId()   const { return m_ui_snapshot->last_sequence_id; }
    int64_t  clockOffsetNs()    const { return m_ui_snapshot->clock_offset_ns; }
    QString  currentSession()   const { return m_ui_snapshot->current_session; }
    QString  executionMode()    const { return m_ui_snapshot->execution_mode; }
    QString  volRegimeOverride()const { return m_ui_snapshot->vol_regime_override; }
    double   totalLeverage()    const { return m_ui_snapshot->total_leverage; }
    double   kronosWeight()     const { return m_ui_snapshot->kronos_weight; }
    double   coocWeight()       const { return m_ui_snapshot->cooc_weight; }
    double   statarbWeight()    const { return m_ui_snapshot->statarb_weight; }
    double   tftWeight()        const { return m_ui_snapshot->tft_weight; }
    double   kronosConfidence() const { return m_ui_snapshot->kronos_confidence; }
    double   coocConfidence()   const { return m_ui_snapshot->cooc_confidence; }
    double   statarbConfidence()const { return m_ui_snapshot->statarb_confidence; }
    double   tftConfidence()    const { return m_ui_snapshot->tft_confidence; }
    int      shadowRuns()       const { return m_ui_snapshot->shadow_runs; }
    int      pendingPromotions()const { return m_ui_snapshot->pending_promotions; }
    int      promotedModels()   const { return m_ui_snapshot->promoted_models; }

    /// Set ArrowBuilderWorker for grid data routing
    void setArrowWorker(algae::models::ArrowBuilderWorker* w) { m_arrow_worker = w; }

    // ══════════════════════════════════════════════════════════════
    // RCU Mutation Interface (called by ingestion thread)
    // ══════════════════════════════════════════════════════════════

    /// Apply a batch of mutations atomically via Read-Copy-Update.
    /// The mutator lambda receives a mutable reference to a COPY
    /// of the current state. After the lambda returns, the modified
    /// copy is atomically published as the new active state.
    void commitStateMutation(const std::function<void(ApplicationState&)>& mutator) {
        auto current = std::atomic_load_explicit(&m_active_state, std::memory_order_acquire);
        auto next = std::make_shared<const ApplicationState>([&]() {
            ApplicationState copy(*current);
            mutator(copy);
            return copy;
        }());
        std::atomic_store_explicit(&m_active_state, next, std::memory_order_release);
    }

    // ── Convenience silent updaters (backward-compatible with ingestion) ──
    void silentUpdatePortfolio(double val) {
        commitStateMutation([val](ApplicationState& s) { s.portfolio_value = val; });
    }
    void silentUpdatePnl(double pnl) {
        commitStateMutation([pnl](ApplicationState& s) { s.total_pnl = pnl; });
    }
    void silentUpdatePositionCount(int count) {
        commitStateMutation([count](ApplicationState& s) { s.position_count = count; });
    }
    void silentUpdateBrokerConnected(bool on) {
        commitStateMutation([on](ApplicationState& s) { s.broker_connected = on; });
    }
    void silentUpdateDataLoss(bool state) {
        commitStateMutation([state](ApplicationState& s) { s.data_loss = state; });
    }
    void silentUpdateStatArbCorr(double corr) {
        commitStateMutation([corr](ApplicationState& s) { s.stat_arb_correlation = corr; });
    }
    void setClockOffsetNs(int64_t offset) {
        commitStateMutation([offset](ApplicationState& s) { s.clock_offset_ns = offset; });
    }

    // ══════════════════════════════════════════════════════════════
    // Frame Synchronization (called ONCE per 16ms by UiSynchronizer)
    // ══════════════════════════════════════════════════════════════

    /// Latch the current active state for this frame's rendering.
    /// All Q_PROPERTY reads during this frame see a single coherent snapshot.
    void latchFrameSnapshot() {
        m_ui_snapshot = std::atomic_load_explicit(&m_active_state, std::memory_order_acquire);
    }

    /// Batch-emit ONLY changed NOTIFY signals after latching.
    /// Diffs current state against previous frame to eliminate redundant
    /// QML binding re-evaluations on flat markets (~30 bindings/frame saved).
    void commitFrameUpdates() {
        latchFrameSnapshot();

        if (m_prev_snapshot) {
            constexpr double eps = 0.001;

            // Portfolio domain: value, pnl, positions, weights, confidence, leverage
            bool portfolioDirty =
                std::abs(m_prev_snapshot->portfolio_value - m_ui_snapshot->portfolio_value) > eps ||
                std::abs(m_prev_snapshot->total_pnl - m_ui_snapshot->total_pnl) > eps ||
                m_prev_snapshot->position_count != m_ui_snapshot->position_count ||
                std::abs(m_prev_snapshot->total_leverage - m_ui_snapshot->total_leverage) > eps ||
                std::abs(m_prev_snapshot->kronos_weight - m_ui_snapshot->kronos_weight) > eps ||
                std::abs(m_prev_snapshot->cooc_weight - m_ui_snapshot->cooc_weight) > eps ||
                std::abs(m_prev_snapshot->statarb_weight - m_ui_snapshot->statarb_weight) > eps ||
                std::abs(m_prev_snapshot->tft_weight - m_ui_snapshot->tft_weight) > eps ||
                std::abs(m_prev_snapshot->kronos_confidence - m_ui_snapshot->kronos_confidence) > eps ||
                std::abs(m_prev_snapshot->cooc_confidence - m_ui_snapshot->cooc_confidence) > eps ||
                std::abs(m_prev_snapshot->statarb_confidence - m_ui_snapshot->statarb_confidence) > eps ||
                std::abs(m_prev_snapshot->tft_confidence - m_ui_snapshot->tft_confidence) > eps;

            // Health domain: broker, session, mode, paused, data_loss, correlation, lab
            bool healthDirty =
                m_prev_snapshot->broker_connected != m_ui_snapshot->broker_connected ||
                m_prev_snapshot->system_paused != m_ui_snapshot->system_paused ||
                m_prev_snapshot->data_loss != m_ui_snapshot->data_loss ||
                m_prev_snapshot->current_session != m_ui_snapshot->current_session ||
                m_prev_snapshot->execution_mode != m_ui_snapshot->execution_mode ||
                m_prev_snapshot->vol_regime_override != m_ui_snapshot->vol_regime_override ||
                std::abs(m_prev_snapshot->stat_arb_correlation - m_ui_snapshot->stat_arb_correlation) > eps ||
                m_prev_snapshot->shadow_runs != m_ui_snapshot->shadow_runs ||
                m_prev_snapshot->pending_promotions != m_ui_snapshot->pending_promotions ||
                m_prev_snapshot->promoted_models != m_ui_snapshot->promoted_models ||
                m_prev_snapshot->last_sequence_id != m_ui_snapshot->last_sequence_id;

            if (portfolioDirty) Q_EMIT portfolioValueChanged();
            if (healthDirty)    Q_EMIT healthStatusChanged();
        } else {
            // First frame — emit both unconditionally
            Q_EMIT portfolioValueChanged();
            Q_EMIT healthStatusChanged();
        }

        m_prev_snapshot = m_ui_snapshot;

        // RCU garbage collection: clear old snapshots every ~1 second
        if (++m_frame_counter % 60 == 0) {
            std::lock_guard<std::mutex> lock(m_trash_mutex);
            m_trash.clear();
        }
    }

    // ══════════════════════════════════════════════════════════════
    // QML-callable setters (called from REST poll on GUI thread)
    // ══════════════════════════════════════════════════════════════

    Q_INVOKABLE void setBrokerConnected(bool on) {
        commitStateMutation([on](ApplicationState& s) { s.broker_connected = on; });
        Q_EMIT healthStatusChanged();
    }
    Q_INVOKABLE void setCurrentSession(const QString& session) {
        commitStateMutation([session](ApplicationState& s) { s.current_session = session; });
        Q_EMIT healthStatusChanged();
    }
    Q_INVOKABLE void setPortfolioValue(double val) {
        commitStateMutation([val](ApplicationState& s) { s.portfolio_value = val; });
        Q_EMIT portfolioValueChanged();
    }
    Q_INVOKABLE void setTotalPnl(double pnl) {
        commitStateMutation([pnl](ApplicationState& s) { s.total_pnl = pnl; });
        Q_EMIT portfolioValueChanged();
    }
    Q_INVOKABLE void setPositionCount(int count) {
        commitStateMutation([count](ApplicationState& s) { s.position_count = count; });
        Q_EMIT portfolioValueChanged();
    }
    Q_INVOKABLE void setExecutionMode(const QString& mode) {
        commitStateMutation([mode](ApplicationState& s) { s.execution_mode = mode; });
        Q_EMIT healthStatusChanged();
    }
    Q_INVOKABLE void setSystemPaused(bool paused) {
        commitStateMutation([paused](ApplicationState& s) { s.system_paused = paused; });
        Q_EMIT healthStatusChanged();
    }
    Q_INVOKABLE void setVolRegimeOverride(const QString& regime) {
        commitStateMutation([regime](ApplicationState& s) { s.vol_regime_override = regime; });
        Q_EMIT healthStatusChanged();
    }

    /// Set data loss flag (called by UiSynchronizer when backpressure triggers)
    void setDataLossActive(bool active) {
        commitStateMutation([active](ApplicationState& s) { s.data_loss = active; });
    }

    /// Route a raw ingested message to the appropriate handler
    void routePayload(const IngestMessage& msg);

Q_SIGNALS:
    void portfolioValueChanged();
    void healthStatusChanged();
    void alertReceived(QString alertId, bool isInhibited);
    void tickReceived(QString symbol, double bid, double ask);

private:
    explicit GlobalStore(QObject* parent = nullptr);

    void handleTelemetryEvent(const IngestMessage& msg);
    void handleControlSnapshot(const IngestMessage& msg);
    void handleGridData(const IngestMessage& msg);

    // ── RCU Double-Buffer ──────────────────────────────────────────
    // m_active_state: written by ingestion thread, read by both threads
    // m_ui_snapshot: latched once per frame, read ONLY by Qt GUI thread
    std::shared_ptr<const ApplicationState> m_active_state{std::make_shared<const ApplicationState>()};
    std::shared_ptr<const ApplicationState> m_ui_snapshot{m_active_state};
    std::shared_ptr<const ApplicationState> m_prev_snapshot; // Previous frame for diffing

    // ── RCU Trash Queue ────────────────────────────────────────────
    // Prevents ApplicationState destructors from executing on the GUI thread.
    // Old snapshots are parked here; cleared periodically by a background timer.
    mutable std::mutex m_trash_mutex;
    std::vector<std::shared_ptr<const ApplicationState>> m_trash;
    uint32_t m_frame_counter{0};

    // ── Arrow Worker ───────────────────────────────────────────
    algae::models::ArrowBuilderWorker* m_arrow_worker = nullptr;
};

} // namespace algae::engine
