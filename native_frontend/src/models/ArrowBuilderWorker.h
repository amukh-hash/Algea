// ─────────────────────────────────────────────────────────────────────
// ArrowBuilderWorker — Background thread that batches incoming ticks
// into Arrow RecordBatch objects and atomically swaps them into the
// ArrowTableModel.
//
// Blind Spot 3 Mitigation: Avoid tick-by-tick append into immutable
// ChunkedArray (O(N) realloc per tick). Instead:
//   - Buffer ticks into mutable C++ vectors
//   - Seal into RecordBatch at 1,000 items OR 500ms timer
//   - Atomic pointer swap into ArrowTableModel (via shared_ptr)
// ─────────────────────────────────────────────────────────────────────
#pragma once

// Arrow headers MUST come before Qt to avoid `signals` macro collision
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>

#include <QObject>
#include <QTimer>

namespace algae::models {

class ArrowTableModel;

} // namespace algae::models

// Forward decls for scene graph routing targets
namespace algae::rendering {
    class FanChartItem;
    class SankeyDiagramItem;
}

namespace algae::models {

/// Incoming tick data to be buffered into an Arrow RecordBatch
struct TickRow {
    uint64_t sequence_id;
    uint64_t timestamp_ns;
    std::string symbol;
    double bid;
    double ask;
    double last;
    double volume;
};

/// Background worker that buffers tick data and periodically seals
/// it into Arrow RecordBatches for the ArrowTableModel.
///
/// Seal triggers:
///   1. Buffer reaches 1,000 rows (capacity trigger)
///   2. 500ms elapsed since last seal (time trigger)
///
/// The sealed batch is atomically swapped into the target model
/// via ArrowTableModel::swapBatch().
class ArrowBuilderWorker : public QObject {
    Q_OBJECT

public:
    static constexpr size_t BATCH_CAPACITY = 1'000;
    static constexpr int SEAL_INTERVAL_MS = 500;

    /// @param target The ArrowTableModel to receive sealed batches
    explicit ArrowBuilderWorker(ArrowTableModel* target, QObject* parent = nullptr);
    ~ArrowBuilderWorker() override = default;

    /// Start the seal timer
    void start();

    /// Stop the seal timer
    void stop();

    /// Append a single tick. Thread-safe (called from UiSynchronizer context).
    void appendTick(const TickRow& tick);

    /// Append a batch of ticks at once (more efficient)
    void appendTicks(const std::vector<TickRow>& ticks);

    /// Process an incoming Arrow IPC payload from ZMQ (schema-hardened, no .ValueOrDie())
    void processIpcPayload(const std::string& topic, const uint8_t* data, size_t size);

    /// Set rendering targets for scene graph hydration
    void setFanChartTarget(algae::rendering::FanChartItem* target) { m_fan_chart_item = target; }
    void setSankeyTarget(algae::rendering::SankeyDiagramItem* target) { m_sankey_item = target; }
    void setPositionsTarget(ArrowTableModel* target) { m_positions_model = target; }

    /// Current buffer size (diagnostic)
    size_t bufferSize() const;

Q_SIGNALS:
    void batchSealed(int rowCount);

private Q_SLOTS:
    void onSealTimer();

private:
    void sealAndSwap();
    std::shared_ptr<arrow::RecordBatch> buildBatch();

    ArrowTableModel* m_target;
    QTimer* m_sealTimer;

    // Scene graph routing targets (set via setXxxTarget)
    algae::rendering::FanChartItem* m_fan_chart_item = nullptr;
    algae::rendering::SankeyDiagramItem* m_sankey_item = nullptr;
    ArrowTableModel* m_positions_model = nullptr;

    mutable std::mutex m_bufferMutex;
    std::vector<uint64_t> m_buf_seq;
    std::vector<uint64_t> m_buf_ts;
    std::vector<std::string> m_buf_symbol;
    std::vector<double> m_buf_bid;
    std::vector<double> m_buf_ask;
    std::vector<double> m_buf_last;
    std::vector<double> m_buf_volume;
};

} // namespace algae::models
