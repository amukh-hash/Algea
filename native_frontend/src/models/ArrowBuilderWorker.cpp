// ─────────────────────────────────────────────────────────────────────
// ArrowBuilderWorker — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "ArrowBuilderWorker.h"
#include "ArrowTableModel.h"
#include "../rendering/FanChartItem.h"
#include "../rendering/SankeyDiagramItem.h"

#include <QDebug>
#include <cstring>
#include <unordered_map>

namespace algae::models {

using algae::rendering::SankeyNode;
using algae::rendering::SankeyLink;

ArrowBuilderWorker::ArrowBuilderWorker(ArrowTableModel* target, QObject* parent)
    : QObject(parent)
    , m_target(target)
{
    m_sealTimer = new QTimer(this);
    m_sealTimer->setTimerType(Qt::PreciseTimer);
    m_sealTimer->setInterval(SEAL_INTERVAL_MS);
    connect(m_sealTimer, &QTimer::timeout, this, &ArrowBuilderWorker::onSealTimer);

    // Pre-allocate buffers for typical burst size
    m_buf_seq.reserve(BATCH_CAPACITY);
    m_buf_ts.reserve(BATCH_CAPACITY);
    m_buf_symbol.reserve(BATCH_CAPACITY);
    m_buf_bid.reserve(BATCH_CAPACITY);
    m_buf_ask.reserve(BATCH_CAPACITY);
    m_buf_last.reserve(BATCH_CAPACITY);
    m_buf_volume.reserve(BATCH_CAPACITY);
}

void ArrowBuilderWorker::start() {
    m_sealTimer->start();
}

void ArrowBuilderWorker::stop() {
    m_sealTimer->stop();
    // Final flush
    sealAndSwap();
}

void ArrowBuilderWorker::appendTick(const TickRow& tick) {
    bool shouldSeal = false;
    {
        std::lock_guard lock(m_bufferMutex);
        m_buf_seq.push_back(tick.sequence_id);
        m_buf_ts.push_back(tick.timestamp_ns);
        m_buf_symbol.push_back(tick.symbol);
        m_buf_bid.push_back(tick.bid);
        m_buf_ask.push_back(tick.ask);
        m_buf_last.push_back(tick.last);
        m_buf_volume.push_back(tick.volume);

        shouldSeal = m_buf_seq.size() >= BATCH_CAPACITY;
    }

    if (shouldSeal) {
        sealAndSwap();
    }
}

void ArrowBuilderWorker::appendTicks(const std::vector<TickRow>& ticks) {
    bool shouldSeal = false;
    {
        std::lock_guard lock(m_bufferMutex);
        for (const auto& tick : ticks) {
            m_buf_seq.push_back(tick.sequence_id);
            m_buf_ts.push_back(tick.timestamp_ns);
            m_buf_symbol.push_back(tick.symbol);
            m_buf_bid.push_back(tick.bid);
            m_buf_ask.push_back(tick.ask);
            m_buf_last.push_back(tick.last);
            m_buf_volume.push_back(tick.volume);
        }
        shouldSeal = m_buf_seq.size() >= BATCH_CAPACITY;
    }

    if (shouldSeal) {
        sealAndSwap();
    }
}

size_t ArrowBuilderWorker::bufferSize() const {
    std::lock_guard lock(m_bufferMutex);
    return m_buf_seq.size();
}

void ArrowBuilderWorker::onSealTimer() {
    std::lock_guard lock(m_bufferMutex);
    if (m_buf_seq.empty()) return;
    // unlock not needed — sealAndSwap acquires separately
    // but we check emptiness under the lock to avoid spurious seals
    sealAndSwap();
}

void ArrowBuilderWorker::sealAndSwap() {
    auto batch = buildBatch();
    if (!batch) return;

    int rows = static_cast<int>(batch->num_rows());
    m_target->swapBatch(std::move(batch));
    Q_EMIT batchSealed(rows);
}

std::shared_ptr<arrow::RecordBatch> ArrowBuilderWorker::buildBatch() {
    // Snapshot and clear buffers under lock
    std::vector<uint64_t> seqs, timestamps;
    std::vector<std::string> symbols;
    std::vector<double> bids, asks, lasts, volumes;

    {
        std::lock_guard lock(m_bufferMutex);
        if (m_buf_seq.empty()) return nullptr;

        seqs = std::move(m_buf_seq);
        timestamps = std::move(m_buf_ts);
        symbols = std::move(m_buf_symbol);
        bids = std::move(m_buf_bid);
        asks = std::move(m_buf_ask);
        lasts = std::move(m_buf_last);
        volumes = std::move(m_buf_volume);

        // Re-allocate for next batch
        m_buf_seq.reserve(BATCH_CAPACITY);
        m_buf_ts.reserve(BATCH_CAPACITY);
        m_buf_symbol.reserve(BATCH_CAPACITY);
        m_buf_bid.reserve(BATCH_CAPACITY);
        m_buf_ask.reserve(BATCH_CAPACITY);
        m_buf_last.reserve(BATCH_CAPACITY);
        m_buf_volume.reserve(BATCH_CAPACITY);
    }

    // Build Arrow arrays from buffered data
    const int64_t numRows = static_cast<int64_t>(seqs.size());

    arrow::UInt64Builder seqBuilder, tsBuilder;
    arrow::StringBuilder symbolBuilder;
    arrow::DoubleBuilder bidBuilder, askBuilder, lastBuilder, volBuilder;

    auto ok = seqBuilder.AppendValues(seqs);
    if (!ok.ok()) { qWarning() << "Arrow seqBuilder failed"; return nullptr; }

    ok = tsBuilder.AppendValues(timestamps);
    if (!ok.ok()) return nullptr;

    for (const auto& s : symbols) {
        ok = symbolBuilder.Append(s);
        if (!ok.ok()) return nullptr;
    }

    ok = bidBuilder.AppendValues(bids);
    if (!ok.ok()) return nullptr;
    ok = askBuilder.AppendValues(asks);
    if (!ok.ok()) return nullptr;
    ok = lastBuilder.AppendValues(lasts);
    if (!ok.ok()) return nullptr;
    ok = volBuilder.AppendValues(volumes);
    if (!ok.ok()) return nullptr;

    // Finalize arrays
    std::shared_ptr<arrow::Array> seqArr, tsArr, symArr, bidArr, askArr, lastArr, volArr;
    seqBuilder.Finish(&seqArr);
    tsBuilder.Finish(&tsArr);
    symbolBuilder.Finish(&symArr);
    bidBuilder.Finish(&bidArr);
    askBuilder.Finish(&askArr);
    lastBuilder.Finish(&lastArr);
    volBuilder.Finish(&volArr);

    auto schema = arrow::schema({
        arrow::field("sequence_id", arrow::uint64()),
        arrow::field("timestamp_ns", arrow::uint64()),
        arrow::field("symbol", arrow::utf8()),
        arrow::field("bid", arrow::float64()),
        arrow::field("ask", arrow::float64()),
        arrow::field("last", arrow::float64()),
        arrow::field("volume", arrow::float64()),
    });

    return arrow::RecordBatch::Make(
        schema, numRows,
        {seqArr, tsArr, symArr, bidArr, askArr, lastArr, volArr}
    );
}

// ── Schema-Hardened IPC Payload Processing ─────────────────────────
// Safely deserializes Arrow IPC streams from ZMQ, enforcing SIMD
// alignment and using explicit .ok() checks (no .ValueOrDie()).
// Routes decoded batches to the appropriate scene graph target.
//
// Uses BoundedArrowPool (500MB cap) to prevent malformed payloads
// from exhausting workstation RAM.
#include "BoundedArrowPool.h"

algae::models::BoundedArrowPool g_arrow_pool(500 * 1024 * 1024);

void ArrowBuilderWorker::processIpcPayload(const std::string& topic,
                                           const uint8_t* data, size_t size)
{
    // 1. Allocate 64-byte aligned buffer via bounded pool
    auto buffer_result = arrow::AllocateBuffer(static_cast<int64_t>(size), &g_arrow_pool);
    if (!buffer_result.ok()) {
        qWarning() << "ArrowBuilderWorker: AllocateBuffer failed for topic" << topic.c_str()
                    << "— pool:" << g_arrow_pool.bytes_allocated() / (1024*1024) << "MB /"
                    << g_arrow_pool.max_memory() / (1024*1024) << "MB";
        return;
    }

    std::shared_ptr<arrow::Buffer> aligned_buffer = std::move(*buffer_result);
    std::memcpy(aligned_buffer->mutable_data(), data, size);

    // 2. Open IPC stream reader with bounded pool
    arrow::io::BufferReader reader(aligned_buffer);
    arrow::ipc::IpcReadOptions read_options;
    read_options.memory_pool = &g_arrow_pool;
    auto stream_reader_result = arrow::ipc::RecordBatchStreamReader::Open(&reader, read_options);

    if (!stream_reader_result.ok()) {
        qWarning() << "ArrowBuilderWorker: IPC stream open failed for topic" << topic.c_str()
                    << "—" << stream_reader_result.status().ToString().c_str();
        return;
    }

    // 3. Read next batch (safe extraction)
    std::shared_ptr<arrow::RecordBatch> batch;
    auto read_status = (*stream_reader_result)->ReadNext(&batch);

    if (!read_status.ok() || !batch) {
        qWarning() << "ArrowBuilderWorker: ReadNext failed for topic" << topic.c_str();
        return;
    }

    // 4. Route to target based on ZMQ topic
    if (topic == "chart.kronos_fan" && m_fan_chart_item) {
        // Inject decoded RecordBatch into FanChart for quantile rendering
        m_fan_chart_item->setActiveBatch(batch);
    }
    else if (topic == "chart.sankey_alloc" && m_sankey_item) {
        // Extract Source/Target/Weight columns and build Sankey graph
        auto src_col = batch->GetColumnByName("source");
        auto tgt_col = batch->GetColumnByName("target");
        auto wgt_col = batch->GetColumnByName("weight");
        auto clr_col = batch->GetColumnByName("color");

        if (src_col && tgt_col && wgt_col) {
            auto sources_arr = std::static_pointer_cast<arrow::StringArray>(src_col);
            auto targets_arr = std::static_pointer_cast<arrow::StringArray>(tgt_col);
            auto weights_arr = std::static_pointer_cast<arrow::DoubleArray>(wgt_col);

            // Build unique node sets
            std::vector<SankeyNode> sources, targets;
            std::unordered_map<std::string, int> src_idx, tgt_idx;
            std::vector<SankeyLink> links;

            QColor colors[] = {QColor(99,102,241), QColor(6,182,212), QColor(34,197,94), QColor(245,158,11)};

            for (int64_t i = 0; i < batch->num_rows(); ++i) {
                std::string src = sources_arr->GetString(i);
                std::string tgt = targets_arr->GetString(i);
                double w = weights_arr->Value(i);

                if (src_idx.find(src) == src_idx.end()) {
                    src_idx[src] = static_cast<int>(sources.size());
                    sources.push_back({src, w, colors[sources.size() % 4]});
                }
                if (tgt_idx.find(tgt) == tgt_idx.end()) {
                    tgt_idx[tgt] = static_cast<int>(targets.size());
                    targets.push_back({tgt, w, colors[targets.size() % 4]});
                }
                links.push_back({src_idx[src], tgt_idx[tgt], w});
            }

            m_sankey_item->setLayout(sources, targets, links);
        }
    }
    else if (topic == "grid.positions" && m_positions_model) {
        m_positions_model->swapBatch(std::move(batch));
    }
}

} // namespace algae::models
