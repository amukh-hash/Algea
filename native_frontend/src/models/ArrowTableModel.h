// ─────────────────────────────────────────────────────────────────────
// ArrowTableModel — QAbstractTableModel backed by Apache Arrow
//
// Maps data() directly to Arrow columnar memory for O(1) viewport
// virtualization. Uses std::shared_lock to protect against pointer
// swap during Arrow IPC reads from the ArrowBuilderWorker.
// ─────────────────────────────────────────────────────────────────────
#pragma once

// Arrow headers MUST come before Qt headers because Qt's
// `#define signals public` macro rewrites Arrow's parameter
// named `signals` in cancel.h, causing a parse error.
#include <memory>
#include <atomic>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/ipc/reader.h>

#include <QAbstractTableModel>
#include <QVariant>

namespace algae::models {

/// Arrow-backed table model for streaming grids.
///
/// The ArrowBuilderWorker thread buffers ticks into C++ vectors, seals
/// them into RecordBatch at 1,000 items / 500ms, then atomically swaps
/// the shared_ptr here. QML TableView with uniformRowHeights renders
/// only visible rows for O(1) complexity.
class ArrowTableModel : public QAbstractTableModel {
    Q_OBJECT

public:
    explicit ArrowTableModel(QObject* parent = nullptr);

    // QAbstractTableModel interface
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    /// Atomically swap the underlying RecordBatch (called from ArrowBuilderWorker)
    void swapBatch(std::shared_ptr<arrow::RecordBatch> newBatch);

    /// Deserialize an Arrow IPC stream and swap the batch
    bool loadFromIPC(const uint8_t* data, size_t size);

    /// Current row count (lock-free read for status display)
    int currentRowCount() const;

Q_SIGNALS:
    void batchSwapped(int rowCount);

private:
    std::shared_ptr<arrow::RecordBatch> m_active_batch;
    std::vector<std::string> m_column_names;
};

} // namespace algae::models
