// ─────────────────────────────────────────────────────────────────────
// ArrowTableModel — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "ArrowTableModel.h"

#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>

#include <QDebug>

namespace algae::models {

ArrowTableModel::ArrowTableModel(QObject* parent)
    : QAbstractTableModel(parent)
{
}

int ArrowTableModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    auto batch = std::atomic_load_explicit(&m_active_batch, std::memory_order_acquire);
    return batch ? static_cast<int>(batch->num_rows()) : 0;
}

int ArrowTableModel::columnCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    auto batch = std::atomic_load_explicit(&m_active_batch, std::memory_order_acquire);
    return batch ? batch->num_columns() : 0;
}

QVariant ArrowTableModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || role != Qt::DisplayRole) return QVariant();

    auto batch = std::atomic_load_explicit(&m_active_batch, std::memory_order_acquire);
    if (!batch) return QVariant();
    if (index.row() >= batch->num_rows()) return QVariant();
    if (index.column() >= batch->num_columns()) return QVariant();

    auto column = batch->column(index.column());

    switch (column->type_id()) {
        case arrow::Type::DOUBLE: {
            auto arr = std::static_pointer_cast<arrow::DoubleArray>(column);
            return arr->Value(index.row());
        }
        case arrow::Type::FLOAT: {
            auto arr = std::static_pointer_cast<arrow::FloatArray>(column);
            return static_cast<double>(arr->Value(index.row()));
        }
        case arrow::Type::INT64: {
            auto arr = std::static_pointer_cast<arrow::Int64Array>(column);
            return static_cast<qlonglong>(arr->Value(index.row()));
        }
        case arrow::Type::INT32: {
            auto arr = std::static_pointer_cast<arrow::Int32Array>(column);
            return arr->Value(index.row());
        }
        case arrow::Type::UINT64: {
            auto arr = std::static_pointer_cast<arrow::UInt64Array>(column);
            return static_cast<qulonglong>(arr->Value(index.row()));
        }
        case arrow::Type::STRING: {
            auto arr = std::static_pointer_cast<arrow::StringArray>(column);
            return QString::fromStdString(arr->GetString(index.row()));
        }
        case arrow::Type::LARGE_STRING: {
            auto arr = std::static_pointer_cast<arrow::LargeStringArray>(column);
            return QString::fromStdString(arr->GetString(index.row()));
        }
        case arrow::Type::BOOL: {
            auto arr = std::static_pointer_cast<arrow::BooleanArray>(column);
            return arr->Value(index.row());
        }
        default:
            return QVariant();
    }
}

QVariant ArrowTableModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (role != Qt::DisplayRole) return QVariant();

    if (orientation == Qt::Horizontal) {
        auto batch = std::atomic_load_explicit(&m_active_batch, std::memory_order_acquire);
        if (batch && section < batch->num_columns()) {
            return QString::fromStdString(batch->schema()->field(section)->name());
        }
    } else {
        return section; // Row numbers
    }
    return QVariant();
}

QHash<int, QByteArray> ArrowTableModel::roleNames() const {
    QHash<int, QByteArray> roles;
    roles[Qt::DisplayRole] = "display";
    return roles;
}

void ArrowTableModel::swapBatch(std::shared_ptr<arrow::RecordBatch> newBatch) {
    // Wait-free atomic pointer swap. Qt Render thread reads m_active_batch without locking.
    std::atomic_store_explicit(&m_active_batch, newBatch, std::memory_order_release);
    
    // Notify Qt UI of row insertion (invoked on Main Thread)
    QMetaObject::invokeMethod(this, [this]() {
        beginResetModel();
        endResetModel();
        
        auto batch = std::atomic_load_explicit(&m_active_batch, std::memory_order_acquire);
        int rows = batch ? static_cast<int>(batch->num_rows()) : 0;
        Q_EMIT batchSwapped(rows);
    }, Qt::QueuedConnection);
}

bool ArrowTableModel::loadFromIPC(const uint8_t* data, size_t size) {
    auto buffer = std::make_shared<arrow::Buffer>(data, static_cast<int64_t>(size));
    auto bufferReader = std::make_shared<arrow::io::BufferReader>(buffer);

    // Use BoundedArrowPool (500MB cap) to prevent OOM from malformed payloads.
    // The global pool instance is defined in ArrowBuilderWorker.cpp.
    extern BoundedArrowPool g_arrow_pool;
    arrow::ipc::IpcReadOptions options;
    options.memory_pool = &g_arrow_pool;

    auto result = arrow::ipc::RecordBatchStreamReader::Open(bufferReader, options);
    if (!result.ok()) {
        qWarning() << "Failed to open Arrow IPC stream:"
                    << QString::fromStdString(result.status().ToString());
        return false;
    }

    auto reader = result.ValueUnsafe();
    std::shared_ptr<arrow::RecordBatch> batch;
    auto readResult = reader->ReadNext(&batch);

    if (!readResult.ok() || !batch) {
        qWarning() << "Failed to read Arrow RecordBatch:"
                    << QString::fromStdString(readResult.ToString());
        return false;
    }

    swapBatch(std::move(batch));
    return true;
}

int ArrowTableModel::currentRowCount() const {
    auto batch = std::atomic_load_explicit(&m_active_batch, std::memory_order_acquire);
    return batch ? static_cast<int>(batch->num_rows()) : 0;
}

} // namespace algae::models
