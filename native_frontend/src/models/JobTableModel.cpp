#include "JobTableModel.h"

namespace algae::models {

JobTableModel::JobTableModel(QObject* parent) : QAbstractListModel(parent) {}

int JobTableModel::rowCount(const QModelIndex& parent) const {
    if (parent.isValid()) return 0;
    return m_rows.size();
}

QVariant JobTableModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || index.row() < 0 || index.row() >= m_rows.size()) {
        return {};
    }
    const auto& row = m_rows.at(index.row());
    switch (role) {
        case NameRole: return row.name;
        case StatusRole: return row.status;
        case LastRunRole: return row.lastRun;
        case DurationRole: return row.durationSeconds;
        case SessionsRole: return row.sessions;
        case DependencyCountRole: return row.dependencyCount;
        case ErrorSummaryRole: return row.errorSummary;
        case FreshnessMsRole: return row.freshnessMs;
        default: return {};
    }
}

QHash<int, QByteArray> JobTableModel::roleNames() const {
    return {
        {NameRole, "name"},
        {StatusRole, "status"},
        {LastRunRole, "lastRun"},
        {DurationRole, "durationSeconds"},
        {SessionsRole, "sessions"},
        {DependencyCountRole, "dependencyCount"},
        {ErrorSummaryRole, "errorSummary"},
        {FreshnessMsRole, "freshnessMs"},
    };
}

void JobTableModel::setRows(QVector<JobRow> rows) {
    beginResetModel();
    m_rows = std::move(rows);
    endResetModel();
}

} // namespace algae::models
