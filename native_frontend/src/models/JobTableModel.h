#pragma once

#include <QAbstractListModel>
#include <QDateTime>
#include <QString>
#include <QVector>

namespace algae::models {

struct JobRow {
    QString name;
    QString status;
    QString lastRun;
    double durationSeconds = 0.0;
    QString sessions;
    int dependencyCount = 0;
    QString errorSummary;
    qint64 freshnessMs = 0;
};

class JobTableModel : public QAbstractListModel {
    Q_OBJECT
public:
    enum JobRoles {
        NameRole = Qt::UserRole + 1,
        StatusRole,
        LastRunRole,
        DurationRole,
        SessionsRole,
        DependencyCountRole,
        ErrorSummaryRole,
        FreshnessMsRole,
    };

    explicit JobTableModel(QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QHash<int, QByteArray> roleNames() const override;

    void setRows(QVector<JobRow> rows);

private:
    QVector<JobRow> m_rows;
};

} // namespace algae::models
