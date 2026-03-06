// ─────────────────────────────────────────────────────────────────────
// AlertDag — Alert fatigue mitigation via DAG root-cause analysis
//
// Utilizes std::unordered_map with internal DagNode shared_ptr pointers
// to build graph relationships dynamically in memory.
// Topological O(1) hash lookup for root-cause inhibition.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QString>
#include <QVariantList>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace algae::engine {

/// Internal DAG node — leaf metadata only.
/// Parent-child relationships are tracked exclusively by the
/// m_symptomsByRoot inverted index (Single Source of Truth).
struct DagNode {
    std::string id;
    std::string root_cause_id;
    uint32_t severity = 0;
    std::string message;
    std::string source;
    bool is_inhibited = false;
    uint64_t timestamp_ns = 0;
};

/// DAG-based alert engine that suppresses symptom alerts when their
/// root cause is already active, preventing operator cognitive overload.
class AlertDag : public QObject {
    Q_OBJECT

    Q_PROPERTY(int activeAlertCount READ activeAlertCount NOTIFY alertsChanged)
    Q_PROPERTY(int inhibitedCount READ inhibitedCount NOTIFY alertsChanged)
    Q_PROPERTY(QVariantList visibleAlerts READ visibleAlerts NOTIFY alertsChanged)

public:
    explicit AlertDag(QObject* parent = nullptr);

    /// Process an incoming alert — evaluates root cause and sets inhibition
    void processAlert(
        const std::string& id,
        const std::string& root_cause_id,
        uint32_t severity,
        const std::string& message,
        const std::string& source,
        uint64_t timestamp_ns
    );

    /// Clear a specific alert and un-inhibit orphaned symptoms
    void clearAlert(const std::string& id);

    /// Clear all alerts
    void clearAll();

    int activeAlertCount() const;
    int inhibitedCount() const;
    QVariantList visibleAlerts() const;

Q_SIGNALS:
    void alertsChanged();
    void alertInhibited(QString alertId);
    void activeAlertGenerated(QString alertId);
    void rootCauseDetected(QString alertId, uint32_t severity);

private:
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, std::shared_ptr<DagNode>> m_nodes;

    // O(1) inverted index: root_cause_id → set of symptom IDs
    // Uses unordered_set for O(1) insert AND O(1) erase (vs O(K) vector shift).
    // Eliminates the O(N) full-scan that caused CPU starvation during alert storms.
    std::unordered_map<std::string, std::unordered_set<std::string>> m_symptomsByRoot;

    int m_inhibitedCount = 0;
};

} // namespace algae::engine
