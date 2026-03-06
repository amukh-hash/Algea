// ─────────────────────────────────────────────────────────────────────
// AlertDag — Implementation
//
// Processes incoming alerts via O(1) unordered_map lookup.
// Bidirectional inhibition: new symptoms are suppressed if their root
// cause is active, AND new root causes retroactively suppress existing
// un-inhibited symptoms.
//
// Parent-child relationships tracked exclusively by m_symptomsByRoot
// inverted index (Single Source of Truth). DagNode has no child refs.
// ─────────────────────────────────────────────────────────────────────
#include "AlertDag.h"

#include <QVariantMap>

namespace algae::engine {

AlertDag::AlertDag(QObject* parent)
    : QObject(parent)
{
}

void AlertDag::processAlert(
    const std::string& id,
    const std::string& root_cause_id,
    uint32_t severity,
    const std::string& message,
    const std::string& source,
    uint64_t timestamp_ns
) {
    std::lock_guard lock(m_mutex);

    auto new_node = std::make_shared<DagNode>();
    new_node->id = id;
    new_node->root_cause_id = root_cause_id;
    new_node->severity = severity;
    new_node->message = message;
    new_node->source = source;
    new_node->is_inhibited = false;
    new_node->timestamp_ns = timestamp_ns;

    if (!root_cause_id.empty()) {
        // --- SYMPTOM PATH ---
        // 1. O(1) insert into inverted index regardless of root cause presence
        m_symptomsByRoot[root_cause_id].insert(id);

        // 2. O(1) check if root cause is already active → suppress immediately
        auto parent_it = m_nodes.find(root_cause_id);
        if (parent_it != m_nodes.end() && !parent_it->second->is_inhibited) {
            new_node->is_inhibited = true;
            ++m_inhibitedCount;
        }
    } else {
        // --- ROOT CAUSE PATH ---
        // O(1) lookup via inverted index to suppress existing un-inhibited symptoms
        auto idx_it = m_symptomsByRoot.find(id);
        if (idx_it != m_symptomsByRoot.end()) {
            for (const auto& symptom_id : idx_it->second) {
                auto symptom_it = m_nodes.find(symptom_id);
                if (symptom_it != m_nodes.end() && !symptom_it->second->is_inhibited) {
                    symptom_it->second->is_inhibited = true;
                    ++m_inhibitedCount;
                    Q_EMIT alertInhibited(QString::fromStdString(symptom_id));
                }
            }
        }
    }

    m_nodes[id] = new_node;

    if (!new_node->is_inhibited) {
        Q_EMIT activeAlertGenerated(QString::fromStdString(id));
    }

    if (root_cause_id.empty()) {
        Q_EMIT rootCauseDetected(QString::fromStdString(id), severity);
    }

    Q_EMIT alertsChanged();
}

void AlertDag::clearAlert(const std::string& id) {
    std::lock_guard lock(m_mutex);
    
    auto it = m_nodes.find(id);
    if (it == m_nodes.end()) return;

    auto node = it->second;

    // Decrement inhibited counter if this node was suppressed
    if (node->is_inhibited) {
        --m_inhibitedCount;
    }

    if (node->root_cause_id.empty()) {
        // --- CLEARING A ROOT CAUSE ---
        // Un-inhibit its active symptom children via inverted index
        auto idx_it = m_symptomsByRoot.find(id);
        if (idx_it != m_symptomsByRoot.end()) {
            for (const auto& symptom_id : idx_it->second) {
                auto symptom_it = m_nodes.find(symptom_id);
                if (symptom_it != m_nodes.end() && symptom_it->second->is_inhibited) {
                    symptom_it->second->is_inhibited = false;
                    --m_inhibitedCount;
                    Q_EMIT activeAlertGenerated(QString::fromStdString(symptom_id));
                }
            }
        }
        // CRITICAL: We DO NOT erase from m_symptomsByRoot here.
        // The symptoms still exist in the UI — if this root cause
        // fires again, the index must still know about them to
        // immediately re-suppress.
    } else {
        // --- CLEARING A SYMPTOM ---
        // O(1) removal from its parent's set in the inverted index
        auto idx_it = m_symptomsByRoot.find(node->root_cause_id);
        if (idx_it != m_symptomsByRoot.end()) {
            idx_it->second.erase(id); // O(1) unordered_set::erase
            // Only destroy the map entry when no symptoms reference this root cause
            if (idx_it->second.empty()) {
                m_symptomsByRoot.erase(idx_it);
            }
        }
    }

    m_nodes.erase(it);
    Q_EMIT alertsChanged();
}

void AlertDag::clearAll() {
    std::lock_guard lock(m_mutex);
    m_nodes.clear();
    m_symptomsByRoot.clear();
    m_inhibitedCount = 0;
    Q_EMIT alertsChanged();
}

int AlertDag::activeAlertCount() const {
    std::lock_guard lock(m_mutex);
    return static_cast<int>(m_nodes.size());
}

int AlertDag::inhibitedCount() const {
    std::lock_guard lock(m_mutex);
    return m_inhibitedCount;
}

QVariantList AlertDag::visibleAlerts() const {
    std::lock_guard lock(m_mutex);
    QVariantList result;
    for (const auto& [_, node] : m_nodes) {
        if (node->is_inhibited) continue; // Skip suppressed symptoms

        QVariantMap alertMap;
        alertMap["id"] = QString::fromStdString(node->id);
        alertMap["severity"] = node->severity;
        alertMap["message"] = QString::fromStdString(node->message);
        alertMap["source"] = QString::fromStdString(node->source);
        alertMap["isRootCause"] = node->root_cause_id.empty();

        // Count symptoms via inverted index (single source of truth)
        int symptomCount = 0;
        if (node->root_cause_id.empty()) {
            auto idx_it = m_symptomsByRoot.find(node->id);
            if (idx_it != m_symptomsByRoot.end()) {
                symptomCount = static_cast<int>(idx_it->second.size());
            }
        }
        alertMap["symptomCount"] = symptomCount;

        result.append(alertMap);
    }
    return result;
}

} // namespace algae::engine
