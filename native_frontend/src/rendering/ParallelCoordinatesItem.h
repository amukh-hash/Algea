// ─────────────────────────────────────────────────────────────────────
// ParallelCoordinatesItem — Instanced rendering for sleeve comparison
//
// Hardware-accelerated Scene Graph lines for SMoE Parallel Coordinates.
// Renders gating Shannon Entropy as color distribution natively.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QQuickItem>
#include <QSGGeometryNode>
#include <QSGVertexColorMaterial>
#include <QColor>
#include <vector>
#include <string>
#include <shared_mutex>

namespace algae::rendering {

/// A single data trace across all parallel axes
struct ParallelTrace {
    std::string label;
    std::vector<double> values; 
};

class ParallelCoordinatesItem : public QQuickItem {
    Q_OBJECT
    QML_ELEMENT

public:
    explicit ParallelCoordinatesItem(QQuickItem* parent = nullptr);

    // SMoE expert routing logic
    void setDistributionData(const std::vector<ParallelTrace>& trajectories, const std::vector<double>& expert_distributions);

protected:
    // Overriding the Qt Quick Scene Graph direct hardware hook
    QSGNode* updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) override;

private:
    std::vector<ParallelTrace> m_trajectories;
    std::vector<double> m_expert_distributions;
    
    int m_dimensions = 6;
    double m_max_entropy = 3.0; // log2(8 experts)
    std::shared_mutex m_data_mutex;
};

} // namespace algae::rendering
