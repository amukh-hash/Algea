// ─────────────────────────────────────────────────────────────────────
// FanChartItem — Probabilistic fan chart Scene Graph rendering
//
// Matches the implementation manual's Hardware-Accelerated Rendering
// and Camera-Relative Rendering (Blind Spot 2).
// ─────────────────────────────────────────────────────────────────────
#pragma once

// Arrow headers MUST come before Qt to avoid `signals` macro collision
#include <memory>
#include <atomic>
#include <vector>

#include <arrow/api.h>

#include <QQuickItem>
#include <QSGGeometryNode>
#include <QSGGeometry>
#include <QSGMaterial>
#include <QSGMaterialShader>

namespace algae::rendering {

// Dummy Material class to fulfill updatePaintNode signature expectation
class FanChartMaterial : public QSGMaterial {
public:
    QSGMaterialType* type() const override { static QSGMaterialType t; return &t; }
    QSGMaterialShader* createShader(QSGRendererInterface::RenderMode) const override { return nullptr; }
    int compare(const QSGMaterial* other) const override { return this == other ? 0 : 1; }
};

/// Hardware-accelerated Qt Scene Graph item visualizing quantile bands.
/// Directly maps Apache Arrow RecordBatches to custom Vulkan VBOs.
/// Utilizes camera-relative positioning to prevent Float32 mantissa exhaustion.
class FanChartItem : public QQuickItem {
    Q_OBJECT
    QML_ELEMENT

    Q_PROPERTY(double viewportCenterTimestamp READ viewportCenterTimestamp WRITE setViewportCenterTimestamp NOTIFY viewportChanged)

public:
    explicit FanChartItem(QQuickItem* parent = nullptr);

    /// Wait-free injection of the newly sealed RecordBatch
    void setActiveBatch(std::shared_ptr<arrow::RecordBatch> batch);

    double viewportCenterTimestamp() const { return m_viewport_center_timestamp; }
    void setViewportCenterTimestamp(double x);

Q_SIGNALS:
    void viewportChanged();

protected:
    QSGNode* updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) override;

private:
    std::shared_ptr<arrow::RecordBatch> m_active_batch;
    double m_viewport_center_timestamp = 0.0;

    std::vector<float> m_precalculated_distances;
    std::vector<float> m_precalculated_iqr_widths;
};

} // namespace algae::rendering
