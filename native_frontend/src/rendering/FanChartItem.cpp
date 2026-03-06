// ─────────────────────────────────────────────────────────────────────
// FanChartItem — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "FanChartItem.h"

#include <QSGGeometryNode>
#include <QDebug>

// Standard GL_TRIANGLE_STRIP definition for QSGGeometry
#ifndef GL_TRIANGLE_STRIP
#define GL_TRIANGLE_STRIP 0x0005
#endif

namespace algae::rendering {

FanChartItem::FanChartItem(QQuickItem* parent)
    : QQuickItem(parent)
{
    setFlag(ItemHasContents, true); // Enforces updatePaintNode execution lifecycle
}

void FanChartItem::setActiveBatch(std::shared_ptr<arrow::RecordBatch> batch) {
    // Wait-free atomic swap allows the UI thread to push updates to the render node
    std::atomic_store_explicit(&m_active_batch, batch, std::memory_order_release);
    update(); // Schedule a render frame
}

void FanChartItem::setViewportCenterTimestamp(double x) {
    if (m_viewport_center_timestamp != x) {
        m_viewport_center_timestamp = x;
        Q_EMIT viewportChanged();
        update();
    }
}

QSGNode* FanChartItem::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) {
    QSGGeometryNode* node = static_cast<QSGGeometryNode*>(oldNode);
    
    if (!node) {
        node = new QSGGeometryNode;
        // Use the default colored point 2D attributes (position + color)
        // Custom Vulkan vertex layouts would require QSGMaterialShader overrides
        auto* geom = new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        geom->setDrawingMode(QSGGeometry::DrawTriangleStrip);
        node->setGeometry(geom);
        node->setFlag(QSGNode::OwnsGeometry);

        // Bind the compiled SPIR-V Vulkan Material
        auto* material = new FanChartMaterial(); 
        node->setMaterial(material);
        node->setFlag(QSGNode::OwnsMaterial);
    } 

    // 2. Safely load the atomic RecordBatch pointer set by ArrowBuilderWorker/GlobalStore
    std::shared_ptr<arrow::RecordBatch> batch = std::atomic_load_explicit(&m_active_batch, std::memory_order_acquire);
    if (!batch || batch->num_rows() == 0) return node;

    QSGGeometry* geometry = node->geometry();
    geometry->allocate(static_cast<int>(batch->num_rows()));
    float* vertexData = static_cast<float*>(geometry->vertexData());

    // Expecting columns: 0 = int64 timestamp, 1 = double baseline price
    if (batch->num_columns() < 2) return node;

    auto ts_col = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
    auto price_col = std::static_pointer_cast<arrow::DoubleArray>(batch->column(1));

    // Ensure our mock dimension vectors size match
    if (m_precalculated_distances.size() < static_cast<size_t>(batch->num_rows())) {
        m_precalculated_distances.resize(batch->num_rows(), 0.05f);
        m_precalculated_iqr_widths.resize(batch->num_rows(), 2.0f);
    }

    // 3. Populate VBO utilizing Camera-Relative precision adjustments
    double center_time = m_viewport_center_timestamp; 

    for (int i = 0; i < batch->num_rows(); ++i) {
        // CPU calculates double-precision offset before truncating to float32
        vertexData[i * 4 + 0] = static_cast<float>(ts_col->Value(i) - center_time); 
        vertexData[i * 4 + 1] = static_cast<float>(price_col->Value(i));
        vertexData[i * 4 + 2] = m_precalculated_distances[i]; 
        vertexData[i * 4 + 3] = m_precalculated_iqr_widths[i]; 
    }

    node->markDirty(QSGNode::DirtyGeometry);
    return node;
}

} // namespace algae::rendering
