// ─────────────────────────────────────────────────────────────────────
// VulkanChartItem — QQuickItem with QSGGeometryNode for Vulkan rendering
//
// Base class for all algorithmic chart visualizations that bypass QML
// delegates and render directly into the Qt Scene Graph via custom
// QSGGeometryNode instances.
//
// Critical: Uses updatePaintNode() — NOT QQuickPaintedItem — to inject
// Vulkan-compatible geometry into the Qt 6 RHI pipeline while
// preserving Scene Graph Z-ordering and compositing.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QQuickItem>
#include <QSGGeometryNode>
#include <QSGFlatColorMaterial>
#include <QSGGeometry>
#include <QColor>

#include <vector>
#include <mutex>

namespace algae::rendering {

/// Vertex with position and per-vertex color
struct ChartVertex {
    float x, y;
    float r, g, b, a;
};

/// Base class for GPU-accelerated chart items rendered via Qt Scene Graph.
///
/// Subclasses override rebuildGeometry() to provide vertices and indices.
/// The base class handles Scene Graph node lifecycle and dirty flags.
class VulkanChartItem : public QQuickItem {
    Q_OBJECT
    QML_ELEMENT
    QML_UNCREATABLE("VulkanChartItem is an abstract base class")

    Q_PROPERTY(QColor baseColor READ baseColor WRITE setBaseColor NOTIFY baseColorChanged)
    Q_PROPERTY(bool dirty READ isDirty NOTIFY dirtyChanged)

public:
    explicit VulkanChartItem(QQuickItem* parent = nullptr);
    ~VulkanChartItem() override = default;

    QColor baseColor() const { return m_baseColor; }
    void setBaseColor(const QColor& color);

    bool isDirty() const { return m_dirty; }

    /// Mark geometry as needing rebuild on next frame
    void markDirty();

Q_SIGNALS:
    void baseColorChanged();
    void dirtyChanged();

protected:
    /// Override in subclasses to provide chart-specific geometry
    virtual void rebuildGeometry(
        std::vector<ChartVertex>& vertices,
        std::vector<uint32_t>& indices
    ) = 0;

    /// Qt Scene Graph integration point
    QSGNode* updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData* data) override;

    /// Swapchain protection: destroy geometry when scene changes
    /// during KDDockWidgets tear-off to prevent VK_ERROR_SURFACE_LOST_KHR
    void itemChange(ItemChange change, const ItemChangeData& value) override;

private:
    QColor m_baseColor{"#1a6bd1"};
    bool m_dirty = true;
    mutable std::mutex m_geometryMutex;
};

} // namespace algae::rendering
