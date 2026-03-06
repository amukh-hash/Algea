// ─────────────────────────────────────────────────────────────────────
// VulkanChartItem — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "VulkanChartItem.h"

#include <QSGVertexColorMaterial>

namespace algae::rendering {

VulkanChartItem::VulkanChartItem(QQuickItem* parent)
    : QQuickItem(parent)
{
    setFlag(ItemHasContents, true);
}

void VulkanChartItem::setBaseColor(const QColor& color) {
    if (m_baseColor != color) {
        m_baseColor = color;
        m_dirty = true;
        Q_EMIT baseColorChanged();
        update();
    }
}

void VulkanChartItem::markDirty() {
    m_dirty = true;
    Q_EMIT dirtyChanged();
    update();
}

QSGNode* VulkanChartItem::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) {
    if (!m_dirty && oldNode) return oldNode;

    // Rebuild geometry
    std::vector<ChartVertex> vertices;
    std::vector<uint32_t> indices;
    {
        std::lock_guard lock(m_geometryMutex);
        rebuildGeometry(vertices, indices);
    }

    if (vertices.empty()) {
        delete oldNode;
        return nullptr;
    }

    QSGGeometryNode* node = nullptr;
    QSGGeometry* geometry = nullptr;

    if (!oldNode) {
        node = new QSGGeometryNode();

        // Use default colored point 2D attributes (position + color per vertex)
        // Custom vertex layouts require a static const array, not a braced init
        auto& vertexColorAttribs = QSGGeometry::defaultAttributes_ColoredPoint2D();

        if (indices.empty()) {
            geometry = new QSGGeometry(vertexColorAttribs,
                                       static_cast<int>(vertices.size()));
            geometry->setDrawingMode(QSGGeometry::DrawTriangles);
        } else {
            geometry = new QSGGeometry(vertexColorAttribs,
                                       static_cast<int>(vertices.size()),
                                       static_cast<int>(indices.size()),
                                       QSGGeometry::UnsignedIntType);
            geometry->setDrawingMode(QSGGeometry::DrawTriangles);
        }

        node->setGeometry(geometry);
        node->setFlag(QSGNode::OwnsGeometry);

        auto* material = new QSGVertexColorMaterial();
        node->setMaterial(material);
        node->setFlag(QSGNode::OwnsMaterial);
    } else {
        node = static_cast<QSGGeometryNode*>(oldNode);
        geometry = node->geometry();

        if (indices.empty()) {
            geometry->allocate(static_cast<int>(vertices.size()));
        } else {
            geometry->allocate(static_cast<int>(vertices.size()),
                               static_cast<int>(indices.size()));
        }
    }

    // Upload vertices
    auto* vertexData = static_cast<ChartVertex*>(geometry->vertexData());
    std::memcpy(vertexData, vertices.data(), vertices.size() * sizeof(ChartVertex));

    // Upload indices
    if (!indices.empty()) {
        auto* indexData = geometry->indexDataAsUInt();
        std::memcpy(indexData, indices.data(), indices.size() * sizeof(uint32_t));
    }

    node->markDirty(QSGNode::DirtyGeometry);
    m_dirty = false;
    return node;
}

void VulkanChartItem::itemChange(ItemChange change, const ItemChangeData& value) {
    // Swapchain protection: when a panel is torn off the main window via
    // KDDockWidgets, the QML Window is destroyed and recreated on a new
    // HWND/Vulkan surface. We must force a full geometry rebuild to
    // prevent VK_ERROR_SURFACE_LOST_KHR from the old swapchain.
    if (change == ItemSceneChange) {
        m_dirty = true;
        update(); // Schedule rebuild on the new render surface
    }
    QQuickItem::itemChange(change, value);
}

} // namespace algae::rendering
