// ─────────────────────────────────────────────────────────────────────
// ParallelCoordinatesItem — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "ParallelCoordinatesItem.h"
#include <QSGGeometryNode>
#include <QSGFlatColorMaterial>
#include <cmath>
#include <algorithm>

namespace algae::rendering {

ParallelCoordinatesItem::ParallelCoordinatesItem(QQuickItem *parent) : QQuickItem(parent) {
    setFlag(ItemHasContents, true);
}

void ParallelCoordinatesItem::setDistributionData(const std::vector<ParallelTrace>& trajectories, const std::vector<double>& expert_distributions) {
    std::unique_lock lock(m_data_mutex);
    m_trajectories = trajectories;
    m_expert_distributions = expert_distributions;
    update();
}

QSGNode* ParallelCoordinatesItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) {
    QSGGeometryNode *node = static_cast<QSGGeometryNode *>(oldNode);
    
    if (!node) {
        node = new QSGGeometryNode;
        // Allocate geometry for colored points (x, y, r, g, b, a)
        auto* geom = new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        geom->setDrawingMode(QSGGeometry::DrawLineStrip);
        geom->setLineWidth(1.5f); // Vulkan wide-line support required in physical device features
        node->setGeometry(geom);
        node->setFlag(QSGNode::OwnsGeometry);

        auto* material = new QSGVertexColorMaterial();
        node->setMaterial(material);
        node->setFlag(QSGNode::OwnsMaterial);
    }

    std::shared_lock lock(m_data_mutex);
    if (m_trajectories.empty()) return node;

    // 1. Calculate Router Entropy: H = -Sum(P(x) * log2(P(x)))
    double entropy = 0.0;
    for (const auto& expert_prob : m_expert_distributions) {
        if (expert_prob > 0) entropy -= expert_prob * std::log2(expert_prob);
    }
    
    // 2. Map entropy to color shift (High entropy = Green/Stable, Low entropy = Red/Collapse)
    float red_channel = static_cast<float>(std::max(0.0, 1.0 - (entropy / m_max_entropy)));
    float green_channel = static_cast<float>(std::min(1.0, entropy / m_max_entropy));

    QSGGeometry *geometry = node->geometry();
    int vertex_count = static_cast<int>(m_trajectories.size() * m_dimensions);
    
    // Blind Spot 2 Mitigation: Geometric Over-provisioning to avoid Vulkan VRAM Fragmentation
    if (geometry->vertexCount() < vertex_count || geometry->vertexCount() > vertex_count * 2) {
        // Over-provision by nearest power of 2 or chunk
        int target_alloc = ((vertex_count / 1024) + 1) * 1024;
        geometry->allocate(target_alloc);
    }

    QSGGeometry::ColoredPoint2D *vertices = geometry->vertexDataAsColoredPoint2D();

    int idx = 0;
    for (const auto& trajectory : m_trajectories) {
        // Ensure bounds to avoid vector overflow
        int dim_limit = std::min(m_dimensions, static_cast<int>(trajectory.values.size()));
        
        for (int dim = 0; dim < dim_limit; ++dim) {
            float x = (width() / std::max(1, m_dimensions - 1)) * dim;
            float y = static_cast<float>(height() - (trajectory.values[dim] * height())); // Normalize 0-1 to pixel height
            
            // set(x, y, r, g, b, a) expects 0-255 scaling
            vertices[idx].set(x, y, 
                static_cast<unsigned char>(red_channel * 255), 
                static_cast<unsigned char>(green_channel * 255), 
                0, 
                150);
            idx++;
        }
    }

    // Set draw count to active vertices, leaving capacity allocated in GPU memory
    geometry->setVertexDataPattern(QSGGeometry::DynamicPattern);
    node->markDirty(QSGNode::DirtyGeometry);
    return node;
}

} // namespace algae::rendering
