// ─────────────────────────────────────────────────────────────────────
// SankeyDiagramItem — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "SankeyDiagramItem.h"
#include <QSGGeometryNode>
#include <QSGVertexColorMaterial>
#include <QVector2D>
#include <QPointF>
#include <cmath>
#include <algorithm>
#include <QQuickWindow>

namespace algae::rendering {

SankeyDiagramItem::SankeyDiagramItem(QQuickItem *parent) : QQuickItem(parent) {
    setFlag(ItemHasContents, true);
}

void SankeyDiagramItem::setLayout(const std::vector<SankeyNode>& sources,
                                  const std::vector<SankeyNode>& targets,
                                  const std::vector<SankeyLink>& links) {
    m_sources = sources;
    m_targets = targets;
    m_links = links;
    // Signal Qt Scene Graph to re-render
    update();
}

QSGNode* SankeyDiagramItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) {
    QSGGeometryNode *node = static_cast<QSGGeometryNode *>(oldNode);
    
    if (!node) {
        node = new QSGGeometryNode;
        auto* geom = new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        geom->setDrawingMode(QSGGeometry::DrawTriangleStrip);
        node->setGeometry(geom);
        node->setFlag(QSGNode::OwnsGeometry);

        auto* material = new QSGVertexColorMaterial();
        node->setMaterial(material);
        node->setFlag(QSGNode::OwnsMaterial);
    }

    if (m_sources.empty() || m_targets.empty() || m_links.empty()) return node;

    // Calculate total required vertices: Each link has 30 segments * 2 vertices per segment = 62 
    // Plus nodes (not drawn here, just flowing lines, but we could add nodes)
    int vertex_count = static_cast<int>(m_links.size() * (30 + 1) * 2);

    QSGGeometry *geometry = node->geometry();
    geometry->allocate(vertex_count);
    QSGGeometry::ColoredPoint2D *vertices = geometry->vertexDataAsColoredPoint2D();
    
    int v_idx = 0;

    // Dummy layout positions for the sake of the rendering algorithm demonstration
    float width_f = static_cast<float>(width());
    float height_f = static_cast<float>(height());

    for (size_t i = 0; i < m_links.size(); ++i) {
        const auto& link = m_links[i];
        
        // P0 (start) and P3 (end) control points
        QPointF p0(100.0f, (height_f / m_links.size()) * i);
        QPointF p3(width_f - 100.0f, (height_f / m_links.size()) * (m_links.size() - i));
        
        // P1 and P2 (curve control points to make horizontal exit/entry)
        QPointF p1(p0.x() + (p3.x() - p0.x()) / 3.0f, p0.y());
        QPointF p2(p0.x() + 2.0f * (p3.x() - p0.x()) / 3.0f, p3.y());

        float flow_thickness = static_cast<float>(std::max(5.0, link.value * 20.0));
        QColor color = (i % 2 == 0) ? QColor(50, 150, 255) : QColor(255, 100, 50);

        generateBezierMesh(vertices, v_idx, p0, p1, p2, p3, flow_thickness, color);
    }

    // Qt 6.10: setDrawCount removed; re-allocate to exact vertex count
    if (v_idx < vertex_count) {
        geometry->allocate(v_idx);
    }
    node->markDirty(QSGNode::DirtyGeometry);
    return node;
}

void SankeyDiagramItem::generateBezierMesh(QSGGeometry::ColoredPoint2D* vertices, int& v_idx, 
                                           QPointF p0, QPointF p1, QPointF p2, QPointF p3, 
                                           float flow_thickness, QColor color) {
    const int segments = 30; // Mesh resolution
    
    for (int i = 0; i <= segments; ++i) {
        float t = static_cast<float>(i) / segments;
        float u = 1.0f - t;
        
        // 1. Calculate Cubic Bezier Point
        QPointF point = static_cast<double>(u*u*u)*p0 + 
                        static_cast<double>(3*u*u*t)*p1 + 
                        static_cast<double>(3*u*t*t)*p2 + 
                        static_cast<double>(t*t*t)*p3;
        
        // 2. Calculate Tangent Vector to find the Normal
        QPointF tangent = static_cast<double>(-3*u*u)*p0 + 
                          static_cast<double>(3*(u*u - 2*u*t))*p1 + 
                          static_cast<double>(3*(2*u*t - t*t))*p2 + 
                          static_cast<double>(3*t*t)*p3;
                          
        QVector2D normal(-static_cast<float>(tangent.y()), static_cast<float>(tangent.x()));
        normal.normalize();
        
        // 3. Extrude vertices outward along the normal to create a thick polygon band
        QPointF top = point + (normal.toPointF() * static_cast<double>(flow_thickness / 2.0f));
        QPointF bottom = point - (normal.toPointF() * static_cast<double>(flow_thickness / 2.0f));
        
        vertices[v_idx++].set(static_cast<float>(top.x()), static_cast<float>(top.y()), 
                              color.red(), color.green(), color.blue(), 200);
        vertices[v_idx++].set(static_cast<float>(bottom.x()), static_cast<float>(bottom.y()), 
                              color.red(), color.green(), color.blue(), 200);
    }
}

} // namespace algae::rendering
