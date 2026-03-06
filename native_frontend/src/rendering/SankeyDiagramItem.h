// ─────────────────────────────────────────────────────────────────────
// SankeyDiagramItem — Meta-Allocation flow visualization
//
// Generates thick Bézier curves triangulated on the CPU for capital 
// allocation flows using QSGGeometry::DrawTriangleStrip.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QQuickItem>
#include <QSGGeometryNode>
#include <QSGVertexColorMaterial>
#include <QVector2D>
#include <string>
#include <vector>

namespace algae::rendering {

struct SankeyNode {
    std::string label;
    double value;
    QColor color;
};

struct SankeyLink {
    int sourceIndex;
    int targetIndex;
    double value;
};

class SankeyDiagramItem : public QQuickItem {
    Q_OBJECT
    QML_ELEMENT

public:
    explicit SankeyDiagramItem(QQuickItem* parent = nullptr);

    void setLayout(const std::vector<SankeyNode>& sources,
                   const std::vector<SankeyNode>& targets,
                   const std::vector<SankeyLink>& links);

protected:
    QSGNode* updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) override;

private:
    void generateBezierMesh(QSGGeometry::ColoredPoint2D* vertices, int& v_idx, 
                            QPointF p0, QPointF p1, QPointF p2, QPointF p3, 
                            float flow_thickness, QColor color);

    std::vector<SankeyNode> m_sources;
    std::vector<SankeyNode> m_targets;
    std::vector<SankeyLink> m_links;
};

} // namespace algae::rendering
