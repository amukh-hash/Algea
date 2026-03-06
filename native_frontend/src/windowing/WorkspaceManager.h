// ─────────────────────────────────────────────────────────────────────
// WorkspaceManager — KDDockWidgets orchestration and state persistence
//
// Manages the 8 domain panels as tear-off DockWidgets across multiple
// monitors. Workspace layouts are serialized to workspace.dat for
// session restore via QSettings/LayoutManager.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QString>
#include <QByteArray>
#include <QHash>

#include <functional>
#include <vector>

namespace algae::windowing {

/// Metadata for each domain panel
struct DomainPanel {
    QString uniqueName;     // e.g. "SystemOverview", "RiskSafeguards"
    QString title;          // e.g. "Domain 1: System Overview"
    int domainIndex;        // 0-7
    bool isVisible = true;
    bool isFloating = false;
};

/// Manages KDDockWidgets layout, multi-monitor placement, and
/// workspace state serialization/restoration.
///
/// In builds without KDDockWidgets (HAS_KDDOCKWIDGETS=0), falls back
/// to a simple QSettings-based geometry save/restore.
class WorkspaceManager : public QObject {
    Q_OBJECT

    Q_PROPERTY(int panelCount READ panelCount CONSTANT)
    Q_PROPERTY(QString activeWorkspace READ activeWorkspace NOTIFY workspaceChanged)

public:
    explicit WorkspaceManager(QObject* parent = nullptr);

    /// Initialize the 8 domain panels
    void initializePanels();

    /// Save the current layout to file
    Q_INVOKABLE bool saveWorkspace(const QString& filename = "workspace.dat");

    /// Restore layout from file
    Q_INVOKABLE bool restoreWorkspace(const QString& filename = "workspace.dat");

    /// Reset to default layout (all docked, single monitor)
    Q_INVOKABLE void resetToDefault();

    /// Show/hide a specific domain panel
    Q_INVOKABLE void setPanelVisible(int domainIndex, bool visible);

    /// Float/dock a specific domain panel  
    Q_INVOKABLE void setPanelFloating(int domainIndex, bool floating);

    /// Bring a panel to front (for exception-based alert routing)
    Q_INVOKABLE void focusPanel(int domainIndex);

    int panelCount() const { return static_cast<int>(m_panels.size()); }
    QString activeWorkspace() const { return m_activeWorkspace; }

    /// Get the list of all panels
    const std::vector<DomainPanel>& panels() const { return m_panels; }

Q_SIGNALS:
    void workspaceChanged();
    void panelVisibilityChanged(int domainIndex, bool visible);
    void panelFocused(int domainIndex);

private:
    void validatePanelBounds();

    std::vector<DomainPanel> m_panels;
    QString m_activeWorkspace = "default";
    QString m_layout_path;
    QByteArray m_savedState;
};

} // namespace algae::windowing
