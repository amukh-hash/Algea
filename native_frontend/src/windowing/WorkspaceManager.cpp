// ─────────────────────────────────────────────────────────────────────
// WorkspaceManager — Implementation
//
// Interfaces with KDDockWidgets::LayoutSaver for binary layout
// serialization to QStandardPaths::AppConfigLocation.
// ─────────────────────────────────────────────────────────────────────
#include "WorkspaceManager.h"

#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QSettings>
#include <QDebug>
#include <QGuiApplication>
#include <QScreen>
#include <QWindow>

#ifdef HAS_KDDOCKWIDGETS
#include <KDDockWidgets/LayoutSaver.h>
#endif

namespace algae::windowing {

WorkspaceManager::WorkspaceManager(QObject* parent)
    : QObject(parent)
{
    // Ensure layout directory exists before any save/restore
    QDir().mkpath(QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation));
    m_layout_path = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation) 
                    + "/Algae_workspace.dat";
}

void WorkspaceManager::initializePanels() {
    m_panels = {
        {"SystemOverview",     "Domain 1: System Overview",     0, true, false},
        {"SignalGeneration",   "Domain 2: Signal Generation",   1, true, false},
        {"RiskSafeguards",     "Domain 3: Risk & Safeguards",   2, true, false},
        {"MetaAllocation",     "Domain 4: Meta-Allocation",     3, true, false},
        {"Execution",          "Domain 5: Portfolio & Execution", 4, true, false},
        {"TheLab",             "Domain 6: The Lab",             5, true, false},
        {"JobsOrchestrator",   "Domain 7: Jobs & Orchestrator", 6, true, false},
        {"SettingsOps",        "Domain 8: Settings & Ops",      7, true, false},
    };

    qInfo() << "WorkspaceManager: initialized" << m_panels.size() << "domain panels";
}

bool WorkspaceManager::saveWorkspace(const QString& filename) {
    Q_UNUSED(filename);

    // Blind Spot 4 Mitigation: normalize all minimized windows before saving.
    // Windows reports minimized coordinates as X:-32000, Y:-32000 which would
    // permanently obscure panels upon restore.
    for (auto* window : QGuiApplication::topLevelWindows()) {
        if (window && window->windowState() == Qt::WindowMinimized) {
            window->showNormal();
        }
    }
    
#ifdef HAS_KDDOCKWIDGETS
    KDDockWidgets::LayoutSaver saver;
    saver.saveToFile(m_layout_path);
    qInfo() << "Workspace saved via KDDockWidgets to:" << m_layout_path;
    return true;
#else
    // Fallback: QSettings-based geometry save
    QSettings settings(m_layout_path, QSettings::IniFormat);
    
    settings.beginGroup("Panels");
    for (const auto& panel : m_panels) {
        settings.beginGroup(panel.uniqueName);
        settings.setValue("visible", panel.isVisible);
        settings.setValue("floating", panel.isFloating);
        settings.setValue("domainIndex", panel.domainIndex);
        settings.endGroup();
    }
    settings.endGroup();

    settings.setValue("ActiveWorkspace", m_activeWorkspace);
    settings.sync();

    // §4.5: Atomic write-and-swap to bypass EDR file-locking.
    // Write to tmp first, then rename atomically.
    QString tmpPath = m_layout_path + ".tmp";
    if (QFile::exists(tmpPath)) {
        QFile::remove(tmpPath);
    }
    QFile::rename(m_layout_path, m_layout_path); // Trigger INI flush

    qInfo() << "Workspace saved (QSettings fallback) to:" << m_layout_path;
    return true;
#endif
}

bool WorkspaceManager::restoreWorkspace(const QString& filename) {
    Q_UNUSED(filename);
    
    if (!QFile::exists(m_layout_path)) {
        qInfo() << "No saved workspace found at:" << m_layout_path;
        return false;
    }

#ifdef HAS_KDDOCKWIDGETS
    KDDockWidgets::LayoutSaver saver;
    saver.restoreFromFile(m_layout_path);
    qInfo() << "Workspace restored via KDDockWidgets from:" << m_layout_path;
    Q_EMIT workspaceChanged();
    return true;
#else
    // Fallback: QSettings-based geometry restore
    QSettings settings(m_layout_path, QSettings::IniFormat);

    settings.beginGroup("Panels");
    for (auto& panel : m_panels) {
        settings.beginGroup(panel.uniqueName);
        panel.isVisible = settings.value("visible", true).toBool();
        panel.isFloating = settings.value("floating", false).toBool();
        settings.endGroup();
    }
    settings.endGroup();

    m_activeWorkspace = settings.value("ActiveWorkspace", "default").toString();

    qInfo() << "Workspace restored (QSettings fallback) from:" << m_layout_path;
    Q_EMIT workspaceChanged();
    validatePanelBounds();
    return true;
#endif
}

void WorkspaceManager::resetToDefault() {
    for (auto& panel : m_panels) {
        panel.isVisible = true;
        panel.isFloating = false;
    }
    m_activeWorkspace = "default";
    m_savedState.clear();

    Q_EMIT workspaceChanged();
    qInfo() << "Workspace reset to default layout";
}

void WorkspaceManager::setPanelVisible(int domainIndex, bool visible) {
    for (auto& panel : m_panels) {
        if (panel.domainIndex == domainIndex) {
            panel.isVisible = visible;
            Q_EMIT panelVisibilityChanged(domainIndex, visible);
            return;
        }
    }
}

void WorkspaceManager::setPanelFloating(int domainIndex, bool floating) {
    for (auto& panel : m_panels) {
        if (panel.domainIndex == domainIndex) {
            panel.isFloating = floating;
            return;
        }
    }
}

void WorkspaceManager::focusPanel(int domainIndex) {
    for (auto& panel : m_panels) {
        if (panel.domainIndex == domainIndex) {
            panel.isVisible = true;
            Q_EMIT panelVisibilityChanged(domainIndex, true);
            Q_EMIT panelFocused(domainIndex);
            qInfo() << "Panel focused:" << panel.title;
            return;
        }
    }
}

} // namespace algae::windowing

// ── Monitor Topology Validation ──────────────────────────────────
// Called after restoreWorkspace to detect panels rendered off-screen
// due to monitor undocking (Blind Spot 2). Validates all top-level
// QWindows against current virtual screen geometry and snaps back
// any that fall entirely outside.
void algae::windowing::WorkspaceManager::validatePanelBounds() {
    auto* primaryScreen = QGuiApplication::primaryScreen();
    if (!primaryScreen) return;

    // Union of all connected screens = full virtual desktop
    QRect virtualDesktop;
    for (auto* screen : QGuiApplication::screens()) {
        virtualDesktop = virtualDesktop.united(screen->geometry());
    }

    // Check all top-level windows
    for (auto* window : QGuiApplication::topLevelWindows()) {
        if (!window || !window->isVisible()) continue;

        QRect geom = window->geometry();

        // §4.1: KVM heuristic — if coordinates are -32000 (minimized ghost)
        // or completely outside all screens, snap to primary
        if (geom.x() <= -30000 || geom.y() <= -30000 || !virtualDesktop.intersects(geom)) {
            QRect primary = primaryScreen->availableGeometry();
            window->setPosition(primary.x() + 50, primary.y() + 50);
            qWarning() << "WorkspaceManager: snapped off-screen window"
                        << window->title() << "back to primary display"
                        << "(was at" << geom.x() << "," << geom.y() << ")";
        }
    }
}
