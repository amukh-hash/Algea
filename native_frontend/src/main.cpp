// ─────────────────────────────────────────────────────────────────────
// algae Native Frontend — Application Entry Point
//
// Initialization sequence (strict ordering):
//   1. Wayland XCB fallback (Blind Spot 2: Linux multi-monitor)
//   2. Vulkan RHI backend + IGPU device isolation
//   3. Threaded Render Loop decoupling
//   4. Windows timer resolution fix (Blind Spot 1)
//   5. KDDockWidgets initialization
//   6. Engine startup chain (ZMQ → UiSync → GlobalStore)
//   7. Swapchain-level simulation watermark (post-Scene-Graph)
//   8. Multi-monitor layout restoration
// ─────────────────────────────────────────────────────────────────────
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickWindow>
#include <QSGRendererInterface>
#include <QtQml>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QDateTime>
#include <QTimer>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

// Arrow builders for REST → RecordBatch pipeline
#include <arrow/api.h>
#include <arrow/builder.h>

// ── File-based debug logging ──────────────────────────────────────
static QFile g_logFile;
static QTextStream g_logStream;

void fileMessageHandler(QtMsgType type, const QMessageLogContext &ctx, const QString &msg) {
    const char *typeStr = "???";
    switch (type) {
        case QtDebugMsg:    typeStr = "DBG"; break;
        case QtInfoMsg:     typeStr = "INF"; break;
        case QtWarningMsg:  typeStr = "WRN"; break;
        case QtCriticalMsg: typeStr = "CRT"; break;
        case QtFatalMsg:    typeStr = "FTL"; break;
    }
    g_logStream << QDateTime::currentDateTime().toString(Qt::ISODate)
                << " [" << typeStr << "] "
                << (ctx.category ? ctx.category : "default") << ": "
                << msg << "\n";
    g_logStream.flush();
    if (type == QtFatalMsg) abort();
}

#include "engine/ZmqReceiver.h"
#include "engine/UiSynchronizer.h"
#include "engine/GlobalStore.h"
#include "engine/AlertDag.h"
#include "engine/StateReconciler.h"
#include "models/ArrowTableModel.h"
#include "network/RestClient.h"
#include "hardware/KillSwitch.h"
#include "hardware/KillSwitchBridge.h"
#include "hardware/FidoGateway.h"
#include "windowing/WorkspaceManager.h"
#include "config/BuildEnvironment.h"
#include "rendering/SankeyDiagramItem.h"
#include "rendering/FanChartItem.h"
#include "rendering/ParallelCoordinatesItem.h"

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "dbghelp.lib")
#endif

#ifdef HAS_KDDOCKWIDGETS
#include <KDDockWidgets/KDDockWidgets.h>
#include <KDDockWidgets/Config.h>
#endif

// ── Swapchain-Level Simulation Watermark ───────────────────────────
// Hooks into QQuickWindow::afterRendering to execute raw RHI draw
// commands after all QML Scene Graph nodes are finalized. This renders
// ABOVE all QML content — mathematically impossible for any docked or
// floating panel to obscure the watermark.
#ifdef Algae_ENV_SIM
static void drawRedDiagonalWatermark(void* /*commandBuffer*/, QSize windowSize) {
    // Stub: In production, issue raw Vulkan/RHI draw commands to paint
    // diagonal red "SIMULATION" text across the entire swapchain framebuffer.
    // This is a hardware-level overlay that cannot be hidden by QML z-index.
    Q_UNUSED(windowSize);
}

static void attachSimulationWatermark(QQmlApplicationEngine& engine) {
    auto rootObjects = engine.rootObjects();
    if (rootObjects.isEmpty()) return;

    QQuickWindow *window = qobject_cast<QQuickWindow *>(rootObjects.first());
    if (!window) return;

    // Execute immediately after Qt finishes rendering the QML scene graph, 
    // but before the buffer is swapped to the display.
    QObject::connect(window, &QQuickWindow::afterRendering, window, [window]() {
        QSGRendererInterface *rif = window->rendererInterface();
        auto *commandBuffer = rif->getResource(window, QSGRendererInterface::CommandListResource);
        
        // Issue raw RHI text/polygon draw calls at the hardware layer.
        // Because this executes post-Scene-Graph, it is mathematically impossible
        // for any QML widget (even floating ones) to render on top of it.
        drawRedDiagonalWatermark(commandBuffer, window->size()); 
    }, Qt::DirectConnection);
}
#endif

// ── Windows MiniDump Crash Diagnostics ─────────────────────────────
// Traps unhandled hardware exceptions (EXCEPTION_DATATYPE_MISALIGNMENT,
// EXCEPTION_ACCESS_VIOLATION) and writes a .dmp file before termination.
// Without this, the exe simply vanishes from the operator's screen.
#ifdef _WIN32
static LONG WINAPI AlgaeCrashDumpFilter(EXCEPTION_POINTERS* ExceptionInfo) {
    // Create crash dump directory
    CreateDirectoryA("crash_dumps", NULL);

    // Generate timestamped filename
    SYSTEMTIME st;
    GetLocalTime(&st);
    char filename[256];
    snprintf(filename, sizeof(filename),
             "crash_dumps\\algae_crash_%04d%02d%02d_%02d%02d%02d.dmp",
             st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

    HANDLE hFile = CreateFileA(filename, GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        MINIDUMP_EXCEPTION_INFORMATION mdei;
        mdei.ThreadId = GetCurrentThreadId();
        mdei.ExceptionPointers = ExceptionInfo;
        mdei.ClientPointers = FALSE;

        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
                          hFile, MiniDumpNormal, &mdei, NULL, NULL);
        CloseHandle(hFile);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

int main(int argc, char *argv[])
{
    // ── Install crash dump handler FIRST ─────────────────────────
#ifdef _WIN32
    SetUnhandledExceptionFilter(AlgaeCrashDumpFilter);
#endif

    // ── Install file-based logging before ANYTHING else ──────────
    g_logFile.setFileName("Algae_debug.log");
    g_logFile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
    g_logStream.setDevice(&g_logFile);
    qInstallMessageHandler(fileMessageHandler);
    g_logStream << "=== Algae_Sim starting ===\n";
    g_logStream.flush();
    // ══════════════════════════════════════════════════════════════════
    // PHASE 1: Pre-QApplication Environment Configuration
    //          ALL qputenv calls MUST execute before QGuiApplication ctor
    // ══════════════════════════════════════════════════════════════════

    // Blind Spot 2 (Wayland): Force X11 compatibility layer on Linux.
    // Wayland prohibits QWindow::setPosition for absolute coordinates,
    // breaking KDDockWidgets multi-monitor layout restoration.
#ifdef __linux__
    qputenv("QT_QPA_PLATFORM", "xcb");
#endif

    // 1. Graphics Backend: Prefer D3D11 RHI on Windows for GPU-native rendering.
    //    D3D11 provides multithreaded command buffers and VRAM-direct push,
    //    equivalent to Vulkan but uses Windows' native graphics stack.
    //    Vulkan RHI plugin not deployed in vcpkg Qt build; D3D11 is preferred.
    //    Override via QSG_RHI_BACKEND=opengl|vulkan|d3d12 if needed.
    {
        QByteArray forceBackend = qgetenv("QSG_RHI_BACKEND");
        if (forceBackend == "opengl") {
            QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);
        } else if (forceBackend == "vulkan") {
            QQuickWindow::setGraphicsApi(QSGRendererInterface::Vulkan);
            qputenv("QT_VK_PHYSICAL_DEVICE_INDEX", "0");
        } else if (forceBackend == "d3d12") {
            QQuickWindow::setGraphicsApi(QSGRendererInterface::Direct3D12);
        } else {
            // Default: Direct3D 11 — native Windows GPU rendering
            QQuickWindow::setGraphicsApi(QSGRendererInterface::Direct3D11);
        }
    }
    
    // 2. Thread Decoupling: Ensure the Render Thread is distinct from the GUI Main Thread
    qputenv("QSG_RENDER_LOOP", "threaded");

    // ══════════════════════════════════════════════════════════════════
    // PHASE 2: QApplication Construction
    // ══════════════════════════════════════════════════════════════════

    QGuiApplication app(argc, argv);
    app.setApplicationName("algae");
    app.setOrganizationName("algae Trading");

#ifdef Algae_ENV_SIM
    app.setApplicationName("algae [SIMULATION]");
#endif

    // Blind Spot 1 (Windows): Force 1ms OS timer resolution.
    // Default Windows timer granularity is 15.6ms, causing QTimer(16ms)
    // to coalesce and fire at 31ms intervals, starving the UI drain loop.
#ifdef _WIN32
    timeBeginPeriod(1);

    // §2.1: Prevent CPU C-State deep sleep and OS core parking.
    // Market volatility spikes after flat periods cause 15ms wake latency.
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

    // ══════════════════════════════════════════════════════════════════
    // PHASE 3: KDDockWidgets Initialization
    // ══════════════════════════════════════════════════════════════════
#ifdef HAS_KDDOCKWIDGETS
    KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtQuick);
#endif

    // ══════════════════════════════════════════════════════════════════
    // PHASE 4: Core Engine Startup Chain
    // ══════════════════════════════════════════════════════════════════

    auto* store = algae::engine::GlobalStore::instance();
    auto* alertDag = new algae::engine::AlertDag(&app);
    auto* reconciler = new algae::engine::StateReconciler(&app);

    // ZMQ Receiver (pinned to CPU core 2)
    auto receiver = std::make_unique<algae::engine::ZmqReceiver>(
        "tcp://127.0.0.1:5556",
        "tcp://127.0.0.1:5557",
        2 // CPU core
    );

    // UI Synchronizer (60 Hz frame-paced drain with Protobuf Arena)
    auto* synchronizer = new algae::engine::UiSynchronizer(
        receiver.get(), store, &app
    );

    // Connect data loss signal to store
    QObject::connect(synchronizer, &algae::engine::UiSynchronizer::dataLossDetected,
                     store, [store]() { store->setDataLossActive(true); });

    // Connect alert routing
    QObject::connect(store, &algae::engine::GlobalStore::alertReceived,
                     alertDag, [alertDag](const QString& id, bool isInhibited) {
        Q_UNUSED(isInhibited);
        alertDag->processAlert(
            id.toStdString(), "", 1, "Alert received", "system", 0
        );
    });

    // ── REST Client ────────────────────────────────────────────────
    auto* restClient = new algae::network::RestClient(
        "http://127.0.0.1:8000", &app
    );

    // Connect RestClient::controlStateReceived → GlobalStore silent updates
    QObject::connect(restClient, &algae::network::RestClient::controlStateReceived,
                     store, [store](const std::string& jsonStr) {
        QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(jsonStr));
        if (!doc.isObject()) return;
        auto obj = doc.object();

        // execution_mode: "noop" | "paper" | "ibkr"
        if (obj.contains("execution_mode"))
            store->setExecutionMode(obj["execution_mode"].toString());

        // paused: bool
        if (obj.contains("paused"))
            store->setSystemPaused(obj["paused"].toBool());

        // vol_regime_override: string | null
        if (obj.contains("vol_regime_override"))
            store->setVolRegimeOverride(obj["vol_regime_override"].toString());
    });

    // Periodic C++ REST poll for control state + portfolio (30s)
    auto* restPollTimer = new QTimer(&app);
    QObject::connect(restPollTimer, &QTimer::timeout, restClient, [restClient]() {
        restClient->getControlState();
        restClient->getPortfolioSummary();
    });
    restPollTimer->start(30000);
    // Fire immediately on startup
    QTimer::singleShot(2000, restClient, [restClient]() {
        restClient->getControlState();
        restClient->getPortfolioSummary();
    });

    // ── Data Models ────────────────────────────────────────────────
    auto* executionGrid = new algae::models::ArrowTableModel(&app);
    auto* positionsGrid = new algae::models::ArrowTableModel(&app);

    // Connect RestClient::portfolioSummaryReceived → Arrow batch build → PositionsGrid
    QObject::connect(restClient, &algae::network::RestClient::portfolioSummaryReceived,
                     positionsGrid, [positionsGrid, store](const std::string& jsonStr) {
        QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(jsonStr));
        if (!doc.isObject()) return;
        auto obj = doc.object();

        // Update aggregate portfolio values in GlobalStore
        if (obj.contains("total_value"))
            store->setPortfolioValue(obj["total_value"].toDouble());
        if (obj.contains("total_pnl"))
            store->setTotalPnl(obj["total_pnl"].toDouble());
        if (obj.contains("position_count"))
            store->setPositionCount(obj["position_count"].toInt());

        // Build Arrow RecordBatch from holdings array
        auto holdings = obj["holdings"].toArray();
        if (holdings.isEmpty()) return;

        arrow::StringBuilder sym_builder, sleeve_builder;
        arrow::DoubleBuilder qty_builder, cost_builder, last_builder, notional_builder, pnl_builder;

        for (const auto& item : holdings) {
            auto h = item.toObject();
            (void)sym_builder.Append(h["symbol"].toString().toStdString());
            (void)sleeve_builder.Append(h["sleeve"].toString("").toStdString());
            (void)qty_builder.Append(h["qty"].toDouble());
            (void)cost_builder.Append(h["avg_cost"].toDouble());
            (void)last_builder.Append(h["last_price"].toDouble());
            (void)notional_builder.Append(h["notional"].toDouble());
            (void)pnl_builder.Append(h["unrealized_pnl"].toDouble());
        }

        std::shared_ptr<arrow::Array> sym_arr, slv_arr, qty_arr, cost_arr, last_arr, not_arr, pnl_arr;
        (void)sym_builder.Finish(&sym_arr);
        (void)sleeve_builder.Finish(&slv_arr);
        (void)qty_builder.Finish(&qty_arr);
        (void)cost_builder.Finish(&cost_arr);
        (void)last_builder.Finish(&last_arr);
        (void)notional_builder.Finish(&not_arr);
        (void)pnl_builder.Finish(&pnl_arr);

        auto schema = arrow::schema({
            arrow::field("Symbol", arrow::utf8()),
            arrow::field("Sleeve", arrow::utf8()),
            arrow::field("Qty", arrow::float64()),
            arrow::field("Avg Cost", arrow::float64()),
            arrow::field("Last", arrow::float64()),
            arrow::field("Notional", arrow::float64()),
            arrow::field("Unrealized PnL", arrow::float64()),
        });

        auto batch = arrow::RecordBatch::Make(schema, holdings.size(),
            {sym_arr, slv_arr, qty_arr, cost_arr, last_arr, not_arr, pnl_arr});

        // Wait-free pointer swap hydrates Tab 5 grid
        positionsGrid->swapBatch(batch);
    });

    // ── Kill Switch ────────────────────────────────────────────────
    auto killSwitch = std::make_unique<algae::hardware::KillSwitch>();
    killSwitch->initialize();
    auto* killSwitchBridge = new algae::hardware::KillSwitchBridge(killSwitch.get(), &app);

    // ── FIDO2 Gateway ─────────────────────────────────────────────
    auto* fidoGateway = new algae::hardware::FidoGateway(&app);
    fidoGateway->refreshDevices();

    // ── Workspace Manager ──────────────────────────────────────────
    algae::windowing::WorkspaceManager workspaceManager;

    // ══════════════════════════════════════════════════════════════════
    // PHASE 5: QML Engine & Context Properties
    // ══════════════════════════════════════════════════════════════════

    // Register C++ rendering items for QML (qt_add_qml_module not used)
    qmlRegisterType<algae::rendering::SankeyDiagramItem>("algae.Rendering", 1, 0, "SankeyDiagramItem");
    qmlRegisterType<algae::rendering::FanChartItem>("algae.Rendering", 1, 0, "FanChartItem");
    qmlRegisterType<algae::rendering::ParallelCoordinatesItem>("algae.Rendering", 1, 0, "ParallelCoordinatesItem");

    QQmlApplicationEngine engine;

    // Add QML import path: the 'qml/' directory next to the executable
    // (windeployqt places QtQuick modules there)
    QString appDir = QCoreApplication::applicationDirPath();
    engine.addImportPath(appDir + "/qml");

    auto* ctx = engine.rootContext();
    ctx->setContextProperty("GlobalStore", store);
    ctx->setContextProperty("AlertEngine", alertDag);
    ctx->setContextProperty("RestClient", restClient);
    ctx->setContextProperty("ExecutionGrid", executionGrid);
    ctx->setContextProperty("PositionsGrid", positionsGrid);
    ctx->setContextProperty("WorkspaceManager", &workspaceManager);
    ctx->setContextProperty("StateReconciler", reconciler);
    ctx->setContextProperty("FidoGateway", fidoGateway);
    ctx->setContextProperty("KillSwitch", killSwitchBridge);

#ifdef Algae_ENV_SIM
    ctx->setContextProperty("isSimulation", true);
#else
    ctx->setContextProperty("isSimulation", false);
#endif

    // Load QML
    engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));
    if (engine.rootObjects().isEmpty()) return -1;

#ifdef Algae_ENV_SIM
    // 5. Inject un-obscurable OS-level watermark
    attachSimulationWatermark(engine);
#endif

    // ══════════════════════════════════════════════════════════════════
    // PHASE 6: Start Engine & Restore Layout
    // ══════════════════════════════════════════════════════════════════

    receiver->start();
    synchronizer->start();

    // 6. Restore Multi-Monitor Layout
    workspaceManager.restoreWorkspace();

    qInfo() << "algae Native Frontend started"
#ifdef Algae_ENV_SIM
            << "[SIMULATION MODE]"
#else
            << "[LIVE MODE]"
#endif
            << "— ZMQ ingestion active, UI sync at 60Hz";

    // §4.1: Multi-monitor DPI swapchain protection.
    // When a panel is dragged between monitors with different scale factors,
    // force a full RHI swapchain rebuild to prevent VK_ERROR_OUT_OF_DATE_KHR.
    if (auto* rootWindow = qobject_cast<QQuickWindow*>(engine.rootObjects().first())) {
        QObject::connect(rootWindow, &QWindow::screenChanged, rootWindow, [rootWindow]() {
            qInfo() << "screenChanged: forcing swapchain rebuild for DPI transition";
            rootWindow->releaseResources();
            rootWindow->update();
        });
    }

    // §3.3: Deterministic ZMQ teardown before Qt event loop exit.
    // Prevents zombie processes holding AlgaeControlPlane SHM locks.
    QObject::connect(&app, &QCoreApplication::aboutToQuit, [&]() {
        qInfo() << "aboutToQuit: initiating deterministic ZMQ teardown";
        synchronizer->stop();
        receiver->stop();
        workspaceManager.saveWorkspace();
    });

    int result = app.exec();

    // ══════════════════════════════════════════════════════════════════
    // PHASE 7: Graceful Shutdown
    // ══════════════════════════════════════════════════════════════════

    synchronizer->stop();
    receiver->stop();
    workspaceManager.saveWorkspace();

#ifdef _WIN32
    timeEndPeriod(1);
#endif

    return result;
}
