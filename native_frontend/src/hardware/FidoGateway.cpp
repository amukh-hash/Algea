// ─────────────────────────────────────────────────────────────────────
// FidoGateway — FIDO2/CTAP2 Hardware Key Authentication
//
// When HAS_LIBFIDO2 is defined (vcpkg), uses the real libfido2 C API
// with QtConcurrent::run to offload blocking hardware I/O.
// Otherwise, falls back to stubbed implementation.
//
// Blind Spot 2 Mitigation: fido_dev_get_assert() is synchronously
// blocking (5-15s human touch). QtConcurrent::run dispatches to the
// global thread pool so the GUI thread maintains 60Hz rendering.
//
// Blind Spot 3 Mitigation: Uses "windows://" protocol path on Windows
// to route through native WebAuthn API, bypassing raw HID permission
// restrictions that require Administrator elevation.
// ─────────────────────────────────────────────────────────────────────
#include "FidoGateway.h"

#include <QDebug>
#include <QByteArray>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>

#ifdef HAS_LIBFIDO2
#include <fido.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace algae::hardware {

FidoGateway::FidoGateway(QObject* parent)
    : QObject(parent)
{
#ifdef HAS_LIBFIDO2
    qInfo() << "FidoGateway: libfido2 linked — FIDO2 hardware auth enabled";
#else
    qInfo() << "FidoGateway: libfido2 not available — FIDO2 hardware auth disabled";
#endif
}

FidoGateway::~FidoGateway() = default;

void FidoGateway::refreshDevices() {
#ifdef HAS_LIBFIDO2
    fido_dev_info_t *devlist = fido_dev_info_new(64);
    size_t ndevs = 0;

    if (fido_dev_info_manifest(devlist, 64, &ndevs) == FIDO_OK && ndevs > 0) {
        const fido_dev_info_t *di = fido_dev_info_ptr(devlist, 0);
        m_available = true;
        m_devicePath = fido_dev_info_path(di);
        m_deviceName = QString::fromUtf8(fido_dev_info_product_string(di));
    } else {
        m_available = false;
        m_deviceName = "";
        m_devicePath = "";
    }

    fido_dev_info_free(&devlist, 64);
#else
    m_available = false;
    m_deviceName = "";
    m_devicePath = "";
#endif
    Q_EMIT availabilityChanged();
    Q_EMIT deviceChanged();
}

void FidoGateway::requestSignature(const QString& strategyGitHash) {
#ifdef HAS_LIBFIDO2
    // Guard against concurrent hardware calls from aggressive UI clicking
    if (m_waiting) return;

    m_waiting = true;
    Q_EMIT waitingChanged();
    Q_EMIT touchRequired();

    // Deep copy the hash by value to prevent dangling references in background thread
    std::string hash = strategyGitHash.toStdString();

    // Dispatch blocking hardware I/O to the global C++ thread pool
    QFuture<std::pair<bool, QString>> future = QtConcurrent::run([hash]() {
        fido_init(0);
        fido_dev_t* dev = fido_dev_new();

        // On Windows, "windows://" routes through native WebAuthn API
        // bypassing raw HID permission blocks (Blind Spot 3)
#ifdef _WIN32
        const char* devPath = "windows://";
#else
        const char* devPath = nullptr; // Use first available on Linux
#endif
        if (fido_dev_open(dev, devPath) != FIDO_OK) {
            fido_dev_free(&dev);
            return std::make_pair(false, QString("Failed to open FIDO2 hardware token."));
        }

        fido_assert_t* assert_obj = fido_assert_new();
        fido_assert_set_clientdata_hash(assert_obj,
            reinterpret_cast<const unsigned char*>(hash.data()),
            static_cast<size_t>(hash.size()));

        // THREAD BLOCKS HERE waiting for physical human touch
        int r = fido_dev_get_assert(dev, assert_obj, nullptr);

        QString signature;
        if (r == FIDO_OK) {
            // Base64Url without padding — JWT/WebAuthn compatible
            QByteArray raw_sig(
                reinterpret_cast<const char*>(fido_assert_sig_ptr(assert_obj, 0)),
                static_cast<qsizetype>(fido_assert_sig_len(assert_obj, 0))
            );
            signature = QString::fromLatin1(
                raw_sig.toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals)
            );
            // §4.3: Obliterate raw signature bytes in RAM
#ifdef _WIN32
            SecureZeroMemory(raw_sig.data(), static_cast<size_t>(raw_sig.size()));
#else
            explicit_bzero(raw_sig.data(), static_cast<size_t>(raw_sig.size()));
#endif
        } else if (r == FIDO_ERR_TX || r == FIDO_ERR_RX) {
            // Blind Spot 4: YubiKey yanked mid-assertion — abort USB IRP
            fido_dev_cancel(dev);
            signature = QString("Hardware token disconnected mid-assertion. Code: %1").arg(r);
        } else {
            signature = QString("Hardware assertion failed. Code: %1").arg(r);
        }

        // Strict C-API memory management
        fido_assert_free(&assert_obj);
        fido_dev_close(dev);
        fido_dev_free(&dev);

        return std::make_pair(r == FIDO_OK, signature);
    });

    // Marshal the async result back to the Qt GUI thread
    auto* watcher = new QFutureWatcher<std::pair<bool, QString>>(this);
    connect(watcher, &QFutureWatcher<std::pair<bool, QString>>::finished,
            this, [this, watcher]() {
        auto result = watcher->result();
        m_waiting = false;
        Q_EMIT waitingChanged();

        // §4.3: Emit by value then obliterate source signature from heap
        Q_EMIT signatureComplete(result.first, result.second, result.first ? "" : result.second);
#ifdef _WIN32
        if (result.first) {
            auto& sig = const_cast<QString&>(result.second);
            SecureZeroMemory(sig.data(), static_cast<size_t>(sig.capacity()) * sizeof(QChar));
        }
#endif
        watcher->deleteLater();
    });

    watcher->setFuture(future);
#else
    Q_UNUSED(strategyGitHash);
    Q_EMIT signatureComplete(false, "", "FIDO2 not available (libfido2 not linked)");
#endif
}

void FidoGateway::cancelRequest() {
    m_cancelRequested = true;
    // Note: fido_dev_get_assert has no cancellation API — 
    // the request will complete or timeout naturally
}

void FidoGateway::doRequestSignature(
    const std::string& challenge,
    const std::string& devicePath
) {
    Q_UNUSED(challenge);
    Q_UNUSED(devicePath);
    // Legacy method — requestSignature() now handles everything via QtConcurrent
}

std::string FidoGateway::encodeBase64(const unsigned char* data, size_t len) {
    // Using QByteArray::toBase64 now, but keeping for API compat
    QByteArray raw(reinterpret_cast<const char*>(data), static_cast<qsizetype>(len));
    return raw.toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals).toStdString();
}

} // namespace algae::hardware
