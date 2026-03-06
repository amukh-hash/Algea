// ─────────────────────────────────────────────────────────────────────
// FidoGateway — FIDO2/CTAP2 hardware key authentication via libfido2
//
// For strategy promotion from shadow→live, the PM must physically
// touch their YubiKey. The strategy's Git hash is used as the
// cryptographic challenge, creating a verifiable audit trail.
//
// Uses libfido2 (NOT Botan) — Botan lacks CTAP1/CTAP2 support.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QString>

#include <functional>
#include <string>
#include <vector>

namespace algae::hardware {

/// Result of a FIDO2 assertion request
struct FidoResult {
    bool success = false;
    std::string signature_b64;    // Base64-encoded ECDSA signature
    std::string error;
    std::string device_path;      // e.g. "/dev/hidraw0" or "\\\\?\\hid#..."
};

/// FIDO2 hardware key gateway for strategy promotion authorization.
///
/// Protocol:
///   1. Enumerate connected FIDO2 devices
///   2. Set strategy Git hash as clientdata challenge
///   3. Block on physical YubiKey touch (CTAP2 up=true)
///   4. Return ECDSA signature for backend verification
///
/// Thread: Runs blocking I/O in a background thread so the QML
///         UI shows "Touch your key..." without freezing.
class FidoGateway : public QObject {
    Q_OBJECT

    Q_PROPERTY(bool available READ isAvailable NOTIFY availabilityChanged)
    Q_PROPERTY(bool waiting READ isWaiting NOTIFY waitingChanged)
    Q_PROPERTY(QString deviceName READ deviceName NOTIFY deviceChanged)

public:
    explicit FidoGateway(QObject* parent = nullptr);
    ~FidoGateway() override;

    /// Check if any FIDO2 device is connected
    bool isAvailable() const { return m_available; }

    /// True while waiting for user to touch key
    bool isWaiting() const { return m_waiting; }

    /// Human-readable device name
    QString deviceName() const { return m_deviceName; }

    /// Enumerate connected FIDO2 devices
    Q_INVOKABLE void refreshDevices();

    /// Request an assertion signature using the strategy hash as challenge.
    /// This runs on a background thread and emits signatureComplete() when done.
    Q_INVOKABLE void requestSignature(const QString& strategyGitHash);

    /// Cancel a pending signature request
    Q_INVOKABLE void cancelRequest();

Q_SIGNALS:
    void availabilityChanged();
    void waitingChanged();
    void deviceChanged();

    /// Emitted when the FIDO2 assertion completes (success or failure)
    void signatureComplete(bool success, QString signatureB64, QString error);

    /// Emitted when user is expected to touch the key
    void touchRequired();

private:
    void doRequestSignature(const std::string& challenge, const std::string& devicePath);
    std::string encodeBase64(const unsigned char* data, size_t len);

    bool m_available = false;
    bool m_waiting = false;
    QString m_deviceName;
    std::string m_devicePath;
    bool m_cancelRequested = false;
};

} // namespace algae::hardware
