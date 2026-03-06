// ─────────────────────────────────────────────────────────────────────
// KillSwitchBridge — QObject wrapper exposing KillSwitch to QML
//
// Thin delegation layer: all SHM atomics remain in KillSwitch.cpp.
// This class merely adds Q_OBJECT, Q_INVOKABLE, and Q_PROPERTY macros
// so QML Switch toggles can invoke haltSleeve/resumeSleeve directly.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QString>
#include <QtQml/qqmlregistration.h>

#include "KillSwitch.h"

namespace algae::hardware {

class KillSwitchBridge : public QObject {
    Q_OBJECT

    Q_PROPERTY(uint haltMask READ haltMask NOTIFY haltMaskChanged)

public:
    explicit KillSwitchBridge(KillSwitch* backend, QObject* parent = nullptr)
        : QObject(parent), m_backend(backend) {}

    uint haltMask() const { return m_backend ? m_backend->haltMask() : 0; }

    Q_INVOKABLE void haltSleeve(int sleeveId, const QString& reason) {
        if (!m_backend) return;
        m_backend->haltSleeve(static_cast<uint32_t>(sleeveId), reason.toStdString());
        Q_EMIT haltMaskChanged();
    }

    Q_INVOKABLE void resumeSleeve(int sleeveId) {
        if (!m_backend) return;
        m_backend->resumeSleeve(static_cast<uint32_t>(sleeveId));
        Q_EMIT haltMaskChanged();
    }

    Q_INVOKABLE void resumeAll() {
        if (!m_backend) return;
        m_backend->resumeAll();
        Q_EMIT haltMaskChanged();
    }

    Q_INVOKABLE bool isSleeveHalted(int sleeveId) const {
        if (!m_backend) return false;
        return m_backend->isSleeveHalted(static_cast<uint32_t>(sleeveId));
    }

Q_SIGNALS:
    void haltMaskChanged();

private:
    KillSwitch* m_backend = nullptr;
};

} // namespace algae::hardware
