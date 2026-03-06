// ─────────────────────────────────────────────────────────────────────
// BuildEnvironment — QML Singleton exposing compile-time metadata
//
// QML_ELEMENT + QML_SINGLETON macros are CRITICAL: they force the
// linker to preserve MOC metadata tables even under aggressive LTO.
// Without these, the MSVC linker strips the getter methods since
// they are only invoked dynamically via QML string lookups.
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QString>
#include <QtQml/qqmlregistration.h>

#include "build_config.h"

namespace algae::config {

class BuildEnvironment : public QObject {
    Q_OBJECT
    QML_ELEMENT
    QML_SINGLETON

    Q_PROPERTY(QString buildVersion READ buildVersion CONSTANT)
    Q_PROPERTY(QString compilerId READ compilerId CONSTANT)
    Q_PROPERTY(QString cxxStandard READ cxxStandard CONSTANT)
    Q_PROPERTY(QString arrowVersion READ arrowVersion CONSTANT)
    Q_PROPERTY(QString protobufVersion READ protobufVersion CONSTANT)
    Q_PROPERTY(QString qtVersion READ qtVersion CONSTANT)
    Q_PROPERTY(QString buildType READ buildType CONSTANT)

public:
    static BuildEnvironment* instance() {
        static BuildEnvironment env;
        return &env;
    }

    QString buildVersion()    const { return QString::fromLatin1(BUILD_VERSION); }
    QString compilerId()      const { return QString::fromLatin1(BUILD_COMPILER_ID); }
    QString cxxStandard()     const { return QStringLiteral("C++") + QString::fromLatin1(BUILD_CXX_STANDARD); }
    QString arrowVersion()    const { return QString::fromLatin1(BUILD_ARROW_VERSION); }
    QString protobufVersion() const { return QString::fromLatin1(BUILD_PROTOBUF_VERSION); }
    QString qtVersion()       const { return QString::fromLatin1(BUILD_QT_VERSION); }
    QString buildType()       const { return QString::fromLatin1(BUILD_TYPE); }

private:
    explicit BuildEnvironment(QObject* parent = nullptr) : QObject(parent) {}
};

} // namespace algae::config
