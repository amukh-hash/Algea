// ─────────────────────────────────────────────────────────────────────
// RestClient — Qt Network HTTP client for bootstrap/historical queries
//
// Matches the implementation manual's heavily templated QNetworkAccessManager
// pattern to unify JSON deserialization into strictly typed C++ structs
// while ensuring proper KEEP-ALIVE HTTP semantics to prevent TIME_WAIT 
// socket exhaustion (Blind Spot 1).
// ─────────────────────────────────────────────────────────────────────
#pragma once

#include <QObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QJsonDocument>
#include <functional>
#include <string>

#include <google/protobuf/message.h>
#include <google/protobuf/util/json_util.h>

// Forward declarations for structs if not available
namespace algae::portfolio {
    class PortfolioSummary;
    class RiskChecksReport;
}

namespace algae::network {

class RestClient : public QObject {
    Q_OBJECT

public:
    explicit RestClient(const QString& baseUrl = "http://127.0.0.1:8000", QObject *parent = nullptr);
    
    // Core Endpoints demonstrated in the manual
    Q_INVOKABLE void getPortfolioSummary();
    Q_INVOKABLE void getRiskChecks();
    Q_INVOKABLE void getControlState();
    Q_INVOKABLE void getJobGraph();
    Q_INVOKABLE void getBrokerStatus();
    Q_INVOKABLE void getGuardrailStatus();

Q_SIGNALS:
    // We Q_EMIT pointers to avoid deep copying large protobufs, or refs based on usage. 
    // Since this is QML/C++ bridge, typically we Q_EMIT QVariant, but keeping to manual:
    // Note: Manual uses const algae::portfolio::PortfolioSummary&. In standard C++ we'd just declare it 
    // assuming they are generated. For compilation safety we use google::protobuf::Message
    // if the real headers aren't included, but we'll include dummy or generated ones.
    void portfolioSummaryReceived(const std::string& debug_str);
    void riskChecksReceived(const std::string& debug_str);
    void controlStateReceived(const std::string& debug_str);
    void jobGraphReceived(const std::string& debug_str);
    void brokerStatusReceived(const std::string& debug_str);
    void guardrailStatusReceived(const std::string& debug_str);
    void networkError(const QString& endpoint, const QString& errorMessage);

private:
    QNetworkAccessManager* m_nam;
    QString m_base_url;

    // C++20 Templated Dispatcher
    template<typename ProtoMsg>
    void fetchAndParse(const QString& endpoint, std::function<void(const ProtoMsg&)> onSuccess) {
        QNetworkRequest request(QUrl(m_base_url + endpoint));
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        
        // Prevent TIME_WAIT port exhaustion (Blind Spot 1 Mitigation)
        request.setRawHeader("Connection", "keep-alive");

        QNetworkReply* reply = m_nam->get(request);

        connect(reply, &QNetworkReply::finished, this, [this, reply, endpoint, onSuccess]() {
            if (reply->error() == QNetworkReply::NoError) {
                // Protobuf deserialization proxy
                ProtoMsg msg;
                std::string json_str = reply->readAll().toStdString();
                
                auto status = google::protobuf::util::JsonStringToMessage(json_str, &msg);
                if (status.ok()) {
                    onSuccess(msg);
                } else {
                    // Parse failure — emit error instead of corrupting state
                    // with a default-constructed empty protobuf message.
                    qWarning() << "Protobuf parse failed for" << endpoint
                               << ":" << QString::fromStdString(std::string(status.message()));
                    Q_EMIT networkError(endpoint,
                        QString("Protobuf parse failure: %1").arg(
                            QString::fromStdString(std::string(status.message()))));
                }
            } else {
                Q_EMIT networkError(endpoint, reply->errorString());
            }
            // §4.4: Use deleteLater() — we're inside a signal handler,
            // the reply may still be referenced by Qt internals.
            reply->deleteLater();
        });
    }
};

} // namespace algae::network
