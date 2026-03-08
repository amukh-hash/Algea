// ─────────────────────────────────────────────────────────────────────
// RestClient — Implementation
// ─────────────────────────────────────────────────────────────────────
#include "RestClient.h"

#include <QJsonDocument>
#include <QJsonObject>

namespace algae::network {

RestClient::RestClient(const QString& baseUrl, QObject *parent) 
    : QObject(parent)
    , m_nam(new QNetworkAccessManager(this)) 
{
    m_base_url = qEnvironmentVariable("Algae_BACKEND_URL", baseUrl);

    // Disable HTTP response caching to prevent heap fragmentation
    // during 120+ hour continuous uptime (14,000+ requests/week).
    m_nam->setCache(nullptr);

    // HSTS enforcement: prevent MITM downgrade attacks on CurveZMQ
    // key bootstrap (Blind Spot 3). Only enforced outside localhost dev.
    m_nam->setStrictTransportSecurityEnabled(true);
    if (!m_base_url.startsWith("http://localhost") &&
        !m_base_url.startsWith("http://127.0.0.1") &&
        !m_base_url.startsWith("https://")) {
        qCritical() << "RestClient: non-HTTPS base URL in production context:" << m_base_url;
    }
}

void RestClient::getPortfolioSummary() {
    QNetworkRequest req(QUrl(m_base_url + "/api/control/portfolio-summary"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    
    auto* reply = m_nam->get(req);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            Q_EMIT networkError("/api/control/portfolio-summary", reply->errorString());
            return;
        }
        Q_EMIT portfolioSummaryReceived(reply->readAll().toStdString());
    });
}

void RestClient::getRiskChecks() {
    QNetworkRequest req(QUrl(m_base_url + "/api/orchestrator/risk-checks"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    auto* reply = m_nam->get(req);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            Q_EMIT networkError("/api/orchestrator/risk-checks", reply->errorString());
            return;
        }
        Q_EMIT riskChecksReceived(reply->readAll().toStdString());
    });
}


void RestClient::getJobGraph() {
    QNetworkRequest req(QUrl(m_base_url + "/api/control/job-graph"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    auto* reply = m_nam->get(req);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            Q_EMIT networkError("/api/control/job-graph", reply->errorString());
            return;
        }
        Q_EMIT jobGraphReceived(reply->readAll().toStdString());
    });
}

void RestClient::getBrokerStatus() {
    QNetworkRequest req(QUrl(m_base_url + "/api/control/broker-status"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    auto* reply = m_nam->get(req);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            Q_EMIT networkError("/api/control/broker-status", reply->errorString());
            return;
        }
        Q_EMIT brokerStatusReceived(reply->readAll().toStdString());
    });
}

void RestClient::getGuardrailStatus() {
    QNetworkRequest req(QUrl(m_base_url + "/api/control/guardrails/status"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    auto* reply = m_nam->get(req);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            Q_EMIT networkError("/api/control/guardrails/status", reply->errorString());
            return;
        }
        Q_EMIT guardrailStatusReceived(reply->readAll().toStdString());
    });
}
void RestClient::getControlState() {
    QNetworkRequest req(QUrl(m_base_url + "/api/control/state"));
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");

    auto* reply = m_nam->get(req);
    QObject::connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            Q_EMIT networkError("/api/control/state", reply->errorString());
            return;
        }
        Q_EMIT controlStateReceived(reply->readAll().toStdString());
    });
}

} // namespace algae::network
