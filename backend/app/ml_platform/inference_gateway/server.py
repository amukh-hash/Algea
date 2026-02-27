from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

from ..config import MLPlatformConfig
from ..models.chronos2.loader import Chronos2Loader
from ..models.chronos2.service import Chronos2Service
from ..models.chronos2.types import TSFMRequest
from ..models.selector_smoe.loader import SMoELoader
from ..models.selector_smoe.service import SMoEService
from ..models.selector_smoe.types import SMoERankRequest
from ..models.vol_surface.loader import VolSurfaceLoader
from ..models.vol_surface.service import VolSurfaceService
from ..models.vol_surface.types import VolSurfaceRequest
from ..models.itransformer.loader import ITransformerLoader
from ..models.itransformer.service import ITransformerService
from ..models.itransformer.types import ITransformerSignalRequest
from ..registry.store import ModelRegistryStore
from .health import health_payload
from .protocol import InferenceRequestBase, InferenceResponse


class InferenceGatewayServer:
    def __init__(self, cfg: MLPlatformConfig | None = None):
        self.cfg = cfg or MLPlatformConfig()
        self._handlers: dict[str, Callable[[InferenceRequestBase], dict]] = {}
        self.latency_p95_ms: dict[str, float] = {}
        self.endpoints: dict[str, bool] = {}
        self.model_status: dict[str, dict[str, str]] = {}

        store = ModelRegistryStore(self.cfg.registry_db_path, self.cfg.model_root)

        self.chronos2_service = Chronos2Service(Chronos2Loader(store), self.cfg.trace_root)
        try:
            self.chronos2_service.load_alias("prod")
            self.model_status["chronos2:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["chronos2:prod"] = {"status": f"error:{exc}"}

        self.smoe_service = SMoEService(SMoELoader(store), self.cfg.trace_root)
        try:
            self.smoe_service.load_alias("prod")
            self.model_status["selector_smoe:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["selector_smoe:prod"] = {"status": f"optional_missing:{exc}"}

        self.vol_surface_service = VolSurfaceService(VolSurfaceLoader(store), self.cfg.trace_root)
        try:
            self.vol_surface_service.load_alias("prod")
            self.model_status["vol_surface:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["vol_surface:prod"] = {"status": f"optional_missing:{exc}"}


        self.itransformer_service = ITransformerService(ITransformerLoader(store), self.cfg.trace_root)
        try:
            self.itransformer_service.load_alias("prod")
            self.model_status["itransformer:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["itransformer:prod"] = {"status": f"optional_missing:{exc}"}

        self.register("chronos2_forecast", self._chronos2_handler)
        self.register("smoe_rank", self._smoe_handler)
        self.register("vol_surface_forecast", self._vol_surface_handler)
        self.register("itransformer_signal", self._itransformer_handler)

    def _chronos2_handler(self, req: InferenceRequestBase) -> dict:
        payload = TSFMRequest(**req.payload)
        response = self.chronos2_service.forecast(payload)
        return {
            "model_name": "chronos2",
            "model_version": response.model_version,
            "outputs": {"forecast": response.forecast, "uncertainty": response.uncertainty},
            "uncertainty": response.uncertainty.get("iqr_mean", 0.0),
            "calibration_score": response.calibration_score or 0.0,
            "ood_score": response.ood_score or 0.0,
            "warnings": response.warnings,
        }

    def _smoe_handler(self, req: InferenceRequestBase) -> dict:
        payload = SMoERankRequest(**req.payload)
        response = self.smoe_service.rank(payload)
        return {
            "model_name": "selector_smoe",
            "model_version": response.model_version,
            "outputs": {
                "scores": response.scores,
                "router_entropy_mean": response.router_entropy_mean,
                "expert_utilization": response.expert_utilization,
                "load_balance_score": response.load_balance_score,
            },
            "uncertainty": 0.0,
            "calibration_score": 0.7,
            "ood_score": 0.0,
            "warnings": response.warnings,
        }


    def _itransformer_handler(self, req: InferenceRequestBase) -> dict:
        payload = ITransformerSignalRequest(**req.payload)
        response = self.itransformer_service.signal(payload)
        return {
            "model_name": "itransformer",
            "model_version": response.model_version,
            "outputs": {
                "scores": response.scores,
                "uncertainty": response.uncertainty,
                "correlation_regime": response.correlation_regime,
            },
            "uncertainty": response.uncertainty,
            "calibration_score": 0.7,
            "ood_score": 0.0,
            "warnings": response.warnings,
        }

    def _vol_surface_handler(self, req: InferenceRequestBase) -> dict:
        payload = VolSurfaceRequest(**req.payload)
        response = self.vol_surface_service.forecast(payload)
        return {
            "model_name": "vol_surface",
            "model_version": response.model_version,
            "outputs": {
                "predicted_rv": response.predicted_rv,
                "uncertainty": response.uncertainty,
                "drift_score": response.drift_score,
            },
            "uncertainty": sum(response.uncertainty.values()) / max(len(response.uncertainty), 1),
            "calibration_score": 0.7,
            "ood_score": response.ood_score,
            "warnings": response.warnings,
        }

    def register(self, endpoint: str, handler: Callable[[InferenceRequestBase], dict]) -> None:
        self._handlers[endpoint] = handler
        self.endpoints[endpoint] = True

    def infer(self, endpoint: str, req: InferenceRequestBase) -> InferenceResponse:
        if endpoint not in self._handlers:
            raise KeyError(f"unregistered endpoint: {endpoint}")
        start = time.perf_counter()
        out = self._handlers[endpoint](req)
        elapsed = (time.perf_counter() - start) * 1000
        self.latency_p95_ms[endpoint] = elapsed
        return InferenceResponse(
            model_name=out.get("model_name", endpoint),
            model_version=out.get("model_version", "unknown"),
            outputs=out.get("outputs", {}),
            uncertainty=float(out.get("uncertainty", 1.0)),
            calibration_score=float(out.get("calibration_score", 0.0)),
            ood_score=float(out.get("ood_score", 0.0)),
            latency_ms=elapsed,
            warnings=out.get("warnings", []),
        )

    def get_health(self) -> dict:
        return health_payload(self.endpoints, self.latency_p95_ms, self.model_status)

    def get_ready(self) -> dict:
        payload = self.get_health()
        if not payload["ready"]:
            raise RuntimeError("inferenced not ready")
        return payload

    def list_models(self) -> dict:
        return self.model_status

    def chronos2_http_forecast(self, req: TSFMRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id=req.instrument_id, features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = self.infer("chronos2_forecast", wrapped)
        return {"model_name": "chronos2", "model_version": resp.model_version, "forecast": resp.outputs.get("forecast", {}), "uncertainty": resp.outputs.get("uncertainty", {}), "ood_score": resp.ood_score, "calibration_score": resp.calibration_score, "latency_ms": resp.latency_ms, "warnings": resp.warnings}

    def smoe_http_rank(self, req: SMoERankRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id="selector", features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = self.infer("smoe_rank", wrapped)
        return {"model_name": "selector_smoe", "model_version": resp.model_version, "scores": resp.outputs.get("scores", {}), "router_entropy_mean": resp.outputs.get("router_entropy_mean", 0.0), "expert_utilization": resp.outputs.get("expert_utilization", {}), "load_balance_score": resp.outputs.get("load_balance_score", 0.0), "latency_ms": resp.latency_ms, "warnings": resp.warnings}

    def vol_surface_http_forecast(self, req: VolSurfaceRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id=req.underlying_symbol, features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = self.infer("vol_surface_forecast", wrapped)
        return {
            "model_name": "vol_surface",
            "model_version": resp.model_version,
            "predicted_rv": resp.outputs.get("predicted_rv", {}),
            "uncertainty": resp.outputs.get("uncertainty", {}),
            "drift_score": resp.outputs.get("drift_score", resp.ood_score),
            "ood_score": resp.ood_score,
            "latency_ms": resp.latency_ms,
            "warnings": resp.warnings,
        }


    def itransformer_http_signal(self, req: ITransformerSignalRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id="statarb", features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = self.infer("itransformer_signal", wrapped)
        return {
            "model_name": "itransformer",
            "model_version": resp.model_version,
            "scores": resp.outputs.get("scores", {}),
            "uncertainty": resp.outputs.get("uncertainty", 0.0),
            "correlation_regime": resp.outputs.get("correlation_regime", 0.0),
            "latency_ms": resp.latency_ms,
            "warnings": resp.warnings,
        }

    def make_app(self):
        try:
            from fastapi import FastAPI, HTTPException
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("fastapi unavailable for HTTP app") from exc

        app = FastAPI(title="ML Inference Gateway")

        @app.get("/healthz")
        def _healthz() -> dict:
            return self.get_health()

        @app.get("/readyz")
        def _readyz() -> dict:
            try:
                return self.get_ready()
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        @app.get("/models")
        def _models() -> dict:
            return self.list_models()

        @app.post("/v1/chronos2/forecast")
        def _forecast(req: TSFMRequest) -> dict:
            return self.chronos2_http_forecast(req)

        @app.post("/v1/selector/smoe_rank")
        def _smoe(req: SMoERankRequest) -> dict:
            return self.smoe_http_rank(req)

        @app.post("/v1/vrp/vol_surface_forecast")
        def _vol(req: VolSurfaceRequest) -> dict:
            return self.vol_surface_http_forecast(req)

        @app.post("/v1/statarb/itransformer_signal")
        def _itra(req: ITransformerSignalRequest) -> dict:
            return self.itransformer_http_signal(req)

        return app
