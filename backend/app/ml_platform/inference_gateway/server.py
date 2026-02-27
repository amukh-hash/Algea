from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Callable, Any
from dataclasses import dataclass, field
import torch

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
from ..models.rl_policy.loader import RLPolicyLoader
from ..models.rl_policy.service import RLPolicyService
from ..models.rl_policy.types import RLPolicyRequest
from ..models.vol_surface_grid.loader import VolSurfaceGridLoader
from ..models.vol_surface_grid.service import VolSurfaceGridService
from ..models.vol_surface_grid.types import VolSurfaceGridRequest
from ..registry.store import ModelRegistryStore
from .health import health_payload
from .protocol import InferenceRequestBase, InferenceResponse



@dataclass(order=True)
class PrioritizedTask:
    priority: int
    req: InferenceRequestBase = field(compare=False)
    handler: Callable = field(compare=False)
    future: asyncio.Future = field(compare=False)
    endpoint_name: str = field(compare=False)

DEVICE_HEAVY = torch.device("cuda:0")
DEVICE_FAST = torch.device("cuda:1")

class InferenceGatewayServer:
    def __init__(self, cfg: MLPlatformConfig | None = None):
        self.cfg = cfg or MLPlatformConfig()
        self._handlers: dict[str, Callable[[InferenceRequestBase], dict]] = {}
        self.latency_p95_ms: dict[str, float] = {}
        self.endpoints: dict[str, bool] = {}
        self.model_status: dict[str, dict[str, str]] = {}

        store = ModelRegistryStore(self.cfg.registry_db_path, self.cfg.model_root)

        # -------------------------------------------------------------
        # Phase 2: Explicit Hardware Registries 
        # -------------------------------------------------------------
        self.device_heavy = DEVICE_HEAVY # RTX 3090 Ti
        self.device_fast = DEVICE_FAST   # RTX 4070 Super

        self.queue_heavy: asyncio.PriorityQueue | None = None
        self.queue_fast: asyncio.PriorityQueue | None = None

        # cuda:1 Workloads (FAST)
        self.chronos2_service = Chronos2Service(Chronos2Loader(store), self.cfg.trace_root, device=str(self.device_fast))
        try:
            self.chronos2_service.load_alias("prod")
            self.model_status["chronos2:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["chronos2:prod"] = {"status": f"error:{exc}"}

        self.smoe_service = SMoEService(SMoELoader(store), self.cfg.trace_root, device=str(self.device_fast))
        try:
            self.smoe_service.load_alias("prod")
            self.model_status["selector_smoe:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["selector_smoe:prod"] = {"status": f"optional_missing:{exc}"}

        self.vol_surface_service = VolSurfaceService(VolSurfaceLoader(store), self.cfg.trace_root, device=str(self.device_fast))
        try:
            self.vol_surface_service.load_alias("prod")
            self.model_status["vol_surface:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["vol_surface:prod"] = {"status": f"optional_missing:{exc}"}

        # cuda:0 Workloads (HEAVY)
        self.itransformer_service = ITransformerService(ITransformerLoader(store), self.cfg.trace_root, device=str(self.device_heavy))
        try:
            self.itransformer_service.load_alias("prod")
            self.model_status["itransformer:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["itransformer:prod"] = {"status": f"optional_missing:{exc}"}

        self.register("chronos2_forecast", self._chronos2_handler)
        self.register("smoe_rank", self._smoe_handler)

        # cuda:1 Workloads (FAST)
        self.rl_policy_service = RLPolicyService(RLPolicyLoader(store), self.cfg.trace_root, device=str(self.device_fast))
        try:
            self.rl_policy_service.load_alias("prod")
            self.model_status["rl_policy:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["rl_policy:prod"] = {"status": f"optional_missing:{exc}"}

        self.register("vol_surface_forecast", self._vol_surface_handler)
        self.vol_surface_grid_service = VolSurfaceGridService(VolSurfaceGridLoader(store), self.cfg.trace_root, device=str(self.device_fast))
        try:
            self.vol_surface_grid_service.load_alias("prod")
            self.model_status["vol_surface_grid:prod"] = {"status": "loaded"}
        except Exception as exc:
            self.model_status["vol_surface_grid:prod"] = {"status": f"optional_missing:{exc}"}
        self.register("vol_surface_grid_forecast", self._vol_surface_grid_handler)
        self.register("itransformer_signal", self._itransformer_handler)
        self.register("rl_policy_act", self._rl_policy_handler)

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
                "context_sensitivity_score": response.context_sensitivity_score,
                "expert_collapse_score": response.expert_collapse_score,
                "specialization_by_bucket": response.specialization_by_bucket,
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


    def _rl_policy_handler(self, req: InferenceRequestBase) -> dict:
        payload = RLPolicyRequest(**req.payload)
        response = self.rl_policy_service.policy_act(payload)
        return {
            "model_name": "rl_policy",
            "model_version": response.model_version,
            "outputs": {
                "size_multiplier": response.size_multiplier,
                "veto": response.veto,
                "projected_multiplier": response.projected_multiplier,
                "projection_reason": response.projection_reason,
                "projection_applied": response.projection_applied,
                "drift_score": response.drift_score,
            },
            "uncertainty": 1.0 - response.projected_multiplier,
            "calibration_score": 0.7,
            "ood_score": response.ood_score,
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

    def _vol_surface_grid_handler(self, req: InferenceRequestBase) -> dict:
        payload = VolSurfaceGridRequest(**req.payload)
        response = self.vol_surface_grid_service.forecast(payload)
        return {
            "model_name": "vol_surface_grid",
            "model_version": response.model_version,
            "outputs": {
                "grid_forecast": response.grid_forecast,
                "mask_coverage": response.mask_coverage,
                "uncertainty_proxy": response.uncertainty_proxy,
                "drift_score": response.drift_score,
            },
            "uncertainty": response.uncertainty_proxy,
            "calibration_score": 0.7,
            "ood_score": response.drift_score,
            "warnings": response.warnings,
        }

    def register(self, endpoint: str, handler: Callable[[InferenceRequestBase], dict]) -> None:
        self._handlers[endpoint] = handler
        self.endpoints[endpoint] = True

    def infer(self, endpoint: str, req: InferenceRequestBase) -> InferenceResponse:
        """Fallback synchronous method if needed."""
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

    async def async_worker(self, queue: asyncio.PriorityQueue, device_name: str) -> None:
        """Dedicated async worker loop isolating standard ThreadPool processing per GPU queue."""
        loop = asyncio.get_running_loop()
        while True:
            task: PrioritizedTask = await queue.get()
            try:
                start = time.perf_counter()
                out = await loop.run_in_executor(None, task.handler, task.req)
                elapsed = (time.perf_counter() - start) * 1000
                self.latency_p95_ms[task.endpoint_name] = elapsed
                
                resp = InferenceResponse(
                    model_name=out.get("model_name", task.endpoint_name),
                    model_version=out.get("model_version", "unknown"),
                    outputs=out.get("outputs", {}),
                    uncertainty=float(out.get("uncertainty", 1.0)),
                    calibration_score=float(out.get("calibration_score", 0.0)),
                    ood_score=float(out.get("ood_score", 0.0)),
                    latency_ms=elapsed,
                    warnings=out.get("warnings", []),
                )
                if not task.future.done():
                    task.future.set_result(resp)
            except Exception as e:
                if not task.future.done():
                    task.future.set_exception(e)
            finally:
                queue.task_done()

    async def async_infer(self, endpoint: str, req: InferenceRequestBase, priority: int = 1) -> InferenceResponse:
        """Non-blocking Priority submission wrapping."""
        if endpoint not in self._handlers:
            raise KeyError(f"unregistered endpoint: {endpoint}")
            
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Route logic
        if endpoint in ["itransformer_signal"]:
            queue = self.queue_heavy
        else:
            queue = self.queue_fast
            
        if queue is None:
            # Fallback for sync contexts outside of FastAPI loop
            return self.infer(endpoint, req)
            
        ptask = PrioritizedTask(
            priority=priority,
            req=req,
            handler=self._handlers[endpoint],
            future=future,
            endpoint_name=endpoint
        )
        await queue.put(ptask)
        return await future

    def get_health(self) -> dict:
        return health_payload(self.endpoints, self.latency_p95_ms, self.model_status)

    def get_ready(self) -> dict:
        payload = self.get_health()
        if not payload["ready"]:
            raise RuntimeError("inferenced not ready")
        return payload

    def list_models(self) -> dict:
        return self.model_status

    async def chronos2_http_forecast(self, req: TSFMRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id=req.instrument_id, features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = await self.async_infer("chronos2_forecast", wrapped, priority=1)
        return {"model_name": "chronos2", "model_version": resp.model_version, "forecast": resp.outputs.get("forecast", {}), "uncertainty": resp.outputs.get("uncertainty", {}), "ood_score": resp.ood_score, "calibration_score": resp.calibration_score, "latency_ms": resp.latency_ms, "warnings": resp.warnings}

    async def smoe_http_rank(self, req: SMoERankRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id="selector", features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = await self.async_infer("smoe_rank", wrapped, priority=1)
        return {"model_name": "selector_smoe", "model_version": resp.model_version, "scores": resp.outputs.get("scores", {}), "router_entropy_mean": resp.outputs.get("router_entropy_mean", 0.0), "expert_utilization": resp.outputs.get("expert_utilization", {}), "load_balance_score": resp.outputs.get("load_balance_score", 0.0), "context_sensitivity_score": resp.outputs.get("context_sensitivity_score", 0.0), "expert_collapse_score": resp.outputs.get("expert_collapse_score", 0.0), "specialization_by_bucket": resp.outputs.get("specialization_by_bucket", {}), "latency_ms": resp.latency_ms, "warnings": resp.warnings}

    async def vol_surface_http_forecast(self, req: VolSurfaceRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id=req.underlying_symbol, features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = await self.async_infer("vol_surface_forecast", wrapped, priority=2)
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



    async def rl_policy_http_act(self, req: RLPolicyRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id=req.sleeve, features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = await self.async_infer("rl_policy_act", wrapped, priority=1)
        return {
            "model_name": "rl_policy",
            "model_version": resp.model_version,
            "size_multiplier": resp.outputs.get("size_multiplier", 0.0),
            "veto": resp.outputs.get("veto", False),
            "projected_multiplier": resp.outputs.get("projected_multiplier", 0.0),
            "projection_reason": resp.outputs.get("projection_reason", ""),
            "projection_applied": resp.outputs.get("projection_applied", False),
            "drift_score": resp.outputs.get("drift_score", resp.ood_score),
            "ood_score": resp.ood_score,
            "latency_ms": resp.latency_ms,
            "warnings": resp.warnings,
        }

    async def itransformer_http_signal(self, req: ITransformerSignalRequest) -> dict:
        wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id="statarb", features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
        resp = await self.async_infer("itransformer_signal", wrapped, priority=5)
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

        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self.queue_heavy = asyncio.PriorityQueue()
            self.queue_fast = asyncio.PriorityQueue()
            task_heavy = asyncio.create_task(self.async_worker(self.queue_heavy, "cuda:0"))
            task_fast = asyncio.create_task(self.async_worker(self.queue_fast, "cuda:1"))
            yield
            task_heavy.cancel()
            task_fast.cancel()

        app = FastAPI(title="ML Inference Gateway", lifespan=lifespan)

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
        async def _forecast(req: TSFMRequest) -> dict:
            return await self.chronos2_http_forecast(req)

        @app.post("/v1/selector/smoe_rank")
        async def _smoe(req: SMoERankRequest) -> dict:
            return await self.smoe_http_rank(req)

        @app.post("/v1/vrp/vol_surface_forecast")
        async def _vol(req: VolSurfaceRequest) -> dict:
            return await self.vol_surface_http_forecast(req)

        @app.post("/v1/vrp/vol_surface_grid_forecast")
        async def _vol_grid(req: VolSurfaceGridRequest) -> dict:
            wrapped = InferenceRequestBase(asof=datetime.fromisoformat(req.asof), universe_id=req.underlying_symbol, features_hash="", model_alias=req.model_alias, trace_id=req.trace_id, payload=req.model_dump())
            resp = await self.async_infer("vol_surface_grid_forecast", wrapped, priority=2)
            return {"model_name": "vol_surface_grid", "model_version": resp.model_version, "grid_forecast": resp.outputs.get("grid_forecast", {}), "mask_coverage": resp.outputs.get("mask_coverage", 0.0), "uncertainty_proxy": resp.outputs.get("uncertainty_proxy", 1.0), "drift_score": resp.outputs.get("drift_score", resp.ood_score), "latency_ms": resp.latency_ms, "warnings": resp.warnings}

        @app.post("/v1/statarb/itransformer_signal")
        async def _itra(req: ITransformerSignalRequest) -> dict:
            return await self.itransformer_http_signal(req)

        @app.post("/v1/rl/policy_act")
        async def _rl(req: RLPolicyRequest) -> dict:
            return await self.rl_policy_http_act(req)

        return app
