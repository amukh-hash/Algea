from __future__ import annotations

from datetime import datetime

from ..models.chronos2.types import TSFMRequest, TSFMResponse
from ..models.selector_smoe.types import SMoERankRequest, SMoERankResponse
from ..models.vol_surface.types import VolSurfaceRequest, VolSurfaceResponse
from ..models.itransformer.types import ITransformerSignalRequest, ITransformerSignalResponse
from ..models.rl_policy.types import RLPolicyRequest, RLPolicyResponse
from ..models.vol_surface_grid.types import VolSurfaceGridRequest, VolSurfaceGridResponse
from .protocol import InferenceRequestBase, InferenceResponse
from .server import InferenceGatewayServer


class InferenceTimeoutError(TimeoutError):
    pass


class InferenceGatewayClient:
    def __init__(self, server: InferenceGatewayServer, timeout_ms: int = 200, endpoint_timeouts_ms: dict[str, int] | None = None):
        self.server = server
        self.timeout_ms = timeout_ms
        self.endpoint_timeouts_ms = endpoint_timeouts_ms or {}

    def call(self, endpoint: str, req: InferenceRequestBase, critical: bool = True) -> InferenceResponse | None:
        response = self.server.infer(endpoint, req)
        budget_ms = int(self.endpoint_timeouts_ms.get(endpoint, self.timeout_ms))
        if response.latency_ms > budget_ms:
            if critical:
                raise InferenceTimeoutError(
                    f"fail-closed timeout on {endpoint}: {response.latency_ms:.1f}ms > {budget_ms}ms"
                )
            return None
        return response

    def call_pinned(self, endpoint: str, req: InferenceRequestBase, model_version: str, critical: bool = True) -> InferenceResponse | None:
        req.model_version = model_version
        return self.call(endpoint, req, critical=critical)

    def chronos2_forecast(self, req: TSFMRequest, critical: bool = True) -> TSFMResponse | None:
        wrapped = InferenceRequestBase(
            asof=datetime.fromisoformat(req.asof),
            universe_id=req.instrument_id,
            features_hash="",
            model_alias=req.model_alias,
            trace_id=req.trace_id,
            payload=req.model_dump(),
        )
        response = self.call("chronos2_forecast", wrapped, critical=critical)
        if response is None:
            return None
        return TSFMResponse(
            model_version=response.model_version,
            forecast=response.outputs.get("forecast", {}),
            uncertainty=response.outputs.get("uncertainty", {}),
            ood_score=response.ood_score,
            calibration_score=response.calibration_score,
            latency_ms=response.latency_ms,
            warnings=response.warnings,
        )

    def smoe_rank(self, req: SMoERankRequest, critical: bool = True) -> SMoERankResponse | None:
        wrapped = InferenceRequestBase(
            asof=datetime.fromisoformat(req.asof),
            universe_id="selector",
            features_hash="",
            model_alias=req.model_alias,
            trace_id=req.trace_id,
            payload=req.model_dump(),
        )
        response = self.call("smoe_rank", wrapped, critical=critical)
        if response is None:
            return None
        return SMoERankResponse(
            model_version=response.model_version,
            scores=response.outputs.get("scores", {}),
            router_entropy_mean=float(response.outputs.get("router_entropy_mean", 0.0)),
            expert_utilization=response.outputs.get("expert_utilization", {}),
            load_balance_score=float(response.outputs.get("load_balance_score", 0.0)),
            context_sensitivity_score=float(response.outputs.get("context_sensitivity_score", 0.0)),
            expert_collapse_score=float(response.outputs.get("expert_collapse_score", 0.0)),
            specialization_by_bucket=response.outputs.get("specialization_by_bucket", {}),
            latency_ms=response.latency_ms,
            warnings=response.warnings,
        )

    def vol_surface_forecast(self, req: VolSurfaceRequest, critical: bool = True) -> VolSurfaceResponse | None:
        wrapped = InferenceRequestBase(
            asof=datetime.fromisoformat(req.asof),
            universe_id=req.underlying_symbol,
            features_hash="",
            model_alias=req.model_alias,
            trace_id=req.trace_id,
            payload=req.model_dump(),
        )
        response = self.call("vol_surface_forecast", wrapped, critical=critical)
        if response is None:
            return None
        return VolSurfaceResponse(
            model_version=response.model_version,
            predicted_rv={int(k): v for k, v in response.outputs.get("predicted_rv", {}).items()},
            uncertainty={int(k): float(v) for k, v in response.outputs.get("uncertainty", {}).items()},
            ood_score=response.ood_score,
            drift_score=float(response.outputs.get("drift_score", response.ood_score)),
            latency_ms=response.latency_ms,
            warnings=response.warnings,
        )


    def itransformer_signal(self, req: ITransformerSignalRequest, critical: bool = True) -> ITransformerSignalResponse | None:
        wrapped = InferenceRequestBase(
            asof=datetime.fromisoformat(req.asof),
            universe_id="statarb",
            features_hash="",
            model_alias=req.model_alias,
            trace_id=req.trace_id,
            payload=req.model_dump(),
        )
        response = self.call("itransformer_signal", wrapped, critical=critical)
        if response is None:
            return None
        return ITransformerSignalResponse(
            model_version=response.model_version,
            scores=response.outputs.get("scores", {}),
            uncertainty=float(response.outputs.get("uncertainty", 0.0)),
            correlation_regime=float(response.outputs.get("correlation_regime", 0.0)),
            latency_ms=response.latency_ms,
            warnings=response.warnings,
        )

    def rl_policy_act(self, req: RLPolicyRequest, critical: bool = True) -> RLPolicyResponse | None:
        wrapped = InferenceRequestBase(
            asof=datetime.fromisoformat(req.asof),
            universe_id=req.sleeve,
            features_hash="",
            model_alias=req.model_alias,
            trace_id=req.trace_id,
            payload=req.model_dump(),
        )
        response = self.call("rl_policy_act", wrapped, critical=critical)
        if response is None:
            return None
        return RLPolicyResponse(
            model_version=response.model_version,
            size_multiplier=float(response.outputs.get("size_multiplier", 0.0)),
            veto=bool(response.outputs.get("veto", False)),
            projected_multiplier=float(response.outputs.get("projected_multiplier", 0.0)),
            projection_reason=str(response.outputs.get("projection_reason", "")),
            projection_applied=bool(response.outputs.get("projection_applied", False)),
            drift_score=float(response.outputs.get("drift_score", response.ood_score)),
            ood_score=response.ood_score,
            latency_ms=response.latency_ms,
            warnings=response.warnings,
        )

    def vol_surface_grid_forecast(self, req: VolSurfaceGridRequest, critical: bool = True) -> VolSurfaceGridResponse | None:
        wrapped = InferenceRequestBase(
            asof=datetime.fromisoformat(req.asof),
            universe_id=req.underlying_symbol,
            features_hash="",
            model_alias=req.model_alias,
            trace_id=req.trace_id,
            payload=req.model_dump(),
        )
        response = self.call("vol_surface_grid_forecast", wrapped, critical=critical)
        if response is None:
            return None
        return VolSurfaceGridResponse(
            model_version=response.model_version,
            grid_forecast={str(k): float(v) for k, v in response.outputs.get("grid_forecast", {}).items()},
            uncertainty_proxy=float(response.outputs.get("uncertainty_proxy", 1.0)),
            drift_score=float(response.outputs.get("drift_score", response.ood_score)),
            mask_coverage=float(response.outputs.get("mask_coverage", 0.0)),
            latency_ms=response.latency_ms,
            warnings=response.warnings,
        )
