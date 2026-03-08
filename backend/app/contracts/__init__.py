"""Runtime contracts and compatibility validators for orchestrator boundaries.

Canonical contracts (Intent Supremacy) are available via ``contracts.canonical``.
Legacy Pydantic models remain in ``contracts.models`` for backward compatibility.
"""

from .models import (
    BrokerPosition,
    ExecutionResult,
    PositionDeltaPlan,
    RiskDecisionReport,
    SleeveOutput,
)
from .validators import (
    normalize_broker_positions_payload,
    validate_execution_result,
    validate_position_delta_plan,
    validate_risk_decision_report,
    validate_sleeve_output,
)

# Canonical Intent Supremacy contracts
from .canonical import (
    AssetClass,
    ExecutionPhase as CanonicalExecutionPhase,
    PlannedOrder,
    PositionDeltaPlan as CanonicalPositionDeltaPlan,
    RiskDecisionReport as CanonicalRiskDecisionReport,
    SleeveDecision,
    SleeveStatus,
    TargetIntent as CanonicalTargetIntent,
    TraceRef,
    Violation,
)
from .providers import (
    ControlSnapshot,
    MarketDataProvider,
    MarketDataSnapshot,
    PortfolioStateProvider,
    PortfolioStateSnapshot,
    Position,
    Sleeve,
)
from .errors import (
    ContractViolationError,
    DataStaleError,
    DataUnavailableError,
    InvalidConfigError,
    PlanningError,
    RoutingError,
    SleeveError,
)

__all__ = [
    # Legacy Pydantic models
    "SleeveOutput",
    "RiskDecisionReport",
    "PositionDeltaPlan",
    "BrokerPosition",
    "ExecutionResult",
    "validate_sleeve_output",
    "validate_risk_decision_report",
    "validate_position_delta_plan",
    "normalize_broker_positions_payload",
    "validate_execution_result",
    # Canonical contracts
    "SleeveStatus",
    "AssetClass",
    "CanonicalExecutionPhase",
    "TraceRef",
    "CanonicalTargetIntent",
    "SleeveDecision",
    "Violation",
    "CanonicalRiskDecisionReport",
    "PlannedOrder",
    "CanonicalPositionDeltaPlan",
    # Providers
    "MarketDataSnapshot",
    "PortfolioStateSnapshot",
    "ControlSnapshot",
    "Position",
    "MarketDataProvider",
    "PortfolioStateProvider",
    "Sleeve",
    # Errors
    "SleeveError",
    "DataUnavailableError",
    "DataStaleError",
    "InvalidConfigError",
    "ContractViolationError",
    "PlanningError",
    "RoutingError",
]
