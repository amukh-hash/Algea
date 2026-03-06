"""Re-export shim: backend/app/strategies/options_vrp → algae.execution.options."""
from algae.execution.options.vrp_strategy import VRPStrategy  # noqa: F401
from algae.execution.options.config import VRPConfig  # noqa: F401
from algae.execution.options.structures import (  # noqa: F401
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
    StructureType,
)
