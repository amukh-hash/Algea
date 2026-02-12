"""Re-export shim: backend/app/strategies/options_vrp → algaie.execution.options."""
from algaie.execution.options.vrp_strategy import VRPStrategy  # noqa: F401
from algaie.execution.options.config import VRPConfig  # noqa: F401
from algaie.execution.options.structures import (  # noqa: F401
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
    StructureType,
)
