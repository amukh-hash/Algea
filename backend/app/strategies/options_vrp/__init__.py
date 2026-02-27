"""Re-export shim: backend/app/strategies/options_vrp → algea.execution.options."""
from algea.execution.options.vrp_strategy import VRPStrategy  # noqa: F401
from algea.execution.options.config import VRPConfig  # noqa: F401
from algea.execution.options.structures import (  # noqa: F401
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
    StructureType,
)
