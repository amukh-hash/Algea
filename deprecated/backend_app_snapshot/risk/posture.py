from enum import Enum
from dataclasses import dataclass
from typing import Optional

class RiskPosture(str, Enum):
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"
