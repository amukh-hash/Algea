
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class UniverseRules:
    """
    Configuration for Rolling UniverseFrame generation.
    Defines thresholds for Observable and Tradable universes.
    """
    # Windows
    adv20_window: int = 20
    adv60_window: int = 60
    vol20_window: int = 20
    missing_window: int = 60
    spike_window: int = 20

    # Observable Universe (Quality/Liquidity/History)
    min_price_observable: float = 5.0
    min_age_observable: int = 126
    min_adv20_observable: float = 10e6
    min_adv60_observable: float = 5e6
    max_missing_60_observable: int = 2
    max_zero_volume_60_observable: int = 2

    # Tradable Universe (Strict Extraction)
    min_price_tradable: float = 7.0
    min_age_tradable: int = 252
    min_adv20_tradable: float = 25e6
    min_adv60_tradable: float = 15e6
    max_missing_60_tradable: int = 0
    max_zero_volume_60_tradable: int = 0
    max_share20_tradable: float = 0.30

    # Tiering
    tier_a_min_adv20: float = 100e6
    tier_b_min_adv20: float = 40e6
    
    # Weighting
    weight_min_adv20: float = 25e6
    weight_max_adv20: float = 500e6

    # Asset Type Filter
    # Exclude if name contains ANY of these (case-insensitive)
    exclude_name_patterns: List[str] = field(default_factory=lambda: [
        " ETF", " TRUST", " DEPOSITARY", " ADR", " ETN", " WARRANT", " PREFERRED", " L.P."
    ])
    
    # Validation
    def __post_init__(self):
        assert self.min_price_tradable >= self.min_price_observable
        assert self.min_age_tradable >= self.min_age_observable
        assert self.min_adv20_tradable >= self.min_adv20_observable
