from __future__ import annotations

import random


def seed_all(seed: int) -> random.Random:
    rng = random.Random(seed)
    return rng
