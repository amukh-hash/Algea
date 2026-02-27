from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IVGridSpec:
    tenors: tuple[int, ...] = (7, 14, 30, 60)
    buckets: tuple[str, ...] = ("10p", "25p", "ATM", "25c", "10c")


def ordered_grid_keys(spec: IVGridSpec) -> list[str]:
    return [f"{t}:{b}" for t in spec.tenors for b in spec.buckets]
