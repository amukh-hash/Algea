from __future__ import annotations

from typing import Dict, Type

from algea.models.foundation.base import FoundationModel


_REGISTRY: Dict[str, Type[FoundationModel]] = {}


def register(name: str, cls: Type[FoundationModel]) -> None:
    _REGISTRY[name] = cls


def get(name: str) -> Type[FoundationModel]:
    if name not in _REGISTRY:
        raise KeyError(f"Foundation model not registered: {name}")
    return _REGISTRY[name]
