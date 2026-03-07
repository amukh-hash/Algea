from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, get_type_hints


@dataclass
class _FieldInfo:
    default: Any
    ge: float | None = None
    le: float | None = None
    gt: float | None = None
    description: str | None = None
    default_factory: Any = None


def Field(default: Any = ..., *, ge: float | None = None, le: float | None = None, gt: float | None = None, description: str | None = None, default_factory=None):
    return _FieldInfo(default=default, ge=ge, le=le, gt=gt, description=description, default_factory=default_factory)


def field_validator(field_name: str):
    def _dec(fn):
        setattr(fn, "_pyd_field_name", field_name)
        return fn

    return _dec


class _BaseModelMeta(type):
    def __new__(mcls, name, bases, attrs):
        validators: dict[str, list] = {}
        for v in attrs.values():
            if isinstance(v, classmethod):
                f = getattr(v, "_pyd_field_name", None) or getattr(v.__func__, "_pyd_field_name", None)
                if f:
                    validators.setdefault(f, []).append(v.__func__)
                continue
            field_name = getattr(v, "_pyd_field_name", None)
            if field_name:
                validators.setdefault(field_name, []).append(v)
        attrs["__validators__"] = validators
        return super().__new__(mcls, name, bases, attrs)


class BaseModel(metaclass=_BaseModelMeta):
    def __init__(self, **kwargs: Any):
        ann = get_type_hints(type(self))
        for key, typ in ann.items():
            class_default = getattr(type(self), key, ...)
            if key in kwargs:
                value = kwargs[key]
            elif isinstance(class_default, _FieldInfo):
                if class_default.default_factory is not None:
                    value = class_default.default_factory()
                elif class_default.default is ...:
                    raise ValueError(f"Missing field: {key}")
                else:
                    value = class_default.default
            elif class_default is not ...:
                value = class_default
            else:
                raise ValueError(f"Missing field: {key}")

            value = self._coerce_enum(typ, value)
            self._apply_field_constraints(key, class_default, value)
            for validator in getattr(type(self), "__validators__", {}).get(key, []):
                value = validator(type(self), value)
            setattr(self, key, value)

    @staticmethod
    def _coerce_enum(typ, value):
        if isinstance(value, Enum):
            return value
        if isinstance(typ, type) and issubclass(typ, Enum):
            return typ(value)
        return value

    @staticmethod
    def _apply_field_constraints(key: str, class_default: Any, value: Any) -> None:
        if not isinstance(class_default, _FieldInfo):
            return
        if class_default.ge is not None and value < class_default.ge:
            raise ValueError(f"{key} must be >= {class_default.ge}")
        if class_default.le is not None and value > class_default.le:
            raise ValueError(f"{key} must be <= {class_default.le}")
        if class_default.gt is not None and value <= class_default.gt:
            raise ValueError(f"{key} must be > {class_default.gt}")

    def model_dump(self) -> dict[str, Any]:
        ann = get_type_hints(type(self))
        return {k: getattr(self, k) for k in ann}

    @classmethod
    def model_validate(cls, payload: dict[str, Any]):
        return cls(**payload)


class ConfigDict(dict):
    pass
