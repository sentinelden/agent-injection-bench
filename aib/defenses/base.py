from __future__ import annotations

from typing import Any, Callable, Protocol

from ..adapters.base import Scenario


class Defense(Protocol):
    name: str

    def __call__(self, scenario: Scenario) -> Scenario: ...


_REGISTRY: dict[str, Callable[..., Defense]] = {}


def register(name: str) -> Callable[[Callable[..., Defense]], Callable[..., Defense]]:
    def decorator(factory: Callable[..., Defense]) -> Callable[..., Defense]:
        _REGISTRY[name] = factory
        return factory
    return decorator


def get(name: str, **kwargs: Any) -> Defense:
    if name not in _REGISTRY:
        raise KeyError(f"unknown defense {name!r}; available: {', '.join(available())}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
