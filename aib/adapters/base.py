from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..dataset import Attack, Tool
from ..scoring import AgentResponse


@dataclass
class Scenario:
    """One fully-assembled attack, ready to hand to a target.

    Assembled by the harness so that every adapter sees the same thing: the
    defense under test has already been applied, and the hostile content has
    already been placed in its channel.
    """

    attack: Attack
    system_prompt: str
    messages: list[dict[str, str]]
    tools: list[Tool]
    # Tool name -> canned result. The harness answers tool calls from here
    # rather than executing anything, so running the benchmark has no side
    # effects and is reproducible.
    tool_results: dict[str, str] = field(default_factory=dict)


class Adapter(Protocol):
    """Run one scenario and report what the agent did."""

    name: str

    def __call__(self, scenario: Scenario) -> AgentResponse: ...


_REGISTRY: dict[str, Callable[..., Adapter]] = {}


def register(name: str) -> Callable[[Callable[..., Adapter]], Callable[..., Adapter]]:
    def decorator(factory: Callable[..., Adapter]) -> Callable[..., Adapter]:
        _REGISTRY[name] = factory
        return factory
    return decorator


def get(name: str, **kwargs: Any) -> Adapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown target {name!r}; available: {', '.join(available())}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
