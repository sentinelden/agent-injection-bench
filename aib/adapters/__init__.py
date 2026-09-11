"""Target adapters.

An adapter is anything that can take a scenario and return an AgentResponse.
The benchmark deliberately knows nothing about any particular provider: a
target is a callable, and the repository ships two reference implementations
(a deterministic local one, and a generic OpenAI-compatible HTTP client that
most vendors and local servers speak).

Keeping the interface this small is what lets the benchmark measure systems
rather than APIs. The interesting subject of an agent-security benchmark is
usually not a bare model but a *stack* -- model plus system prompt plus tool
registry plus whatever filtering sits in between. Any of those can be an
adapter.
"""

from .base import Adapter, Scenario, register, get, available

__all__ = ["Adapter", "Scenario", "register", "get", "available"]
