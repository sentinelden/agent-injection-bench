"""agent-injection-bench: an open benchmark for prompt injection against
tool-calling agents."""

__version__ = "0.1.0"

from .dataset import Attack, load
from .harness import Result, Summary, run, run_one, summarise
from .scoring import AgentResponse, Outcome, ToolCall, judge

__all__ = [
    "Attack", "load", "Result", "Summary", "run", "run_one", "summarise",
    "AgentResponse", "Outcome", "ToolCall", "judge", "__version__",
]
