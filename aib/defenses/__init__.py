"""Defenses under test.

A defense transforms the scenario before the target sees it. Measuring them
side by side against the same corpus is the point of the benchmark: "is this
model safe" is close to unanswerable, while "does spotlighting reduce
tool_result compromise on this corpus, and what does it cost in over-refusal"
is a question with a number attached.

The defenses here are the commonly-deployed ones, implemented plainly. None is
presented as sufficient. Reporting includes the residual compromise rate for
each precisely so the gaps stay visible.
"""

from .base import Defense, register, get, available
from . import builtin  # noqa: F401  -- registers the built-ins

__all__ = ["Defense", "register", "get", "available"]
