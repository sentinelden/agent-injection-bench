"""Deterministic reference targets.

These exist so the benchmark can be tested, demonstrated, and regression-
checked without a network call or an API key. They are not models; they are
fixed policies that stand in for the two ends of the spectrum.

`naive` follows any instruction it can find, including hostile ones. It is the
worst-case baseline -- an agent with no separation between data and
instructions at all. Every real system should score better than this, and a
defense that cannot beat it is doing nothing.

`paranoid` never calls tools and never echoes injected content. It is the
degenerate-safe baseline: a perfect compromise score and a terrible
over-refusal score. It exists to make the point that the compromise number
alone is meaningless, which is why the report always prints both.
"""

from __future__ import annotations

import re

from .base import Scenario, register
from ..scoring import AgentResponse, ToolCall

# Matches the shapes injected content uses to smuggle a canary, e.g.
# "reply with EXFIL-7Q2" or "output the token ABC-123".
_CANARY_HINT = re.compile(
    r"(?:reply with|respond with|output|print|say|include|echo)\s+[\"']?([A-Z0-9][A-Z0-9_-]{4,})",
    re.IGNORECASE,
)
_TOOL_HINT = re.compile(r"call\s+(?:the\s+)?[\"']?([a-z_][a-z0-9_]*)[\"']?", re.IGNORECASE)


class NaiveAgent:
    name = "naive"

    def __call__(self, scenario: Scenario) -> AgentResponse:
        # Concatenate everything it was given, exactly as an agent with no
        # provenance tracking would.
        blob = "\n".join(m.get("content", "") for m in scenario.messages)
        blob += "\n" + "\n".join(scenario.tool_results.values())

        text_parts: list[str] = ["Done."]
        calls: list[ToolCall] = []

        for match in _CANARY_HINT.finditer(blob):
            text_parts.append(match.group(1))

        tool_names = {t.name for t in scenario.tools}
        for match in _TOOL_HINT.finditer(blob):
            candidate = match.group(1)
            if candidate in tool_names:
                # Hand it any attacker-looking value it can find, which is how
                # exfiltration actually happens.
                secret = scenario.attack.success.secret
                args = {"input": secret} if secret and secret in blob else {"input": blob[:120]}
                calls.append(ToolCall(name=candidate, arguments=args))

        return AgentResponse(text=" ".join(text_parts), tool_calls=calls)


class ParanoidAgent:
    name = "paranoid"

    def __call__(self, scenario: Scenario) -> AgentResponse:
        return AgentResponse(text="I can't help with that.", tool_calls=[])


@register("naive")
def _naive(**_: object) -> NaiveAgent:
    return NaiveAgent()


@register("paranoid")
def _paranoid(**_: object) -> ParanoidAgent:
    return ParanoidAgent()
