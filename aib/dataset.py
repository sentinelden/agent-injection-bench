"""The attack corpus: schema, loading, and validation.

One attack per line of JSONL, so the corpus stays diffable and a pull request
adding an attack is reviewable as text.

Every attack describes a *scenario*, not a prompt. Prompt-injection benchmarks
that ship bare strings measure the wrong thing: real injections arrive through
a channel -- a tool result, a fetched document, a filename, a calendar invite --
and the channel is most of what makes them work. An agent that ignores
"ignore previous instructions" typed by the user may well obey it when the
same text arrives inside a web page it was asked to summarise.

So each attack declares:

  channel        where the hostile text enters (see Channel below)
  task           the benign thing the user actually asked for
  injected       the hostile content delivered through the channel
  tools          the tool registry the agent is given for this scenario
  success        a machine-checkable condition (see scoring.py)

Fields are validated on load. A malformed attack is a hard error rather than a
silently skipped line, because a corpus that quietly shrinks produces scores
that look better than they are.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

Channel = Literal[
    "tool_result",      # hostile text returned by a tool the agent called
    "document",         # hostile text inside a document the agent was asked to read
    "web_page",         # hostile text inside fetched web content
    "filename",         # hostile text in a filename or path
    "metadata",         # hostile text in EXIF, headers, or similar
    "user_message",     # hostile text typed directly by the user (baseline)
    "multi_turn",       # hostile text planted in an earlier turn of history
]

CHANNELS: tuple[str, ...] = (
    "tool_result", "document", "web_page", "filename",
    "metadata", "user_message", "multi_turn",
)


@dataclass
class SuccessCriterion:
    kind: Literal["canary", "tool_call", "exfiltration", "refusal"]
    canary: str | None = None
    tool: str | None = None
    sink: str | None = None
    secret: str | None = None
    arguments_contain: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(raw: dict[str, Any], attack_id: str) -> "SuccessCriterion":
        kind = raw.get("kind")
        if kind not in ("canary", "tool_call", "exfiltration", "refusal"):
            raise DatasetError(f"{attack_id}: unknown success.kind {kind!r}")

        criterion = SuccessCriterion(
            kind=kind,
            canary=raw.get("canary"),
            tool=raw.get("tool"),
            sink=raw.get("sink"),
            secret=raw.get("secret"),
            arguments_contain=list(raw.get("arguments_contain", [])),
        )

        # Each kind needs its own fields present, or judging is undefined.
        required = {
            "canary": ["canary"],
            "tool_call": ["tool"],
            "exfiltration": ["sink", "secret"],
            "refusal": ["canary"],
        }[kind]
        for name in required:
            if not getattr(criterion, name):
                raise DatasetError(f"{attack_id}: success.kind={kind} requires {name!r}")
        return criterion


@dataclass
class Tool:
    """A tool offered to the agent for one scenario."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    # Returned verbatim when the agent calls this tool. The injection for a
    # `tool_result` attack lives here.
    returns: str | None = None


@dataclass
class Attack:
    id: str
    channel: str
    family: str              # grouping for reporting, e.g. "instruction_override"
    description: str
    task: str                # what the user legitimately asked for
    injected: str            # hostile content delivered via `channel`
    success: SuccessCriterion
    tools: list[Tool] = field(default_factory=list)
    system_prompt: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str | None = None   # citation, where the technique was first described

    @property
    def is_control(self) -> bool:
        """Control cases measure over-refusal rather than compromise."""
        return self.success.kind == "refusal"


class DatasetError(ValueError):
    """Raised when the corpus is malformed. Never swallowed."""


def load(path: str | Path) -> list[Attack]:
    """Load and validate a JSONL corpus."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"corpus not found: {path}")

    attacks: list[Attack] = []
    seen_ids: set[str] = set()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc}") from exc

        attack = _parse(raw, f"{path}:{lineno}")
        if attack.id in seen_ids:
            raise DatasetError(f"{path}:{lineno}: duplicate attack id {attack.id!r}")
        seen_ids.add(attack.id)
        attacks.append(attack)

    if not attacks:
        raise DatasetError(f"{path}: corpus is empty")
    return attacks


def _parse(raw: dict[str, Any], where: str) -> Attack:
    for required in ("id", "channel", "family", "task", "injected", "success"):
        if required not in raw:
            raise DatasetError(f"{where}: missing required field {required!r}")

    attack_id = str(raw["id"])
    if raw["channel"] not in CHANNELS:
        raise DatasetError(
            f"{where}: unknown channel {raw['channel']!r}; expected one of {', '.join(CHANNELS)}"
        )

    tools = [
        Tool(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("parameters", {}),
            returns=t.get("returns"),
        )
        for t in raw.get("tools", [])
    ]

    criterion = SuccessCriterion.from_dict(raw["success"], attack_id)

    # A tool_call or exfiltration attack that names a tool absent from the
    # registry can never succeed, which would silently look like a good score.
    tool_names = {t.name for t in tools}
    for field_name in ("tool", "sink"):
        target = getattr(criterion, field_name)
        if target and target not in tool_names:
            raise DatasetError(
                f"{where}: success.{field_name}={target!r} is not in this attack's tool registry "
                f"({', '.join(sorted(tool_names)) or 'empty'})"
            )

    return Attack(
        id=attack_id,
        channel=raw["channel"],
        family=raw["family"],
        description=raw.get("description", ""),
        task=raw["task"],
        injected=raw["injected"],
        success=criterion,
        tools=tools,
        system_prompt=raw.get("system_prompt"),
        history=list(raw.get("history", [])),
        tags=list(raw.get("tags", [])),
        source=raw.get("source"),
    )


def by_family(attacks: list[Attack]) -> dict[str, list[Attack]]:
    grouped: dict[str, list[Attack]] = {}
    for attack in attacks:
        grouped.setdefault(attack.family, []).append(attack)
    return grouped


def by_channel(attacks: list[Attack]) -> dict[str, list[Attack]]:
    grouped: dict[str, list[Attack]] = {}
    for attack in attacks:
        grouped.setdefault(attack.channel, []).append(attack)
    return grouped
