"""The attack corpus: schema, loading, and validation.

One attack per line of JSONL, so the corpus stays diffable and a pull request
adding an attack is reviewable as text.

Every attack describes a *scenario*, not a prompt. Prompt-injection benchmarks
that ship bare strings measure the wrong thing: real injections arrive through
a channel -- a tool result, a fetched document, a filename, a calendar invite --
and the channel is most of what makes them work. An agent that ignores
"ignore previous instructions" typed by the user may well obey it when the
same text arrives inside a web page it was asked to summarize.

So each attack declares:

  channel        where the hostile text enters (see Channel below)
  task           the benign thing the user actually asked for
  injected       the hostile content delivered through the channel
  tools          the tool registry the agent is given for this scenario
  success        a machine-checkable condition (see scoring.py)
  steps          for a chain, what arrives after the first turn (see ChainStep)

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
    "tool_chain",       # hostile text that needs several tool calls or sessions to pay off
]

CHANNELS: tuple[str, ...] = (
    "tool_result", "document", "web_page", "filename",
    "metadata", "user_message", "multi_turn", "tool_chain",
)

# Where a later step of a chain comes from.
STEP_CHANNELS: tuple[str, ...] = (
    "tool_result", "document", "web_page", "metadata", "user_message", "memory", "agent",
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
class ChainStep:
    """One later turn of a chain: what arrives, and in which conversation.

    A compromise through intermediate state only works if the state carried it,
    so a step can take its content from what the agent actually did. With
    `from_call`, the content is the arguments of the agent's last call to that
    tool. If the agent never made that call, the chain has broken and later steps
    do not run, because an agent that refused the first stage has resisted.

    `new_session` starts a fresh conversation, so the only thing that survives is
    what the step carries, which is how memory poisoning and hand-offs work.
    """

    channel: str
    content: str | None = None
    from_call: str | None = None
    task: str | None = None
    new_session: bool = False


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
    steps: list[ChainStep] = field(default_factory=list)

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

    steps = [_parse_step(s, where, tool_names) for s in raw.get("steps", [])]
    if steps and raw["channel"] != "tool_chain":
        raise DatasetError(f"{where}: only a tool_chain attack can have steps")

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
        steps=steps,
    )


def _parse_step(raw: dict[str, Any], where: str, tool_names: set[str]) -> ChainStep:
    channel = raw.get("channel")
    if channel not in STEP_CHANNELS:
        raise DatasetError(
            f"{where}: unknown step channel {channel!r}; expected one of {', '.join(STEP_CHANNELS)}"
        )
    content, from_call = raw.get("content"), raw.get("from_call")
    if (content is None) == (from_call is None):
        raise DatasetError(f"{where}: a step needs exactly one of content and from_call")
    # A step fed by a tool the agent cannot call would never run, which would
    # silently look like the agent resisting.
    if from_call is not None and from_call not in tool_names:
        raise DatasetError(f"{where}: step from_call={from_call!r} is not in this attack's tool registry")
    return ChainStep(
        channel=channel,
        content=content,
        from_call=from_call,
        task=raw.get("task"),
        new_session=bool(raw.get("new_session", False)),
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
