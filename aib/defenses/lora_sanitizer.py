"""A production input sanitizer, measured as a defense.

This is the prompt-injection filter that runs in front of every user message
at asklora.io (``src/server/auth/inputSanitizer.ts`` in the public backend,
github.com/Nikhiljangra07/LoRa-EmotionalEngine-v1), ported pattern for
pattern from TypeScript to Python. It is included here for the reason the
README gives for ``filter``: it is the kind of thing that actually ships, and
its residual compromise rate is more useful than its reputation.

Design, as deployed:

* Normalise before matching: strip zero-width / invisible Unicode, collapse
  whitespace. Defeats the cheapest obfuscations.
* Twenty-one patterns in five families: instruction override, role hijack,
  system-prompt extraction, delimiter injection, jailbreak phrases. Each is
  anchored on word boundaries with a bounded gap (``.{0,20}``) so that
  "ignore my previous email about the rules" does not fire but "ignore all
  previous instructions" does.
* Strip, don't block: matches are replaced with ``[filtered]`` and the rest of
  the message is kept. In production this keeps false positives cheap, which
  is why the over-refusal column for this defense should stay at zero.

In production the sanitizer sees only the ``user_message`` channel; here it is
applied to every user turn and every tool result, because the question the
benchmark asks is what the same filter is worth when the hostile text arrives
through a channel it was never tuned for. Expect the answer to be "not much"
for attacks that carry no imperative phrase, and for ``encoding`` attacks the
patterns cannot see through. That gap is the point of measuring it.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .base import register
from ..adapters.base import Scenario

_INVISIBLE = re.compile(r"[​-‏ - ﻿]")
_WS = re.compile(r"\s+")

# Verbatim port of INJECTION_PATTERNS. JavaScript's `.` does not match
# newlines and neither does Python's without DOTALL, so semantics are
# preserved; whitespace is collapsed before matching in both.
_PATTERNS: tuple[tuple[str, str], ...] = (
    # Direct instruction overrides
    ("instruction_override", r"\bignore\b.{0,20}\b(previous|prior|above|earlier)\b.{0,20}\b(instructions?|prompts?|rules?|directives?)\b"),
    ("instruction_override", r"\bdisregard\b.{0,20}\b(previous|prior|above|earlier)\b.{0,20}\b(instructions?|prompts?|rules?)\b"),
    ("instruction_override", r"\bforget\b.{0,20}\b(previous|prior|above|earlier)\b.{0,20}\b(instructions?|prompts?|rules?)\b"),
    ("instruction_override", r"\boverride\b.{0,20}\b(instructions?|prompts?|rules?|system)\b"),
    # Role hijacking
    ("role_hijack", r"\byou are now\b.{0,10}\b(a|an|the)\b"),
    ("role_hijack", r"\bact as\b.{0,15}\b(if you are|a|an|the)\b"),
    ("role_hijack", r"\bpretend\b.{0,10}\b(to be|you are|you're)\b"),
    ("role_hijack", r"\bswitch to\b.{0,15}\bmode\b"),
    ("role_hijack", r"\byou must\b.{0,20}\b(obey|comply|follow)\b"),
    ("role_hijack", r"\bnew (instructions?|persona|identity|character)\b"),
    # System prompt extraction
    ("prompt_extraction", r"\b(show|reveal|display|print|output|repeat|echo|dump)\b.{0,20}\b(system|initial|original)?\s*(prompt|instructions?|rules?|guidelines?)\b"),
    ("prompt_extraction", r"\bwhat (are|is|were) your\b.{0,15}\b(system\s*)?(prompt|instructions?|rules?)\b"),
    ("prompt_extraction", r"\brepeat\b.{0,15}\b(everything|all|word for word)\b.{0,15}\b(above|before|earlier)\b"),
    # Delimiter injection
    ("delimiter_injection", r"\[\s*(?:SYSTEM|INST|ADMIN|ROOT|OVERRIDE)\s*\]"),
    ("delimiter_injection", r"<<\s*(?:SYS|SYSTEM|INSTRUCTIONS?)\s*>>"),
    ("delimiter_injection", r"</?system(?:\s+[^>]*)?\s*>"),
    ("delimiter_injection", r"```\s*system\b"),
    # Jailbreak phrases
    ("jailbreak", r"\b(?:DAN|do anything now)\b.{0,10}\bmode\b"),
    ("jailbreak", r"\benable\b.{0,10}\b(developer|debug|admin|unrestricted|god|jailbreak)\b.{0,5}\bmode\b"),
    ("jailbreak", r"\bjailbreak\b"),
    ("jailbreak", r"\bno restrictions?\b.{0,15}\b(mode|apply|from now)\b"),
)
_COMPILED = tuple((label, re.compile(p, re.IGNORECASE)) for label, p in _PATTERNS)


def sanitize(raw: str) -> tuple[str, list[str]]:
    """Return (cleaned_text, matched_labels), mirroring ``sanitizeInput``.

    JavaScript's ``String.replace`` with a non-global regex replaces only the
    first occurrence; the production code loops once per pattern, so a second
    occurrence of the same pattern survives. That quirk is reproduced here on
    purpose (``count=1``): the benchmark should measure the filter as shipped,
    not an idealised version of it.
    """
    cleaned = _WS.sub(" ", _INVISIBLE.sub("", raw)).strip()
    matched: list[str] = []
    for label, pattern in _COMPILED:
        if pattern.search(cleaned):
            if label not in matched:
                matched.append(label)
            cleaned = pattern.sub("[filtered]", cleaned, count=1)
    return cleaned.strip(), matched


class LoraSanitizer:
    name = "lora-sanitizer"

    def __call__(self, scenario: Scenario) -> Scenario:
        messages = [
            {**m, "content": sanitize(m["content"])[0]} if m.get("role") == "user" else m
            for m in scenario.messages
        ]
        tool_results = {k: sanitize(v)[0] for k, v in scenario.tool_results.items()}
        return replace(scenario, messages=messages, tool_results=tool_results)


register(LoraSanitizer.name)(lambda **_: LoraSanitizer())
