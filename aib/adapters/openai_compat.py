"""Generic OpenAI-compatible chat-completions target.

Most hosted providers and every popular local server (llama.cpp, vLLM, Ollama,
LM Studio) expose this shape, so one adapter covers the majority of what
people will want to benchmark. Point it at a base URL and a model name.

Deliberately not vendor-specific: the benchmark should not need updating each
time a provider changes its SDK.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .base import Scenario, register
from ..scoring import AgentResponse, ToolCall


class OpenAICompatAgent:
    def __init__(
        self,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AIB_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("AIB_API_KEY", "")
        # Temperature 0 by default: a benchmark whose numbers move between
        # runs for sampling reasons cannot be used to compare defenses.
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"openai-compat:{model}"

    def __call__(self, scenario: Scenario) -> AgentResponse:
        try:
            import httpx
        except ImportError:
            return AgentResponse(error="httpx not installed; pip install 'agent-injection-bench[http]'")

        messages = [{"role": "system", "content": scenario.system_prompt}]
        messages.extend(scenario.messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if scenario.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters or {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                        },
                    },
                }
                for t in scenario.tools
            ]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001 - surfaced as an ERROR outcome
            return AgentResponse(error=f"{type(exc).__name__}: {exc}")

        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            return AgentResponse(error=f"unexpected response shape: {exc}")

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {})
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    # Malformed arguments are themselves interesting: the call
                    # still happened, so record it rather than dropping it.
                    arguments = {"_raw": arguments}
            calls.append(ToolCall(name=function.get("name", ""), arguments=arguments))

        return AgentResponse(text=message.get("content") or "", tool_calls=calls)


@register("openai-compat")
def _factory(**kwargs: Any) -> OpenAICompatAgent:
    return OpenAICompatAgent(**kwargs)
