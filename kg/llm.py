"""LLM erişimi: Anthropic API (varsayılan) veya yerel model (OpenAI uyumlu uç nokta).

Seçim .env ile yapılır: LLM_BACKEND=anthropic | openai_compat.
İki işlem gerekir: (1) Pydantic şemasıyla doğrulanmış JSON çıktısı, (2) ajan için araç döngüsü.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import env
from .schema import strict_json_schema

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]


class LLM(Protocol):
    model_name: str

    def complete_json(self, system: str, user: str, schema: type[T]) -> T: ...

    def run_tools(self, system: str, user: str, tools: list[Tool], max_turns: int = 12) -> str: ...


def _call_tool(tool: Tool | None, args: dict[str, Any]) -> tuple[str, bool]:
    if tool is None:
        return "unknown tool", True
    try:
        result = tool.fn(**args)
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return text[:20000], False
    except Exception as exc:  # araç hatası modele is_error olarak döner
        return f"{type(exc).__name__}: {exc}", True


# ---------------------------------------------------------------- Anthropic


class AnthropicLLM:
    def __init__(self, model: str | None = None, effort: str = "high"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.model_name = model or env("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.effort = effort

    def _create(self, **kwargs: Any) -> Any:
        # fallbacks="default": güvenlik sınıflandırıcısı bir isteği reddederse sunucu,
        # reddin kategorisine göre önerilen modelde aynı isteği yeniden çalıştırır.
        response = self.client.beta.messages.create(
            model=self.model_name,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal(f"model isteği reddetti: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise LLMError("max_tokens sınırına ulaşıldı; bölüm daha küçük parçalara ayrılmalı")
        return response

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        response = self._create(
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
            },
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("yanıtta metin bloğu yok")
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"şema doğrulaması başarısız: {exc}") from exc

    def run_tools(self, system: str, user: str, tools: list[Tool], max_turns: int = 12) -> str:
        by_name = {t.name: t for t in tools}
        tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        for _ in range(max_turns):
            response = self._create(
                system=system, messages=messages, tools=tool_defs,
                tool_choice={"type": "auto"}, output_config={"effort": self.effort},
            )
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return "\n".join(b.text for b in response.content if b.type == "text")
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                text, is_error = _call_tool(by_name.get(block.name), dict(block.input or {}))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        raise LLMError("ajan tur sınırına ulaştı")


# ---------------------------------------------------------------- OpenAI uyumlu yerel model


class OpenAICompatLLM:
    """Yerel model sunucusu (vLLM, llama.cpp server, Ollama vb.) için /chat/completions istemcisi."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        import httpx

        self.base_url = (base_url or env("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model_name = model or env("OPENAI_COMPAT_MODEL") or "local-model"
        key = api_key or env("OPENAI_COMPAT_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(timeout=600, headers=headers)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post(f"{self.base_url}/chat/completions", json={"model": self.model_name, **payload})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        msg = self._post({
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": strict_json_schema(schema), "strict": True},
            },
        })
        try:
            return schema.model_validate_json(msg.get("content") or "")
        except ValidationError as exc:
            raise LLMError(f"şema doğrulaması başarısız: {exc}") from exc

    def run_tools(self, system: str, user: str, tools: list[Tool], max_turns: int = 12) -> str:
        by_name = {t.name: t for t in tools}
        tool_defs = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                        "parameters": t.input_schema}} for t in tools]
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(max_turns):
            msg = self._post({"messages": messages, "tools": tool_defs, "temperature": 0})
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or ""
            for call in calls:
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                text, _ = _call_tool(by_name.get(call["function"]["name"]), args)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        raise LLMError("ajan tur sınırına ulaştı")


# ---------------------------------------------------------------- test için sahte model


class FakeLLM:
    """Testler ve çevrimdışı geliştirme için: yanıtları bir fonksiyon üretir."""

    def __init__(self, responder: Callable[[str, str, type[BaseModel]], Any], model_name: str = "fake-llm",
                 tool_responder: Callable[[str, list[Tool]], str] | None = None):
        self.responder = responder
        self.tool_responder = tool_responder
        self.model_name = model_name
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        self.calls.append((schema.__name__, user))
        out = self.responder(system, user, schema)
        return out if isinstance(out, schema) else schema.model_validate(out)

    def run_tools(self, system: str, user: str, tools: list[Tool], max_turns: int = 12) -> str:
        if self.tool_responder is None:
            return ""
        return self.tool_responder(user, tools)


def get_llm() -> LLM:
    backend = (env("LLM_BACKEND", "anthropic") or "anthropic").lower()
    if backend == "anthropic":
        return AnthropicLLM()
    if backend in ("openai_compat", "local"):
        return OpenAICompatLLM()
    raise LLMError(f"bilinmeyen LLM_BACKEND: {backend}")
