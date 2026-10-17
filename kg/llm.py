"""LLM erişimi: Anthropic API (varsayılan), yerel model (OpenAI uyumlu uç nokta) veya elle
kopyala-yapıştır (istem kullanıcıya gösterilir, kullanıcı Claude.ai gibi bir sohbete yapıştırıp
yanıtı geri yapıştırır).

Seçim .env ile yapılır: LLM_BACKEND=anthropic | openai_compat | manual.
İki işlem gerekir: (1) Pydantic şemasıyla doğrulanmış JSON çıktısı, (2) ajan için araç döngüsü.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import DATA_DIR, env, setting
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

        key = setting("ANTHROPIC_API_KEY")
        if not (key or env("ANTHROPIC_AUTH_TOKEN") or env("ANTHROPIC_PROFILE")):
            raise LLMError("ANTHROPIC_API_KEY tanımlı değil (Ayarlar sayfası veya Dokploy → Environment)")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4, **({"api_key": key} if key else {}))
        self.model_name = model or setting("LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
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

        self.base_url = (base_url or setting("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model_name = model or setting("OPENAI_COMPAT_MODEL") or "local-model"
        key = api_key or setting("OPENAI_COMPAT_API_KEY")
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


# ---------------------------------------------------------------- Google Gemini

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiLLM:
    """Google Gemini API (REST, generateContent). Yapılandırılmış çıktı için responseJsonSchema kullanılır;
    model şemayı reddederse JSON modu + istemdeki şemaya düşülür. 429/5xx'te bekleyip yeniden dener."""

    def __init__(self, model: str | None = None, api_key: str | None = None, transport: Any = None):
        import httpx

        self.api_key = api_key or setting("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY tanımlı değil (Ayarlar sayfası veya Dokploy → Environment)")
        self.model_name = model or setting("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.http = httpx.Client(timeout=600, transport=transport)
        self.max_attempts = 6

    def _generate(self, body: dict[str, Any]) -> dict[str, Any]:
        import time as _time

        url = GEMINI_URL.format(model=self.model_name.removeprefix("models/"))
        delay = 2.0
        for attempt in range(1, self.max_attempts + 1):
            r = self.http.post(url, json=body, headers={"x-goog-api-key": self.api_key})
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_attempts:
                wait = r.headers.get("retry-after")
                _time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code >= 400:
                try:
                    msg = r.json().get("error", {}).get("message", r.text)
                except ValueError:
                    msg = r.text
                raise LLMError(f"Gemini API hatası {r.status_code}: {msg[:500]}")
            return r.json()
        raise LLMError("Gemini API yanıt vermedi")

    @staticmethod
    def _candidate(data: dict[str, Any]) -> dict[str, Any]:
        if data.get("promptFeedback", {}).get("blockReason"):
            raise LLMRefusal(f"Gemini isteği engelledi: {data['promptFeedback']['blockReason']}")
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError("Gemini yanıtında aday yok")
        c = cands[0]
        reason = c.get("finishReason")
        if reason == "MAX_TOKENS":
            raise LLMError("max_tokens sınırına ulaşıldı; bölüm daha küçük parçalara ayrılmalı")
        if reason in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
            raise LLMRefusal(f"Gemini yanıtı kesti: {reason}")
        return c.get("content") or {"parts": []}

    @staticmethod
    def _text(content: dict[str, Any]) -> str:
        return "".join(p.get("text", "") for p in content.get("parts", []) if not p.get("thought"))

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        js = strict_json_schema(schema)
        base = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}]}
        cfg = {"temperature": 0, "maxOutputTokens": 65536, "responseMimeType": "application/json"}
        try:
            data = self._generate({**base, "generationConfig": {**cfg, "responseJsonSchema": js}})
        except LLMError as exc:
            if "400" not in str(exc) or "schema" not in str(exc).lower():
                raise
            # eski modeller: şemayı istemde ver, yalnız JSON modu
            hint = "\n\nYanıtı yalnızca şu JSON şemasına uyan tek bir JSON nesnesi olarak ver:\n" + json.dumps(js, ensure_ascii=False)
            data = self._generate({"systemInstruction": {"parts": [{"text": system + hint}]},
                                   "contents": base["contents"], "generationConfig": cfg})
        text = self._text(self._candidate(data))
        try:
            return schema.model_validate_json(extract_json(text) if text.strip()[:1] not in "{[" else text)
        except ValidationError as exc:
            raise LLMError(f"şema doğrulaması başarısız: {exc}") from exc

    def complete_text(self, system: str, user: str) -> str:
        data = self._generate({"systemInstruction": {"parts": [{"text": system}]},
                               "contents": [{"role": "user", "parts": [{"text": user}]}],
                               "generationConfig": {"temperature": 0}})
        return self._text(self._candidate(data))

    def run_tools(self, system: str, user: str, tools: list[Tool], max_turns: int = 12) -> str:
        by_name = {t.name: t for t in tools}
        decls = [{"name": t.name, "description": t.description, "parametersJsonSchema": t.input_schema} for t in tools]
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": user}]}]
        for _ in range(max_turns):
            data = self._generate({"systemInstruction": {"parts": [{"text": system}]}, "contents": contents,
                                   "tools": [{"functionDeclarations": decls}], "generationConfig": {"temperature": 0}})
            content = self._candidate(data)
            contents.append({"role": "model", "parts": content.get("parts", [])})
            calls = [p["functionCall"] for p in content.get("parts", []) if "functionCall" in p]
            if not calls:
                return self._text(content)
            responses = []
            for call in calls:
                text, is_error = _call_tool(by_name.get(call.get("name")), dict(call.get("args") or {}))
                responses.append({"functionResponse": {"name": call.get("name"),
                                                       "response": {"error" if is_error else "content": text}}})
            contents.append({"role": "user", "parts": responses})
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


# ---------------------------------------------------------------- elle kopyala-yapıştır


class ManualBroker(Protocol):
    """İstemi kullanıcıya iletip yanıtını alan aracı (web sayfası veya terminal).

    `parse` yanıtı doğrular; hata fırlatırsa aracı hatayı kullanıcıya gösterip yeni yanıt ister."""

    def ask(self, kind: str, prompt: str, parse: Callable[[str], Any]) -> Any: ...


def extract_json(text: str) -> str:
    """Sohbet yanıtından JSON'u çıkarır: ```json çitlerini ve öncesi/sonrası açıklamaları atar."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("yanıtta JSON nesnesi bulunamadı ({ ... } bekleniyor)")
    return text[start:end + 1]


def manual_json_prompt(system: str, user: str, schema: type[BaseModel]) -> str:
    return (f"{system}\n\n"
            "YANIT BİÇİMİ: Yanıtın YALNIZCA aşağıdaki JSON şemasına uyan tek bir JSON nesnesi olsun. "
            "Açıklama yazma; tüm alanları yaz, bilinmeyen alanlara null ver.\n\n"
            f"JSON ŞEMASI:\n```json\n{json.dumps(strict_json_schema(schema), ensure_ascii=False)}\n```\n\n"
            f"---\n\n{user}")


class ManualLLM:
    """Elle mod: her LLM çağrısı bir istem olarak kullanıcıya gösterilir.

    Yanıtlar istem özetiyle data/cache/manual/ altında saklanır; aynı istem tekrar sorulmaz
    (sunucu yeniden başlasa da kaldığı yerden devam eder)."""

    manual = True

    def __init__(self, broker: ManualBroker, cache_dir: Path | None = None, model_name: str | None = None):
        self.broker = broker
        self.cache_dir = cache_dir or DATA_DIR / "cache" / "manual"
        self.model_name = model_name or setting("MANUAL_MODEL_NAME", "manual (sohbet arayüzü)")

    def _cached(self, prompt: str) -> tuple[Path, str | None]:
        path = self.cache_dir / (hashlib.sha1(prompt.encode("utf-8")).hexdigest() + ".txt")
        return path, (path.read_text(encoding="utf-8") if path.exists() else None)

    def _store(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        prompt = manual_json_prompt(system, user, schema)
        path, cached = self._cached(prompt)

        def parse(text: str) -> T:
            try:
                return schema.model_validate_json(extract_json(text))
            except ValidationError as exc:
                raise ValueError(f"JSON şemaya uymuyor: {exc}") from exc

        if cached is not None:
            try:
                return parse(cached)
            except ValueError:
                pass
        holder: dict[str, str] = {}

        def parse_and_keep(text: str) -> T:
            out = parse(text)
            holder["text"] = text
            return out

        result = self.broker.ask(schema.__name__, prompt, parse_and_keep)
        self._store(path, holder["text"])
        return result

    def complete_text(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n---\n\n{user}"
        path, cached = self._cached(prompt)
        if cached:
            return cached

        def parse(text: str) -> str:
            if not text.strip():
                raise ValueError("yanıt boş")
            return text.strip()

        text = self.broker.ask("answer", prompt, parse)
        self._store(path, text)
        return text

    def run_tools(self, system: str, user: str, tools: list[Tool], max_turns: int = 12) -> str:
        raise LLMError("elle modda ajan araç kullanamaz; ajan yerine bağlam derlenmiş tek bir istem kullanılır")


class CliBroker:
    """Terminal için: istemi data/manual/ altına yazar, yanıt dosyasını bekler."""

    def __init__(self, out_dir: Path | None = None):
        self.out_dir = out_dir or DATA_DIR / "manual"
        self.n = 0

    def ask(self, kind: str, prompt: str, parse: Callable[[str], Any]) -> Any:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n += 1
        stem = self.out_dir / f"{self.n:03d}_{kind}"
        prompt_path, resp_path = stem.with_suffix(".prompt.md"), stem.with_suffix(".response.txt")
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"\n[elle mod] İstem: {prompt_path} ({len(prompt):,} karakter)\n"
              f"Sohbete yapıştırın, yanıtı {resp_path} dosyasına kaydedin ve Enter'a basın (iptal: q).")
        while True:
            if input("> ").strip().lower() == "q":
                raise LLMError("kullanıcı iptal etti")
            if not resp_path.exists():
                print(f"{resp_path} bulunamadı")
                continue
            try:
                return parse(resp_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                print(f"Yanıt geçersiz: {exc}\nDüzeltip tekrar Enter'a basın.")


_manual_broker: ManualBroker | None = None


def set_manual_broker(broker: ManualBroker) -> None:
    global _manual_broker
    _manual_broker = broker


def is_manual(llm: Any) -> bool:
    return bool(getattr(llm, "manual", False))


def get_llm() -> LLM:
    backend = (setting("LLM_BACKEND", "anthropic") or "anthropic").lower()
    if backend == "anthropic":
        return AnthropicLLM()
    if backend == "gemini":
        return GeminiLLM()
    if backend in ("openai_compat", "local"):
        return OpenAICompatLLM()
    if backend == "manual":
        return ManualLLM(_manual_broker or CliBroker())
    raise LLMError(f"bilinmeyen LLM_BACKEND: {backend}")
