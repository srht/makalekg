import json
import sys
import importlib

import httpx
import pytest
from pydantic import BaseModel

from kg.llm import GeminiLLM, LLMError, LLMRefusal, Tool


class Out(BaseModel):
    name: str
    n: int


def _resp(parts, finish="STOP"):
    return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish}]})


def test_gemini_json_schema_and_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    seen = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append((req.url.path, req.headers.get("x-goog-api-key"), body))
        if len(seen) == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "slow down"}})
        return _resp([{"text": "düşünce", "thought": True}, {"text": '{"name": "Ankara", "n": 3}'}])

    llm = GeminiLLM(model="gemini-test", api_key="KEY", transport=httpx.MockTransport(handler))
    out = llm.complete_json("sys", "user", Out)
    assert out == Out(name="Ankara", n=3)
    path, key, body = seen[-1]
    assert path.endswith("/models/gemini-test:generateContent") and key == "KEY" and len(seen) == 2
    gc = body["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and gc["responseJsonSchema"]["required"] == ["name", "n"]
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_schema_fallback_and_errors(monkeypatch):
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if "responseJsonSchema" in body["generationConfig"]:
            return httpx.Response(400, json={"error": {"message": "Invalid JSON schema field"}})
        return _resp([{"text": '```json\n{"name": "x", "n": 1}\n```'}])

    llm = GeminiLLM(api_key="K", transport=httpx.MockTransport(handler))
    assert llm.complete_json("s", "u", Out).n == 1
    assert "JSON şemasına" in calls[1]["systemInstruction"]["parts"][0]["text"]

    llm = GeminiLLM(api_key="K", transport=httpx.MockTransport(lambda r: _resp([], finish="SAFETY")))
    with pytest.raises(LLMRefusal):
        llm.complete_json("s", "u", Out)
    llm = GeminiLLM(api_key="K", transport=httpx.MockTransport(lambda r: httpx.Response(403, json={"error": {"message": "API key not valid"}})))
    with pytest.raises(LLMError, match="API key not valid"):
        llm.complete_json("s", "u", Out)


def test_gemini_tools():
    turns = []

    def handler(req):
        body = json.loads(req.content)
        turns.append(body)
        if len(turns) == 1:
            assert body["tools"][0]["functionDeclarations"][0]["name"] == "topla"
            return _resp([{"functionCall": {"name": "topla", "args": {"a": 2, "b": 3}}}])
        assert body["contents"][-1]["parts"][0]["functionResponse"]["response"]["content"] == "5"
        return _resp([{"text": "Sonuç 5"}])

    tool = Tool(name="topla", description="iki sayıyı topla",
                input_schema={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
                fn=lambda a, b: str(a + b))
    llm = GeminiLLM(api_key="K", transport=httpx.MockTransport(handler))
    assert llm.run_tools("s", "2+3?", [tool]) == "Sonuç 5"


def test_settings_page_gemini(tmp_data, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("APP_PASSWORD", "pw")
    monkeypatch.setenv("APP_USERNAME", "admin")
    monkeypatch.setenv("LLM_BACKEND", "manual")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import kg.okg.store as st
    import kg.web.jobs as jobs_mod
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    monkeypatch.setattr(jobs_mod, "DATA_DIR", tmp_data / "data")
    sys.modules.pop("kg.web.app", None)
    app_mod = importlib.import_module("kg.web.app")
    c = TestClient(app_mod.app)
    c.auth = ("admin", "pw")
    from test_web import _csrf
    page = c.get("/settings").text
    assert "Google Gemini" in page and 'value="manual" checked' in page
    tok = _csrf(c, "/settings")
    r = c.post("/settings", data={"csrf": tok, "LLM_BACKEND": "gemini", "GEMINI_API_KEY": "AIzaSECRET1234",
                                  "GEMINI_MODEL": "gemini-2.5-pro", "LLM_MODEL": ""}, follow_redirects=False)
    assert r.status_code == 303
    page = c.get("/settings").text
    assert "AIzaSECRET1234" not in page and "1234" in page and 'value="gemini" checked' in page
    assert "Google Gemini · gemini-2.5-pro" in c.get("/").text
    from kg.llm import get_llm
    llm = get_llm()
    assert isinstance(llm, GeminiLLM) and llm.api_key == "AIzaSECRET1234" and llm.model_name == "gemini-2.5-pro"
    assert oct((tmp_data / "data" / "settings.json").stat().st_mode & 0o777) == "0o600"
    # boş anahtar alanı kayıtlıyı korur; "sil" kutusu kaldırır
    c.post("/settings", data={"csrf": tok, "LLM_BACKEND": "gemini", "GEMINI_API_KEY": "", "GEMINI_MODEL": ""})
    assert get_llm().api_key == "AIzaSECRET1234" and get_llm().model_name == "gemini-2.5-flash"
    c.post("/settings", data={"csrf": tok, "LLM_BACKEND": "gemini", "GEMINI_API_KEY": "", "clear_GEMINI_API_KEY": "1"})
    assert "API anahtarı yok" in c.get("/").text
    assert c.post("/settings", data={"csrf": "bad", "LLM_BACKEND": "manual"}).status_code == 403
    assert c.post("/api/settings/test", json={"csrf": tok}).json()["ok"] is False  # anahtar yok → açıklayıcı hata
