"""Elle (kopyala-yapıştır) LLM modu."""
import json

import pytest
from conftest import FAKE_CLAIMS

from kg.extract import Extractor
from kg.llm import LLMError, ManualLLM, extract_json
from kg.schema import ExtractionResult


class ScriptedBroker:
    """Yanıtları sırayla verir; parse hata verirse bir sonrakini dener (kullanıcının düzeltmesi gibi)."""

    def __init__(self, answers=None, fn=None):
        self.answers = list(answers or [])
        self.fn = fn
        self.prompts = []
        self.errors = []

    def ask(self, kind, prompt, parse):
        self.prompts.append((kind, prompt))
        while True:
            text = self.fn(kind, prompt) if self.fn else self.answers.pop(0)
            try:
                return parse(text)
            except ValueError as exc:
                self.errors.append(str(exc))
                if self.fn:
                    raise


def test_extract_json_tolerates_chat_formatting():
    assert json.loads(extract_json('İşte yanıt:\n```json\n{"claims": []}\n```\nUmarım yardımcı olur.')) == {"claims": []}
    assert json.loads(extract_json('ön söz {"a": {"b": 1}} son söz')) == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        extract_json("JSON yok")


def test_manual_retries_invalid_and_caches(tmp_path):
    broker = ScriptedBroker(["hiç JSON yok", '{"claims": [{"claim": "eksik"}]}', '```json\n{"claims": []}\n```'])
    llm = ManualLLM(broker, cache_dir=tmp_path)
    out = llm.complete_json("sys", "user", ExtractionResult)
    assert out.claims == [] and len(broker.errors) == 2
    assert "JSON ŞEMASI" in broker.prompts[0][1]
    # aynı istem tekrar sorulmaz (sunucu yeniden başlasa da)
    llm2 = ManualLLM(ScriptedBroker([]), cache_dir=tmp_path)
    assert llm2.complete_json("sys", "user", ExtractionResult).claims == []
    with pytest.raises(LLMError):
        llm.run_tools("s", "u", [])


def _paper_broker():
    def fn(kind, prompt):
        if kind == "ExtractionResult":
            return "Tabii:\n```json\n" + json.dumps({"claims": FAKE_CLAIMS}) + "\n```"
        items = json.loads(prompt.split("=== ITEMS ===\n", 1)[1])
        return json.dumps({"items": [{"index": it["index"], "label": "supported", "reason": "ok"} for it in items]})
    return ScriptedBroker(fn=fn)


def test_manual_extraction_two_prompts_per_paper(tmp_data, seed_doc):
    broker = _paper_broker()
    llm = ManualLLM(broker, cache_dir=tmp_data / "cache", model_name="claude.ai")
    ex = Extractor(llm, out_dir=tmp_data / "ex", review_queue=tmp_data / "rq.jsonl").extract_document(seed_doc)
    assert [k for k, _ in broker.prompts] == ["ExtractionResult", "VerificationResult"]
    prompt = broker.prompts[0][1]
    assert "## Results" in prompt and "## Methods" in prompt and "Fig. 2 |" in prompt
    assert "## Introduction" not in prompt
    assert len(ex.accepted) == 3 and len(ex.rejected) == 1  # birebir olmayan kaynak cümle yine reddedilir
    assert ex.model == "claude.ai"
