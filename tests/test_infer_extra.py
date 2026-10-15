import json
from types import SimpleNamespace

from kg.graph import Edge, GraphSnapshot
from kg.infer.agent import AgentCandidates, GraphAgent
from kg.llm import AnthropicLLM, FakeLLM
from kg.schema import ExtractionResult
from kg.validate.backtest import run_backtest


def _rel(t, s, d, claim, doi, exp):
    return Edge(t, s, d, {"claim_id": claim, "experiment_id": exp, "source_doi": doi, "species": "mouse",
                          "extraction_confidence": 0.9, "stance": "affirmed", "source_sentence": "x"})


def _graph():
    nodes = {
        "old": {"label": "Paper", "doi": "old", "year": 2020, "lab": "A"},
        "new": {"label": "Paper", "doi": "new", "year": 2025, "lab": "B"},
        "x1": {"label": "Experiment", "design": "in vivo", "species": "mouse", "n_per_group": [12]},
        "x2": {"label": "Experiment", "design": "in vivo", "species": "mouse", "n_per_group": [12]},
        "x3": {"label": "Experiment", "design": "in vivo", "species": "mouse", "n_per_group": [20]},
        "c1": {"label": "Claim", "text": "L increases G"}, "c2": {"label": "Claim", "text": "G protects O"},
        "c3": {"label": "Claim", "text": "L protects O"},
        "L": {"label": "Intervention", "name": "loading"}, "G": {"label": "Gene", "name": "Hsp70", "species": "mouse"},
        "O": {"label": "Outcome", "name": "neuron survival"}, "O2": {"label": "Outcome", "name": "memory"},
    }
    edges = [Edge("REPORTS", "old", "x1"), Edge("REPORTS", "old", "x2"), Edge("REPORTS", "new", "x3"),
             Edge("SUPPORTS", "x1", "c1"), Edge("SUPPORTS", "x2", "c2"), Edge("SUPPORTS", "x3", "c3"),
             _rel("INCREASES", "L", "G", "c1", "old", "x1"), _rel("PROTECTS", "G", "O", "c2", "old", "x2"),
             _rel("PROTECTS", "L", "O", "c3", "new", "x3")]
    return GraphSnapshot(nodes, edges)


def test_backtest_detects_later_support(tmp_path):
    r = run_backtest(_graph(), 2024, k=20, out_dir=tmp_path)
    assert r["n_pre_papers"] == 1 and r["n_post_papers"] == 1
    assert r["precision_at_20"] == 1.0  # L→O hipotezi sonradan desteklendi
    assert (tmp_path / "backtest.md").exists()


def test_agent_paths_must_be_real_evidence():
    snap = _graph().filtered({"old"})

    def responder(system, user, schema):
        assert schema is AgentCandidates
        return {"candidates": [
            {"subject": "L", "predicate": "PROTECTS", "object": "O",
             "path": [{"from": "L", "predicate": "INCREASES", "to": "G"}, {"from": "G", "predicate": "PROTECTS", "to": "O"}],
             "rationale": "zincir"},
            {"subject": "L", "predicate": "PROTECTS", "object": "O2",   # uydurma halka → reddedilmeli
             "path": [{"from": "L", "predicate": "PROTECTS", "to": "O2"}], "rationale": "uydurma"}]}

    agent = GraphAgent(FakeLLM(responder, tool_responder=lambda u, tools: "adaylar..."), snap)
    cands = agent.propose()
    assert [(c.subject, c.object) for c in cands] == [("L", "O")]
    assert agent.search_entities("hsp70")[0]["key"] == "G"
    assert agent.find_paths("L", "O")


def test_anthropic_request_shape(monkeypatch):
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm.model_name, llm.effort = "claude-opus-5-5", "high"
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps({"claims": []}))])

    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    out = llm.complete_json("sys", "user", ExtractionResult)
    assert out.claims == []
    assert seen["model"] == "claude-opus-5-5" and seen["fallbacks"] == "default"
    assert seen["betas"] == ["server-side-fallback-2026-07-01"]
    assert seen["thinking"] == {"type": "adaptive"}
    assert seen["output_config"]["format"]["type"] == "json_schema"
