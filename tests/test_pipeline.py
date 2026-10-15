"""Uçtan uca, çevrimdışı: sahte LLM + bellek-içi graf deposu."""
import json

from conftest import SEED_DOI

from kg.graph import MemoryStore
from kg.infer.rules import rule_candidates
from kg.infer.run import run_inference
from kg.load import evidence_query_returns_hypotheses
from kg.pipeline import Pipeline
from kg.schema import Paper
from kg.validate import acceptance, consistency
from kg.validate.spotcheck import apply_results, build_spotcheck


def _ingest(tmp_data, fake_llm, seed_doc, store=None):
    store = store or MemoryStore(path=tmp_data / "data" / "graph.json")
    pipe = Pipeline(store, fake_llm, offline=True)
    paper = Paper(doi=SEED_DOI, title="synthetic", year=2026, lab="Lab A")
    summary = pipe.ingest([(paper, None)], documents={SEED_DOI: seed_doc})
    return store, summary


def test_segmentation(seed_doc):
    kinds = {s.name: s.kind for s in seed_doc.sections}
    assert kinds["Methods"] == "methods"
    assert "References" not in kinds and "Introduction" in kinds
    legends = seed_doc.legends()
    assert [l.figure for l in legends] == ["Fig. 1", "Fig. 2", "Fig. 3"]
    assert "n = 14, 14, 14 and 17" in seed_doc.legend_for("Fig. 2d").text


def test_ingest_rejects_unverbatim_and_is_idempotent(tmp_data, fake_llm, seed_doc):
    store, summary = _ingest(tmp_data, fake_llm, seed_doc)
    assert summary.run.n_claims == 3
    assert summary.run.n_rejected == 1  # uydurma kaynak cümle grafa girmez
    queue = (tmp_data / "data" / "review_queue.jsonl").read_text().splitlines()
    assert len(queue) == 1 and "Alzheimer" in queue[0]

    snap1 = store.snapshot()
    n_nodes, n_edges = len(snap1.nodes), len(snap1.edges)
    # Tekrar çalıştırma: çift kayıt yok (çıkarım önbellekten, yazma MERGE)
    store2, _ = _ingest(tmp_data, fake_llm, seed_doc, store=store)
    snap2 = store2.snapshot()
    assert (len(snap2.nodes), len(snap2.edges)) == (n_nodes, n_edges)
    # Her kenarın kaynağı var
    for e in snap2.edges:
        if e.type in ("SUPPORTS", "CONTRADICTS", "INCREASES", "PROTECTS", "REQUIRES"):
            assert e.props["source_doi"] == SEED_DOI and e.props["source_sentence"]
            assert len(e.props["source_sentence"]) <= 300
    # Fare geni türe özgü düğüm
    assert any(k.startswith("Gene:mouse:") for k in snap2.entities())
    store.flush()
    assert json.loads((tmp_data / "data" / "graph.json").read_text())["ingest_runs"]


def test_inconsistencies_detected(tmp_data, fake_llm, seed_doc):
    _ingest(tmp_data, fake_llm, seed_doc)
    items = consistency.load_report(tmp_data / "reports")
    kinds = {i.kind for i in items}
    assert "n_mismatch_text_legend" in kinds        # metinde n = 12, legend'de n = 10
    assert "exclusion_conflict" in kinds            # "no animals were excluded" vs "two mice were excluded"
    assert "protocol_parameter_conflict" in kinds   # Methods 2 Hz, legend 4 Hz
    assert (tmp_data / "reports" / "inconsistencies.md").exists()


def test_rules_and_hypothesis_isolation(tmp_data, fake_llm, seed_doc):
    store, _ = _ingest(tmp_data, fake_llm, seed_doc)
    snap = store.snapshot()
    cands = rule_candidates(snap)
    triples = {(snap.label(c.subject), c.predicate, snap.label(c.object)) for c in cands}
    # müdahale → Hsp70 → nöron koruması zinciri
    assert ("Intervention", "PROTECTS", "Outcome") in triples

    scored = run_inference(store, cards=True)
    assert scored and all(s.hypothesis.path and s.hypothesis.weakest_link for s in scored)
    assert all(s.hypothesis.status.value == "untested" for s in scored)
    snap = store.snapshot()
    assert snap.hypotheses()
    assert not evidence_query_returns_hypotheses(snap)
    # Hipotezler bir sonraki çalıştırmada kanıt olarak kullanılmaz; tekilleştirme: aynı üçlü tek kenar
    run_inference(store, cards=False)
    snap2 = store.snapshot()
    assert len(snap2.hypotheses()) == len(snap.hypotheses())
    assert len(rule_candidates(snap2)) == len(cands)
    checks = acceptance.phase3(snap2)
    assert all(c["passed"] for c in checks), checks
    card = next((tmp_data / "reports" / "hypotheses").glob("*protects*.md")).read_text()
    assert "HİPOTEZDİR" in card and "En zayıf halka" in card and "klinik" in card


def test_spotcheck_flow(tmp_data, fake_llm, seed_doc):
    store, summary = _ingest(tmp_data, fake_llm, seed_doc)
    path, items = build_spotcheck(store, n=10, seed=1)
    assert path.exists() and len(items) == 3
    html = path.read_text()
    assert "Kaynak cümle" in html and "Sonuçları indir" in html
    results = {"run_id": summary.run.id, "items": [
        {"edge_id": it["edge_id"], "related_edge_ids": it["related_edge_ids"], "label": lbl}
        for it, lbl in zip(items, ["correct", "correct", "wrong"])]}
    out = apply_results(store, results)
    assert out["error_rate"] > 0.10 and not out["passed"]
    snap = store.snapshot()
    assert any(e.props.get("needs_review") for e in snap.edges)
    assert sum(1 for e in snap.edges if e.props.get("verified_by_human")) >= 2


def test_phase1_acceptance_report(tmp_data, fake_llm, seed_doc):
    store, _ = _ingest(tmp_data, fake_llm, seed_doc)
    checks = acceptance.phase1(store.snapshot(), SEED_DOI)
    by = {c["criterion"]: c for c in checks}
    assert by["Ana iddia: Piezo1 gerekliliği"]["passed"]
    assert by["Ana iddia: HSP70"]["passed"]
    assert by["Ana iddia: APOL11a"]["passed"] is False  # sentetik fikstürde yok → dürüstçe kaldı
    assert by["Spot-check hata oranı ≤ %10"]["passed"] is None
    path = acceptance.write_report(1, checks, out_dir=tmp_data / "reports")
    assert "geçmedi" in path.read_text()
