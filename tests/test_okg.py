"""Ontoloji tabanlı KG çekirdeği: şema, çıkarım doğrulamaları, depo, standart ontoloji, Wikidata."""
import json

import pytest

from kg.llm import FakeLLM
from kg.okg import extractor, standards, wikidata
from kg.okg.documents import chunk_text, extract_text
from kg.okg.ontology import TEMPLATES, Ontology
from kg.okg.store import Project, create_project, list_projects
from kg.schema import strict_json_schema

TEXT = ("TEST BELGESİ (sentetik). Türkiye 1952 yılında NATO'ya katıldı. "
        "Ankara Antlaşması Türkiye ile Avrupa Ekonomik Topluluğu arasında imzalandı. "
        "Recep Tayyip Erdoğan Türkiye'yi yönetiyor.")

IR = TEMPLATES["uluslararasi"][1]


def fake_answer(entities=None, relations=None):
    ents = entities if entities is not None else [
        {"name": "Türkiye", "type": "Devlet", "mention": "Türkiye", "description": "Bir ülke"},
        {"name": "NATO", "type": "Uluslararası örgüt", "mention": "NATO", "description": None},
        {"name": "Recep Tayyip Erdoğan", "type": "Lider", "mention": "Recep Tayyip Erdoğan", "description": None},
        {"name": "Atlantis", "type": "Devlet", "mention": "Atlantis", "description": None},  # metinde yok
    ]
    rels = relations if relations is not None else [
        {"subject": "Türkiye", "predicate": "ÜYESİDİR", "object": "NATO",
         "source_sentence": "Türkiye 1952 yılında NATO'ya katıldı.", "confidence": 0.9},
        {"subject": "Recep Tayyip Erdoğan", "predicate": "YÖNETİR", "object": "Türkiye",
         "source_sentence": "Recep Tayyip Erdoğan Türkiye'yi yönetiyor.", "confidence": 0.8},
        {"subject": "NATO", "predicate": "ÜYESİDİR", "object": "Türkiye",  # tür kısıtı ihlali
         "source_sentence": "Türkiye 1952 yılında NATO'ya katıldı.", "confidence": 0.5},
        {"subject": "Türkiye", "predicate": "MÜTTEFİKTİR", "object": "NATO",  # tür kısıtı ihlali
         "source_sentence": "Uydurma cümle.", "confidence": 0.5},
    ]
    return {"entities": ents, "relations": rels}


@pytest.fixture
def proj(tmp_data, monkeypatch):
    import kg.okg.store as st
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    p = create_project("Dış Politika Testi", "deneme")
    p.save_ontology(IR)
    return p


def test_schema_enums_follow_ontology():
    schema = strict_json_schema(IR.extraction_schema())
    s = json.dumps(schema, ensure_ascii=False)
    assert '"Uluslararası örgüt"' in s and '"YAPTIRIM_UYGULAR"' in s and "$ref" not in s


def test_extract_chunk_validates(proj):
    llm = FakeLLM(lambda sys, user, schema: fake_answer())
    ents, rels, rej = extractor.extract_chunk(llm, IR, "doc", TEXT)
    assert {e["name"] for e in ents} == {"Türkiye", "NATO", "Recep Tayyip Erdoğan"}
    assert [(r["predicate"]) for r in rels] == ["ÜYESİDİR", "YÖNETİR"]
    reasons = {r["reason"] for r in rej}
    assert "metinde geçmiyor" in reasons
    assert any(r.startswith("tür kısıtına uymuyor") for r in reasons)
    assert "kaynak cümle metinde birebir geçmiyor" in reasons


def test_process_is_idempotent_and_delete(proj):
    doc = proj.add_document("t.txt", TEXT.encode(), TEXT)
    llm = FakeLLM(lambda sys, user, schema: fake_answer())
    extractor.process_document(proj, doc["id"], llm)
    g1 = proj.graph()
    extractor.process_document(proj, doc["id"], llm)
    g2 = proj.graph()
    assert len(g1["entities"]) == len(g2["entities"]) == 3 and len(g2["relations"]) == 2
    assert proj.document(doc["id"])["status"] == "done"
    # aynı varlık ikinci belgede: tek düğüm, iki kaynak
    doc2 = proj.add_document("u.txt", TEXT.encode(), TEXT)
    extractor.process_document(proj, doc2["id"], llm)
    g3 = proj.graph()
    assert len(g3["entities"]) == 3
    rel = next(r for r in g3["relations"].values() if r["predicate"] == "ÜYESİDİR")
    assert {s["doc_id"] for s in rel["sources"]} == {doc["id"], doc2["id"]}
    proj.delete_document(doc["id"])
    proj.delete_document(doc2["id"])
    assert proj.graph()["entities"] == {} and proj.graph()["relations"] == {}
    assert list_projects()[0]["name"] == "Dış Politika Testi"


def test_suggest_ontology():
    answer = {"domain": "Uİ", "entity_types": [{"name": "Devlet", "description": "ülke"}, {"name": "Örgüt", "description": ""}],
              "relation_types": [{"name": "üyesidir", "description": "", "source": ["Devlet", "Yok"], "target": ["Örgüt"]}]}
    onto = extractor.suggest_ontology(FakeLLM(lambda *a: answer), TEXT)
    assert onto.relation_types[0].name == "ÜYESİDİR" and onto.relation_types[0].source == ["Devlet"]


def test_documents_and_chunks():
    assert extract_text("a.txt", "merhaba\n\n\n\ndünya".encode()) == "merhaba\n\ndünya"
    with pytest.raises(ValueError):
        extract_text("a.exe", b"x")
    parts = chunk_text("a. " * 5000, 1000)
    assert all(len(p) <= 1000 for p in parts) and len(parts) > 10


TTL = b"""@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/> .
ex:Person a owl:Class ; rdfs:label "Person"@en, "Kisi"@tr ; rdfs:comment "A human"@en .
ex:Organization a owl:Class ; rdfs:label "Organization"@en .
ex:worksFor a owl:ObjectProperty ; rdfs:label "worksFor" ; rdfs:domain ex:Person ; rdfs:range ex:Organization .
ex:name a owl:DatatypeProperty ; rdfs:label "name" .
"""


def test_standard_ontology_import():
    parsed = standards.parse_ontology_file("x.ttl", TTL)
    names = [c["name"] for c in parsed["classes"]]
    assert "Kisi" in names and "Organization" in names
    assert [p["name"] for p in parsed["properties"]] == ["worksFor"]
    onto = standards.merge_selection(Ontology(), parsed, ["http://example.org/Person", "http://example.org/Organization"],
                                     ["http://example.org/worksFor"])
    assert onto.entity_types[0].uri == "http://example.org/Person"
    r = onto.relation_types[0]
    assert r.name == "WORKSFOR" and r.source == ["Kisi"] and r.target == ["Organization"]
    with pytest.raises(ValueError):
        standards.parse_ontology_file("bad.ttl", b"bu ttl degil {{{")


class FakeHttp:
    def get(self, url, params=None, headers=None, **kw):
        if "api.php" in url:
            q = params["search"]
            ids = {"Türkiye": "Q43", "NATO": "Q7184", "Recep Tayyip Erdoğan": "Q39723"}
            return {"search": [{"id": ids[q], "label": q, "description": "test"}]} if q in ids else {"search": []}
        return {"results": {"bindings": [
            {"a": {"value": "http://www.wikidata.org/entity/Q43"}, "b": {"value": "http://www.wikidata.org/entity/Q7184"},
             "prop": {"value": "http://www.wikidata.org/entity/P463"}, "propLabel": {"value": "üyesi"}}]}}


def test_wikidata_link_and_enrich(proj):
    doc = proj.add_document("t.txt", TEXT.encode(), TEXT)
    extractor.process_document(proj, doc["id"], FakeLLM(lambda *a: fake_answer()))
    out = wikidata.link_entities(proj, http=FakeHttp())
    assert out["linked"] == 3
    res = wikidata.enrich(proj, http=FakeHttp())
    assert res["relations"] == 1
    g = proj.graph()
    wd = [r for r in g["relations"].values() if r["predicate"] == "WD:üyesi"]
    assert wd and wd[0]["sources"][0]["doc_id"] == "wikidata"
    # yeniden çalıştırmak çoğaltmaz; belge yeniden işlenince Wikidata ilişkisi kalır
    wikidata.enrich(proj, http=FakeHttp())
    extractor.process_document(proj, doc["id"], FakeLLM(lambda *a: fake_answer()))
    assert len([r for r in proj.graph()["relations"].values() if r["predicate"] == "WD:üyesi"]) == 1
    key = Project.entity_key("Devlet", "Türkiye")
    wikidata.set_link(proj, key, None)
    assert "wikidata" not in proj.graph()["entities"][key]


def test_http_honours_retry_after_and_user_agent(tmp_path, monkeypatch):
    import httpx

    from kg import http as http_mod
    sleeps, seen_ua = [], []
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def handler(request):
        seen_ua.append(request.headers["user-agent"])
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"ok": True})

    h = http_mod.Http(cache_dir=tmp_path)
    h.client = httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": http_mod.user_agent()})
    assert h.get("https://www.wikidata.org/w/api.php", params={"a": 1}) == {"ok": True}
    assert 7.0 in sleeps  # Retry-After süresi kadar beklendi
    assert "github.com/srht/makalekg" in seen_ua[0]  # Wikimedia UA politikası: iletişim bilgisi

    h2 = http_mod.Http(cache_dir=tmp_path, max_attempts=2)
    h2.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(http_mod.RateLimited):
        h2.get("https://www.wikidata.org/w/api.php", params={"b": 2})


def test_wikidata_link_resumes_after_rate_limit(proj):
    from kg.http import RateLimited

    doc = proj.add_document("t.txt", TEXT.encode(), TEXT)
    extractor.process_document(proj, doc["id"], FakeLLM(lambda *a: fake_answer()))

    class Flaky(FakeHttp):
        def __init__(self, fail_after):
            self.n, self.fail_after = 0, fail_after

        def get(self, url, params=None, headers=None, **kw):
            self.n += 1
            if self.n > self.fail_after:
                raise RateLimited(url, 30)
            return super().get(url, params, headers)

    old = wikidata.SAVE_EVERY
    wikidata.SAVE_EVERY = 1
    try:
        with pytest.raises(wikidata.WikidataBusy) as ei:
            wikidata.link_entities(proj, http=Flaky(fail_after=1))
        assert "kaldığı yerden" in str(ei.value)
        linked = [e for e in proj.graph()["entities"].values() if e.get("wikidata")]
        assert len(linked) == 1  # ilk varlık kaydedildi
        out = wikidata.link_entities(proj, http=Flaky(fail_after=99))
        assert out["linked"] == 2 and out["skipped"] == 1  # kalanlar eşlendi, ilki tekrar aranmadı
    finally:
        wikidata.SAVE_EVERY = old
