import re
import sys
import importlib
from types import SimpleNamespace

import pytest

from kg.llm import FakeLLM

DOCS = {
    "a.txt": "Başlık: Transformatör arızaları\n\nBu çalışmada güç transformatörü ve kesici arızaları incelendi. "
             "Transformatör sıcaklığı PSCAD ile modellendi. 154 kV şebekede kısa devre analizi yapıldı.",
    "b.txt": "Başlık: Güneş santrali\n\nFotovoltaik sistemde MPPT algoritması ve inverter verimi PSCAD üzerinde incelendi.",
    "c.txt": "Başlık: Rüzgâr türbini\n\nRüzgâr türbininde jeneratör ve inverter kontrolü için MPPT yöntemi önerildi.",
    "d.txt": "Başlık: Koruma\n\nKesici ve röle koordinasyonu kısa devre akımlarına göre yapıldı; transformatör korundu.",
    "e.txt": "Başlık: Dağıtım\n\nDağıtım şebekesinde harmonik bozulma ve inverter etkisi ölçüldü.",
    "f.txt": "Başlık: Yeni\n\nBir inverter ve bir transformatör PSCAD ile test edildi.",
}
GOLD = {"transformatör": "Ekipman", "güç transformatörü": "Ekipman", "kesici": "Ekipman", "inverter": "Ekipman",
        "jeneratör": "Ekipman", "röle": "Ekipman", "PSCAD": "Yazılım / Araç", "MPPT": "Yöntem / Teknik",
        "kısa devre": "Arıza / Olay", "154 kV": "Birim / Değer", "harmonik bozulma": "Arıza / Olay"}


def responder(system, user, schema):
    """Dil modeli taklidi: her METİN bölümünde sözlükteki ifadeleri (metindeki yazılışıyla) döndürür."""
    docs = []
    for i, body in re.findall(r"=== METİN (\d+) ===\n(.*?)(?=\n\n=== METİN |\Z)", user, re.S):
        ents = []
        for k, label in GOLD.items():
            m = re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", body, re.I)
            if m:
                ents.append({"text": m.group(), "label": label})
        ents.append({"text": "uydurma varlık", "label": "Ekipman"})  # metinde yok → reddedilmeli
        docs.append({"index": int(i), "entities": ents})
    return {"documents": docs}


@pytest.fixture
def proj(tmp_data, monkeypatch):
    import kg.okg.store as st
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    p = st.create_project("Elektrik", "test")
    ids = {}
    for name, text in DOCS.items():
        ids[name] = p.add_document(name, text.encode(), text)["id"]
    # a.txt için arşiv metadata grafında bir Yayın düğümü
    p.add_extraction(ids["a.txt"] + "#meta", [{"type": "Yayın", "name": "Transformatör arızaları",
                                                "mention": "Transformatör arızaları", "description": None}], [], [])
    return p, ids


def test_spans_windows_gliner_format():
    from kg.okg import ner
    t = "Ankara'da transformatör ve Transformatör. transformatörün"
    assert ner.find_spans(t, "transformatör") == [(10, 23)]  # birebir; ekli kelime eşleşmez
    assert ner.find_spans(t, "TRANSFORMATÖR") == [(10, 23), (27, 40)]  # birebir yoksa harf duyarsız
    long = " ".join(f"kelime{i}." for i in range(450))
    ws = ner.windows(long, 200)
    assert len(ws) == 3 and all(long[o:o + len(w)] == w for o, w in ws)
    ents = [{"start": 10, "end": 23, "label": "Ekipman"}, {"start": 0, "end": 6, "label": "Yer"}]
    ex = ner.to_gliner(t, ents)[0]
    assert ex["tokenized_text"][:3] == ["Ankara", "'", "da"]
    assert sorted(ex["ner"]) == [[0, 0, "Yer"], [3, 3, "Ekipman"]]
    assert ner.match_preset("Elektrik Mühendisliği") == "elektrik" and ner.match_preset("arkeoloji") is None


def test_pipeline_grouped_sozluk(proj):
    from kg.okg import ner
    p, ids = proj
    cfg = ner.load_config(p)
    cfg["labels"] = [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS["elektrik"]["labels"]]
    ner.save_config(p, cfg)
    assert "Ekipman" in p.ontology().type_names() and any(r.name == "BAHSEDER" for r in p.ontology().relation_types)

    llm = FakeLLM(responder)
    res = ner.annotate(p, llm, doc_ids=[ids[n] for n in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt")], per_request=2)
    assert res["added"] == 5 and res["rejected"] == 5
    assert len(llm.calls) == 3  # 5 belge, istek başına 2 → 3 istek
    rows = {r["doc_id"]: r for r in ner.load_dataset(p)}
    a = rows[ids["a.txt"]]
    assert {(e["text"], e["label"]) for e in a["entities"]} >= {("Transformatör", "Ekipman"), ("PSCAD", "Yazılım / Araç"),
                                                              ("154 kV", "Birim / Değer"), ("güç transformatörü", "Ekipman")}
    assert all(a["text"][e["start"]:e["end"]] == e["text"] for e in a["entities"])
    # grafa yazıldı ve Yayın düğümüne bağlandı
    g = p.graph()
    pub = "Yayın::transformatör arızaları"
    assert any(r["subject"] == pub and r["predicate"] == "BAHSEDER" and r["object"] == "Yazılım / Araç::pscad" for r in g["relations"].values())
    assert p.document(ids["a.txt"])["ner"] == "llm"
    # ikinci çalıştırma: etiketli belgeler atlanır
    assert ner.annotate(p, llm, doc_ids=[ids["a.txt"]])["added"] == 0

    m = ner.train(p, kind="sozluk")
    assert m["model_kind"] == "sozluk" and m["n_test"] == 1 and m["test"]["micro"]["f1"] > 0.5
    out = ner.apply(p)  # yalnızca etiketlenmemiş f.txt
    assert out["docs"] == 1 and out["entities"] >= 3
    f = p.graph()["entities"]
    assert any(m_["doc_id"] == ids["f.txt"] + "#ner" for e in f.values() for m_ in e["mentions"])
    assert p.document(ids["f.txt"])["ner"] == "model"
    # belge silinince NER katkısı da gider
    p.delete_document(ids["f.txt"])
    assert not any(m_["doc_id"].startswith(ids["f.txt"]) for e in p.graph()["entities"].values() for m_ in e["mentions"])
    assert ner.clear_ner(p) == 5


class _FakeBatches:
    def __init__(self):
        self.created, self.polls = None, 0

    def create(self, requests):
        self.created = requests
        return SimpleNamespace(id="msgbatch_1")

    def retrieve(self, bid):
        self.polls += 1
        counts = SimpleNamespace(succeeded=0, errored=0, canceled=0, expired=0, processing=len(self.created))
        return SimpleNamespace(processing_status="ended" if self.polls > 1 else "in_progress", request_counts=counts)

    def results(self, bid):
        import json
        for r in self.created:
            ans = responder("", r["params"]["messages"][0]["content"], None)
            msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(ans, ensure_ascii=False))])
            yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg))


def test_anthropic_batch_api(proj):
    from kg.okg import ner
    p, ids = proj
    cfg = ner.load_config(p)
    cfg["labels"] = [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS["elektrik"]["labels"]]
    ner.save_config(p, cfg)
    batches = _FakeBatches()
    llm = SimpleNamespace(client=SimpleNamespace(messages=SimpleNamespace(batches=batches)), model_name="claude-test", effort="high")
    res = ner.annotate(p, llm, n=6, per_request=3, poll_seconds=0)
    assert res["added"] == 6 and len(batches.created) == 2 and batches.polls == 2
    params = batches.created[0]["params"]
    assert params["output_config"]["format"]["type"] == "json_schema" and params["model"] == "claude-test"
    assert not (ner.ner_dir(p) / "state.json").exists()


@pytest.fixture
def webc(tmp_data, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("APP_PASSWORD", "s3cret-pass")
    monkeypatch.setenv("APP_USERNAME", "admin")
    monkeypatch.setenv("LLM_BACKEND", "manual")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    import kg.okg.store as st
    import kg.web.jobs as jobs_mod
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    monkeypatch.setattr(jobs_mod, "DATA_DIR", tmp_data / "data")
    sys.modules.pop("kg.web.app", None)
    app_mod = importlib.import_module("kg.web.app")
    c = TestClient(app_mod.app)
    c.auth = ("admin", "s3cret-pass")
    return c, app_mod


def test_ner_web(webc):
    c, app_mod = webc
    from test_web import _csrf, _new_project, _wait_job
    import kg.okg.store as st
    slug = _new_project(c, "Elektrik Arşivi")
    p = st.Project(slug)
    for name, text in DOCS.items():
        p.add_document(name, text.encode(), text)
    assert "Hızlı varlık modeli" in c.get(f"/p/{slug}/docs").text
    r = c.get(f"/p/{slug}/ner")
    assert r.status_code == 200 and "1. Varlık türleri" in r.text
    tok = _csrf(c, f"/p/{slug}/ner")
    r = c.post(f"/p/{slug}/ner/suggest", data={"csrf": tok, "domain": "elektrik mühendisliği", "how": "preset"}, follow_redirects=False)
    assert r.status_code == 303
    assert 'value="Ekipman"' in c.get(f"/p/{slug}/ner").text
    app_mod.get_llm = lambda: FakeLLM(responder)
    r = c.post(f"/p/{slug}/ner/annotate", data={"csrf": tok, "n": 5, "per_request": 5, "add_to_graph": "1"}, follow_redirects=False)
    job = r.headers["location"].split("/jobs/")[1].split("?")[0]
    assert _wait_job(c, job)["status"] == "done"
    page = c.get(f"/p/{slug}/ner").text
    assert "Veri seti: 5 belge" in page
    r = c.post(f"/p/{slug}/ner/train", data={"csrf": tok, "kind": "sozluk"}, follow_redirects=False)
    assert _wait_job(c, r.headers["location"].split("/jobs/")[1].split("?")[0])["status"] == "done"
    assert "Sözlük modeli" in c.get(f"/p/{slug}/ner").text
    r = c.post(f"/p/{slug}/ner/apply", data={"csrf": tok, "scope": "unlabeled", "threshold": 0.3}, follow_redirects=False)
    assert _wait_job(c, r.headers["location"].split("/jobs/")[1].split("?")[0])["status"] == "done"
    tr = c.post(f"/api/p/{slug}/ner/try", json={"csrf": tok, "text": "Yeni bir inverter PSCAD ile denendi."}).json()
    assert {e["text"] for e in tr["entities"]} >= {"inverter", "PSCAD"}
    assert c.post(f"/api/p/{slug}/ner/try", json={"csrf": "x", "text": "a"}).status_code == 403
    jl = c.get(f"/p/{slug}/ner/export/dataset.jsonl").text.splitlines()
    assert len(jl) == 5
    gl = c.get(f"/p/{slug}/ner/export/gliner.json").json()
    assert gl["train"] and gl["train"][0]["tokenized_text"] and "Ekipman" in gl["labels"]
    assert "NER" in c.get(f"/p/{slug}/docs").text
    r = c.post(f"/p/{slug}/ner/model-upload", data={"csrf": tok}, files={"file": ("m.zip", b"bozuk", "application/zip")}, follow_redirects=False)
    assert "err=" in r.headers["location"]
