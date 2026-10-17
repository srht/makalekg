"""Web arayüzü: proje → ontoloji → belge → graf akışı (kopyala–yapıştır modu, ağ erişimi yok)."""
import importlib
import json
import re
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from test_okg import TEXT, TTL, fake_answer  # noqa: E402


@pytest.fixture
def client(tmp_data, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret-pass")
    monkeypatch.setenv("APP_USERNAME", "admin")
    monkeypatch.setenv("LLM_BACKEND", "manual")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import kg.okg.store as st
    import kg.web.jobs as jobs_mod
    monkeypatch.setattr(st, "DATA_DIR", tmp_data / "data")
    monkeypatch.setattr(jobs_mod, "DATA_DIR", tmp_data / "data")
    sys.modules.pop("kg.web.app", None)
    app_mod = importlib.import_module("kg.web.app")
    c = TestClient(app_mod.app)
    c.auth = ("admin", "s3cret-pass")
    return c


def _csrf(c, path="/"):
    return re.search(r'name="csrf" value="([a-f0-9]+)"', c.get(path).text).group(1)


def _wait_pending(c, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        pid = c.get("/api/manual").json()["pending"]
        if pid:
            return pid
        time.sleep(0.05)
    raise AssertionError("istem gelmedi")


def _wait_job(c, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        d = c.get(f"/api/jobs/{job_id}").json()
        if d["status"] not in ("queued", "running", "waiting"):
            return d
        time.sleep(0.05)
    raise AssertionError("iş bitmedi")


def _new_project(c, name="Dış Politika"):
    r = c.post("/projects", data={"csrf": _csrf(c), "name": name, "description": "test"}, follow_redirects=False)
    assert r.status_code == 303
    return r.headers["location"].split("/")[2]


def test_auth(client):
    assert client.get("/healthz", auth=None).status_code == 200
    assert client.get("/", auth=None).status_code == 401
    assert client.get("/", auth=("admin", "yanlis")).status_code == 401
    page = client.get("/").text
    assert "Belgelerinden bir bilgi ağı" in page  # ilk açılış: tanıtım + proje oluştur


def test_full_flow_manual(client):
    c = client
    slug = _new_project(c)
    assert "Nasıl başlamak istersin?" in c.get(f"/p/{slug}/ontology").text
    tok = _csrf(c, f"/p/{slug}/ontology")
    c.post(f"/p/{slug}/ontology/template", data={"csrf": tok, "key": "uluslararasi"})
    page = c.get(f"/p/{slug}/ontology").text
    assert "Uluslararası örgüt" in page and "YAPTIRIM_UYGULAR" in page
    # tablo düzenleme: bir tür ekle
    onto = json.loads((c.get(f"/p/{slug}/export/json").text))["ontology"]
    form = {"csrf": tok, "domain": onto["domain"],
            "ent_name": [t["name"] for t in onto["entity_types"]] + ["Şirket"],
            "ent_desc": [t["description"] for t in onto["entity_types"]] + ["Enerji şirketleri"],
            "ent_uri": [""] * (len(onto["entity_types"]) + 1),
            "rel_name": [r["name"] for r in onto["relation_types"]],
            "rel_desc": [r["description"] for r in onto["relation_types"]],
            "rel_source": [", ".join(r["source"]) for r in onto["relation_types"]],
            "rel_target": [", ".join(r["target"]) for r in onto["relation_types"]],
            "rel_uri": [""] * len(onto["relation_types"])}
    c.post(f"/p/{slug}/ontology", data=form)
    assert "Enerji şirketleri" in c.get(f"/p/{slug}/ontology").text

    # belge yükle + hemen işle → kopyala–yapıştır istemi
    r = c.post(f"/p/{slug}/docs/upload", data={"csrf": tok, "process": "1"},
               files=[("files", ("belge.txt", TEXT.encode("utf-8"), "text/plain"))], follow_redirects=False)
    assert r.status_code == 303
    pid = _wait_pending(c)
    manual = c.get("/manual").text
    assert "Varlık ve ilişki çıkarımı" in manual and "NATO" in manual and "JSON ŞEMASI" in manual
    assert "Sıra sende" in c.get(f"/p/{slug}/docs").text
    # önce bozuk yanıt → hata gösterilir, sonra doğru yanıt
    c.post(f"/manual/{pid}", data={"csrf": tok, "answer": "üzgünüm, yapamam"})
    assert "kabul edilmedi" in c.get("/manual").text
    answer = "İşte sonuç:\n```json\n" + json.dumps(fake_answer(), ensure_ascii=False) + "\n```"
    r = c.post(f"/manual/{pid}", data={"csrf": tok, "answer": answer}, follow_redirects=False)
    loc = r.headers["location"]
    if "/jobs/" in loc:
        assert _wait_job(c, loc.rsplit("/", 1)[1])["status"] == "done"
    else:
        assert loc == f"/p/{slug}/docs"  # iş bitti → doğrudan belgelere dönülür

    docs = c.get(f"/p/{slug}/docs").text
    assert "işlendi" in docs
    g = c.get(f"/api/p/{slug}/graph").json()
    assert {n["label"] for n in g["nodes"]} == {"Türkiye", "NATO", "Recep Tayyip Erdoğan"}
    assert len(g["edges"]) == 2 and "sources" not in g["edges"][0]  # harita verisi hafif; ayrıntı ayrıca çekilir
    edge = c.get(f"/api/p/{slug}/edge", params={"id": g["edges"][0]["id"]}).json()
    assert edge["sources"][0]["doc"] == "belge.txt"
    node = c.get(f"/api/p/{slug}/node", params={"id": g["nodes"][0]["id"]}).json()
    assert node["mentions"] and node["mentions"][0]["doc"] == "belge.txt"
    assert c.get(f"/api/p/{slug}/node", params={"id": "yok"}).status_code == 404
    # ETag: değişmeyen graf tekrar gönderilmez
    r = c.get(f"/api/p/{slug}/graph")
    assert c.get(f"/api/p/{slug}/graph", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    # yerleşim kaydı: konumlar sonraki yüklemede veriyle gelir
    pos = {n["id"]: [i * 10.0, -5.5] for i, n in enumerate(g["nodes"])}
    assert c.post(f"/api/p/{slug}/layout", json={"csrf": "x", "pos": pos}).status_code == 403
    assert c.post(f"/api/p/{slug}/layout", json={"csrf": _csrf(c), "pos": {**pos, "olmayan": [1, 2]}}).json()["n"] == 3
    g2 = c.get(f"/api/p/{slug}/graph", headers={"If-None-Match": r.headers["etag"]}).json()
    assert all("x" in n and "y" in n for n in g2["nodes"])
    assert c.get(f"/api/p/{slug}/graph", params={"full": 1}).json()["edges"][0]["sources"]
    gp = c.get(f"/p/{slug}/graph").text
    assert "vis-network.min.js" in gp and "Wikidata" in gp
    assert c.get("/static/vendor/vis-network.min.js").status_code == 200
    a = next(n["id"] for n in g["nodes"] if n["label"] == "Recep Tayyip Erdoğan")
    b = next(n["id"] for n in g["nodes"] if n["label"] == "NATO")
    path = c.get(f"/api/p/{slug}/path", params={"a": a, "b": b}).json()
    assert path["found"] and len(path["edges"]) == 2
    doc_id = c.get(f"/api/p/{slug}/status").json()["docs"][0]["id"]
    detail = c.get(f"/p/{slug}/docs/{doc_id}").text
    assert "Reddedilenler" in detail and "tür kısıtına uymuyor" in detail
    # dışa aktarım
    assert "Türkiye" in c.get(f"/p/{slug}/export/entities.csv").text
    assert "ÜYESİDİR" in c.get(f"/p/{slug}/export/relations.csv").text
    ttl = c.get(f"/p/{slug}/export/ttl").text
    assert "rdfs:label" in ttl and "Türkiye" in ttl
    # proje anasayfası artık grafa yönlendirir
    assert c.get(f"/p/{slug}", follow_redirects=False).headers["location"].endswith("/graph")


def test_ontology_import_and_suggest(client):
    c = client
    slug = _new_project(c, "Genel")
    tok = _csrf(c, f"/p/{slug}/ontology")
    r = c.post(f"/p/{slug}/ontology/import", data={"csrf": tok}, files={"file": ("o.ttl", TTL, "text/turtle")},
               follow_redirects=False)
    assert r.headers["location"].endswith("/ontology/import")
    page = c.get(f"/p/{slug}/ontology/import").text
    assert "Kisi" in page and "worksFor" in page
    c.post(f"/p/{slug}/ontology/import/apply", data={"csrf": tok, "cls": ["http://example.org/Person", "http://example.org/Organization"],
                                                     "prop": ["http://example.org/worksFor"]})
    page = c.get(f"/p/{slug}/ontology").text
    assert "WORKSFOR" in page and "standart ↗" in page
    # belgeden öneri (kopyala–yapıştır)
    r = c.post(f"/p/{slug}/ontology/suggest", data={"csrf": tok, "hint": "dış politika"},
               files={"file": ("ornek.txt", TEXT.encode(), "text/plain")}, follow_redirects=False)
    job_id = r.headers["location"].split("/jobs/")[1].split("?")[0]
    pid = _wait_pending(c)
    sug = {"domain": "Uİ", "entity_types": [{"name": "Devlet", "description": "ülke"}],
           "relation_types": [{"name": "komşudur", "description": "", "source": ["Devlet"], "target": ["Devlet"]}]}
    c.post(f"/manual/{pid}", data={"csrf": tok, "answer": json.dumps(sug)})
    assert _wait_job(c, job_id)["status"] == "done"
    page = c.get(f"/p/{slug}/ontology").text
    assert "Önerilen ontoloji hazır" in page and "KOMŞUDUR" in page
    c.post(f"/p/{slug}/ontology/suggestion", data={"csrf": tok, "action": "apply"})
    assert "Önerilen ontoloji hazır" not in c.get(f"/p/{slug}/ontology").text


def test_csrf_and_errors(client):
    c = client
    slug = _new_project(c, "X")
    assert c.post(f"/p/{slug}/ontology/template", data={"csrf": "bad", "key": "genel"}).status_code == 403
    assert c.get("/p/yok-boyle-proje/graph").status_code == 404
    tok = _csrf(c, f"/p/{slug}/docs")
    r = c.post(f"/p/{slug}/docs/upload", data={"csrf": tok}, files=[("files", ("a.exe", b"x", "application/octet-stream"))],
               follow_redirects=True)
    assert "desteklenmeyen dosya türü" in r.text
    # ontoloji yokken işleme ontolojiye yönlendirir
    r = c.post(f"/p/{slug}/docs/process", data={"csrf": tok}, follow_redirects=False)
    assert r.headers["location"].endswith("/ontology")
    # proje silme adı ister
    assert c.post(f"/p/{slug}/delete", data={"csrf": tok, "confirm": "yanlış"}).status_code == 400
    c.post(f"/p/{slug}/delete", data={"csrf": tok, "confirm": "X"})
    assert c.get(f"/p/{slug}/graph").status_code == 404


def test_templates_are_tracked_by_git():
    """Şablonlar git'e girmezse dağıtılan imajda eksik kalır."""
    import shutil
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if not shutil.which("git") or not (root / ".git").exists():
        pytest.skip("git yok")
    tpl_dir = root / "kg" / "web" / "templates"
    names = {p.name for p in tpl_dir.glob("*.html")}
    used = set(re.findall(r'"([\w_]+\.html)"', (root / "kg" / "web" / "app.py").read_text()))
    for t in tpl_dir.glob("*.html"):
        used |= set(re.findall(r'(?:include|extends) "([\w_]+\.html)"', t.read_text()))
    assert used <= names, f"eksik şablon: {used - names}"
    files = [str(tpl_dir / n) for n in sorted(names)] + [str(root / "kg/web/static/vendor/vis-network.min.js")]
    ignored = subprocess.run(["git", "check-ignore", *files], cwd=root, capture_output=True, text=True).stdout.split()
    assert not ignored, f".gitignore dışlıyor: {ignored}"


def test_oai_harvest_flow(client, monkeypatch, tmp_path):
    import httpx

    import kg.okg.oai as oai_mod
    from kg.http import Http
    from test_oai import make_server

    state = {}

    def mock_http(*a, **k):
        h = Http(cache_dir=tmp_path / "c", min_interval=0)
        h.client = httpx.Client(transport=httpx.MockTransport(make_server(state)))
        return h

    monkeypatch.setattr(oai_mod, "Http", mock_http)
    c = client
    slug = _new_project(c, "Arşiv")
    tok = _csrf(c, f"/p/{slug}/ontology")
    c.post(f"/p/{slug}/ontology/template", data={"csrf": tok, "key": "uluslararasi"})
    d = c.get("/api/oai/sets", params={"base": "https://arsiv.ornek.edu.tr/oai/request"}).json()
    assert d["ok"] and d["name"] == "Örnek Açık Arşiv" and d["sets"][0]["name"] == "Tezler"
    assert c.get("/api/oai/sets", params={"base": "ftp://x"}).json()["ok"] is False
    assert "Kurumsal arşivden çek" in c.get(f"/p/{slug}/docs").text

    r = c.post(f"/p/{slug}/oai/harvest", data={"csrf": tok, "base": "https://arsiv.ornek.edu.tr/oai/request",
                                                "set_spec": "col_1", "limit": "10", "meta_graph": "1", "process": "1"},
               follow_redirects=False)
    job_id = r.headers["location"].rsplit("/", 1)[1]
    pid = _wait_pending(c)
    prompt = c.get(f"/manual/{pid}/prompt.md").text
    assert "Türk dış politikasında NATO" in prompt and "Enerji güvenliği" in prompt  # iki kayıt tek istemde
    ans = {"entities": [{"name": "Türkiye", "type": "Devlet", "mention": "Türkiye", "description": None},
                        {"name": "NATO", "type": "Uluslararası örgüt", "mention": "NATO", "description": None}],
           "relations": [{"subject": "Türkiye", "predicate": "ÜYESİDİR", "object": "NATO",
                          "source_sentence": "Bu tez Türkiye ile NATO ilişkisini inceler.", "confidence": 0.6}]}
    c.post(f"/manual/{pid}", data={"csrf": tok, "answer": json.dumps(ans, ensure_ascii=False)})
    assert _wait_job(c, job_id)["status"] == "done"
    assert any(q.get("set") == "col_1" for q in state["calls"])
    st = c.get(f"/api/p/{slug}/status").json()["docs"]
    assert len(st) == 2 and all(x["status"] == "done" for x in st)
    g = c.get(f"/api/p/{slug}/graph").json()
    origins = {e["origin"] for e in g["edges"]}
    assert origins == {"metadata", "document"}
    doc_edge = next(e for e in g["edges"] if e["origin"] == "document")
    doc_edge = c.get(f"/api/p/{slug}/edge", params={"id": doc_edge["id"]}).json()
    assert doc_edge["sources"][0]["doc"].startswith("Türk dış politikasında NATO")  # doğru kayda atandı
    doc_id = next(x["id"] for x in st)
    page = c.get(f"/p/{slug}/docs/{doc_id}").text
    assert "Arşiv kaydı" in page and "dc:creator" in page


def test_search_page(client):
    import kg.okg.store as st
    slug = _new_project(client, "Arama")
    p = st.Project(slug)
    doc = p.add_document("a.txt", b"x", "Ayse Yilmaz Ankara Universitesi'nde calisir. <b>kalin</b>")
    ents = [{"type": "Kisi", "name": "Ayşe Yılmaz", "mention": "Ayse Yilmaz", "description": None},
            {"type": "Kurum", "name": "Ankara Üniversitesi", "mention": "Ankara Universitesi", "description": None}]
    p.add_extraction(doc["id"], ents, [{"subject_key": p.entity_key("Kisi", "Ayşe Yılmaz"), "predicate": "CALISIR",
                                         "object_key": p.entity_key("Kurum", "Ankara Üniversitesi"),
                                         "sentence": "Ayse Yilmaz Ankara Universitesi'nde calisir. <b>kalin</b>", "confidence": 0.9}], [])
    r = client.get(f"/p/{slug}/search")
    assert r.status_code == 200 and "Grafta ara" in r.text and "Ankara Üniversitesi" in r.text  # örnek çipler
    r = client.get(f"/p/{slug}/search", params={"q": "ankara çalışır"})
    assert r.status_code == 200
    assert "1 ilişki bulundu" in r.text and "<mark>Ankara</mark>" in r.text
    assert "<b>kalin</b>" not in r.text and "&lt;b&gt;kalin" in r.text  # kaynak cümle kaçışlı
    api = client.get(f"/api/p/{slug}/search", params={"q": "yilmaz"}).json()
    assert api["total"] == 1 and api["relations"][0]["sources"][0]["doc_id"] == doc["id"]
    key = api["relations"][0]["subject"]["key"]
    r = client.get(f"/p/{slug}/search", params={"entity": key})
    assert "varlığının ilişkileri" in r.text and "1 ilişki bulundu" in r.text
    csv_text = client.get(f"/p/{slug}/search.csv", params={"q": "ankara"}).text
    assert "CALISIR" in csv_text and "Ayşe Yılmaz" in csv_text
    assert "0 ilişki bulundu" in client.get(f"/p/{slug}/search", params={"q": "zzzz"}).text
    assert client.get("/p/yok/search", params={"q": "a"}).status_code == 404
