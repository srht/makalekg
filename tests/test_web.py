"""Web arayüzü: kimlik doğrulama, CSRF, sayfalar, iş kuyruğu (bellek-içi depo, sahte LLM ile tohumlanmış graf)."""
import importlib
import re
import sys
import time

import pytest
from conftest import SEED_DOI

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_data, fake_llm, seed_doc, monkeypatch):
    monkeypatch.setenv("GRAPH_BACKEND", "memory")
    monkeypatch.setenv("APP_PASSWORD", "s3cret-pass")
    monkeypatch.setenv("APP_USERNAME", "admin")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from kg.graph import MemoryStore
    from kg.pipeline import Pipeline
    from kg.schema import Paper
    from kg import service
    store = MemoryStore()
    Pipeline(store, fake_llm, offline=True).ingest(
        [(Paper(doi=SEED_DOI, year=2026, lab="A"), None)], documents={SEED_DOI: seed_doc})
    service.spotcheck_build(store, seed=1)
    service.infer(store)
    sys.modules.pop("kg.web.app", None)
    import kg.web.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "DATA_DIR", tmp_data / "data")
    app_mod = importlib.import_module("kg.web.app")
    monkeypatch.setattr(app_mod, "DATA_DIR", tmp_data / "data")
    monkeypatch.setattr(app_mod, "REPORTS_DIR", tmp_data / "reports")
    c = TestClient(app_mod.app)
    c.auth = ("admin", "s3cret-pass")
    return c


def _csrf(c):
    return re.search(r'name="csrf" value="([a-f0-9]+)"', c.get("/").text).group(1)


def _wait(c, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        d = c.get(f"/api/jobs/{job_id}").json()
        if d["status"] not in ("queued", "running"):
            return d
        time.sleep(0.1)
    raise AssertionError("iş bitmedi")


def test_auth_required(client):
    assert client.get("/healthz", auth=None).status_code == 200
    assert client.get("/", auth=None).status_code == 401
    assert client.get("/", auth=("admin", "yanlis")).status_code == 401
    assert client.get("/").status_code == 200


def test_pages_render(client):
    for path in ["/", "/claims", "/claims?q=piezo1", "/hypotheses", "/ask", "/spotcheck", "/review", "/reports", "/jobs"]:
        r = client.get(path)
        assert r.status_code == 200, path
    assert "Piezo1" in client.get("/claims").text
    hyp = re.search(r'href="(/hypotheses/e_[a-f0-9]+)"', client.get("/hypotheses").text).group(1)
    page = client.get(hyp).text
    assert "HİPOTEZDİR" in page and "En zayıf halka" in page
    assert "Alzheimer" in client.get("/review").text  # reddedilen uydurma çıkarım kuyrukta
    assert client.get("/reports/view/inconsistencies.md").status_code == 200
    assert client.get("/reports/view/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_csrf_and_jobs(client):
    assert client.post("/jobs/accept", data={"csrf": "x", "phase": 3}, follow_redirects=False).status_code == 403
    tok = _csrf(client)
    r = client.post("/jobs/accept", data={"csrf": tok, "phase": 3}, follow_redirects=False)
    assert r.status_code == 303
    d = _wait(client, r.headers["location"].rsplit("/", 1)[1])
    assert d["status"] == "done" and all(c["passed"] for c in d["result"])
    # LLM anahtarı yokken soru: iş anlaşılır hatayla başarısız olur, sunucu ayakta kalır
    r = client.post("/ask", data={"csrf": tok, "question": "Hsp70 ne yapar?"}, follow_redirects=False)
    d = _wait(client, r.headers["location"].rsplit("/", 1)[1])
    assert d["status"] == "failed" and "ANTHROPIC_API_KEY" in d["error"]


def test_spotcheck_submit(client):
    tok = _csrf(client)
    r = client.post("/spotcheck", data={"csrf": tok, "label_0": "correct", "label_1": "correct", "label_2": "correct"},
                    follow_redirects=False)
    d = _wait(client, r.headers["location"].rsplit("/", 1)[1])
    assert d["status"] == "done" and d["result"]["passed"] and d["result"]["n_labeled"] == 3
    assert "Eşik geçildi" in client.get("/spotcheck").text


def test_pdf_upload_validation(client):
    tok = _csrf(client)
    r = client.post("/jobs/ingest", data={"csrf": tok, "phase": "1"},
                    files={"pdf": ("x.pdf", b"not a pdf", "application/pdf")}, follow_redirects=False)
    assert r.status_code == 400


@pytest.fixture
def manual_client(client, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "manual")
    return client


def test_manual_copy_paste_flow(manual_client):
    c = manual_client
    tok = _csrf(c)
    assert "Kopyala–yapıştır modu açık" in c.get("/").text
    r = c.post("/ask", data={"csrf": tok, "question": "Hsp70 nöronları korur mu?"}, follow_redirects=False)
    job_id = r.headers["location"].rsplit("/", 1)[1]
    end = time.time() + 10
    while not c.get("/api/manual").json()["pending"]:
        assert time.time() < end
        time.sleep(0.05)
    req_id = c.get("/api/manual").json()["pending"]
    page = c.get("/manual").text
    assert "İstemi kopyala" in page and "Hsp70" in page and "GRAF KANITI" in page
    assert "Yanıtınız bekleniyor" in c.get("/claims").text
    assert c.get(f"/api/jobs/{job_id}").json()["status"] == "waiting"
    assert "GRAF KANITI" in c.get(f"/manual/{req_id}/prompt.md").text
    # boş yanıt reddedilir, istem beklemede kalır
    c.post(f"/manual/{req_id}", data={"csrf": tok, "answer": "   "})
    assert c.get("/api/manual").json()["pending"] == req_id
    assert "kabul edilmedi" in c.get("/manual").text
    r = c.post(f"/manual/{req_id}", data={"csrf": tok, "answer": "Sadece farede gösterildi (Fig. 3a)."},
               follow_redirects=False)
    d = _wait(c, job_id)
    assert d["status"] == "done" and "farede" in d["result"]["answer"]
    assert c.get("/api/manual").json()["pending"] is None


def test_manual_cancel(manual_client):
    c = manual_client
    tok = _csrf(c)
    r = c.post("/ask", data={"csrf": tok, "question": "Piezo1?"}, follow_redirects=False)
    job_id = r.headers["location"].rsplit("/", 1)[1]
    end = time.time() + 10
    while not c.get("/api/manual").json()["pending"]:
        assert time.time() < end
        time.sleep(0.05)
    req_id = c.get("/api/manual").json()["pending"]
    c.post(f"/manual/{req_id}/cancel", data={"csrf": tok})
    d = _wait(c, job_id)
    assert d["status"] == "failed" and "iptal" in d["error"]


def test_guide_on_dashboard(client):
    page = client.get("/").text
    assert "Sıradaki adım" in page and "Çıkarımları kontrol et" in page  # tohum işlendi → spot-check sırada
    tok = _csrf(client)
    r = client.post("/guide/approve", data={"csrf": tok, "what": "phase1"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.post("/guide/approve", data={"csrf": "x", "what": "phase1"}).status_code == 403
    # Düğmelerin içine HTML sızmadığını doğrula (eski hata)
    assert "<button <div" not in page and "<button {" not in page
