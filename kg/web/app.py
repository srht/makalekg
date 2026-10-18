"""Web arayüzü: ontoloji tabanlı knowledge graph kurucu.

Akış: Projeler → (1) Ontoloji → (2) Belgeler → (3) Graf.
Uzun işler (belge işleme, ontoloji önerisi, Wikidata) arka plan kuyruğunda çalışır. Kopyala–yapıştır
modunda iş modelin yanıtını beklerken istem /manual sayfasında gösterilir.

Güvenlik: tüm sayfalar HTTP Basic ile korunur (APP_USERNAME / APP_PASSWORD); POST formları CSRF jetonu ister.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import logging
import re
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import BaseModel

from ..config import RUNTIME_KEYS, SECRET_KEYS, ensure_dirs, env, load_dotenv, runtime_settings, save_runtime_settings, setting
from ..llm import DEFAULT_GEMINI_MODEL, get_llm, set_manual_broker
from ..okg import extractor, ner, oai, search as gsearch, standards, wikidata
from ..okg.documents import SUPPORTED, UnsupportedFile, extract_text
from ..okg.ontology import TEMPLATES, EntityType, Ontology, RelationType
from ..okg.store import Project, create_project, list_projects
from .jobs import JobManager
from .manual import WebBroker

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("kg.web")

HERE = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
ensure_dirs()


@asynccontextmanager
async def lifespan(_app):
    yield


app = FastAPI(title="Knowledge Graph Kurucu", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.add_middleware(GZipMiddleware, minimum_size=1024)  # graf verisi ve vis-network.js sıkıştırılarak gönderilir


@app.middleware("http")
async def static_cache(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/vendor/"):
        resp.headers["Cache-Control"] = "public, max-age=604800"  # sürümü sabit kütüphaneler: bir hafta önbellekte
    return resp
templates = Jinja2Templates(directory=HERE / "templates")
CSRF_SECRET = (env("APP_SECRET") or secrets.token_hex(32)).encode()

jobs = JobManager()
broker = WebBroker(jobs)
set_manual_broker(broker)


def csrf_token(user: str) -> str:
    return hmac.new(CSRF_SECRET, user.encode(), "sha256").hexdigest()


# ---------------------------------------------------------------- kimlik doğrulama


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path == "/healthz":
        return await call_next(request)
    if env("KG_WEB_NO_AUTH") == "1":
        request.state.user = "local"
        return await call_next(request)
    password, username = env("APP_PASSWORD"), env("APP_USERNAME", "admin")
    if not password:
        return HTMLResponse("<h1>APP_PASSWORD ayarlanmadı</h1><p>Dokploy → Environment'ta APP_PASSWORD tanımlayın.</p>",
                            status_code=503)
    header = request.headers.get("authorization", "")
    ok = False
    if header.lower().startswith("basic "):
        try:
            user, _, pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            ok = secrets.compare_digest(user, username) and secrets.compare_digest(pw, password)
        except Exception:
            ok = False
    if not ok:
        return Response("Kimlik doğrulama gerekli", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="kg", charset="UTF-8"'})
    request.state.user = username
    return await call_next(request)


def check_csrf(request: Request, token: str) -> None:
    if not secrets.compare_digest(token or "", csrf_token(request.state.user)):
        raise HTTPException(403, "Geçersiz form jetonu; sayfayı yenileyip tekrar deneyin.")


def llm_status() -> dict[str, Any]:
    backend = (setting("LLM_BACKEND", "anthropic") or "anthropic").lower()
    if backend == "manual":
        return {"backend": backend, "ready": True, "label": "Kopyala–yapıştır (Claude.ai vb.)", "manual": True}
    if backend == "anthropic":
        ok = bool(setting("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN"))
        return {"backend": backend, "ready": ok, "label": f"Anthropic API · {setting('LLM_MODEL', 'claude-opus-5-5')}", "manual": False}
    if backend == "gemini":
        ok = bool(setting("GEMINI_API_KEY") or env("GOOGLE_API_KEY"))
        return {"backend": backend, "ready": ok, "label": f"Google Gemini · {setting('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)}", "manual": False}
    return {"backend": backend, "ready": bool(setting("OPENAI_COMPAT_BASE_URL")),
            "label": f"Yerel / OpenAI uyumlu · {setting('OPENAI_COMPAT_MODEL', 'local-model')}", "manual": False}


def page(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {
        "csrf": csrf_token(request.state.user), "current_job": jobs.current, "manual_pending": broker.pending,
        "llm": llm_status(), "path": request.url.path, **ctx})


def get_project(slug: str) -> Project:
    try:
        p = Project(slug)
    except ValueError:
        raise HTTPException(404, "Proje bulunamadı")
    if not p.exists():
        raise HTTPException(404, "Proje bulunamadı")
    return p


def project_ctx(p: Project, step: str) -> dict[str, Any]:
    onto = p.ontology()
    st = p.stats()
    return {"p": p, "meta": p.meta(), "onto": onto, "stats": st, "step": step,
            "steps_done": {"ontology": onto.is_ready(), "docs": st["n_done"] > 0, "graph": st["n_entities"] > 0}}


# ---------------------------------------------------------------- projeler


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return page(request, "home.html", projects=list_projects())


@app.post("/projects")
def new_project(request: Request, csrf: str = Form(...), name: str = Form(...), description: str = Form("")):
    check_csrf(request, csrf)
    if not name.strip():
        return RedirectResponse("/", status_code=303)
    p = create_project(name[:80], description[:500])
    return RedirectResponse(f"/p/{p.slug}/ontology", status_code=303)


@app.post("/p/{slug}/delete")
def delete_project(request: Request, slug: str, csrf: str = Form(...), confirm: str = Form("")):
    check_csrf(request, csrf)
    p = get_project(slug)
    if confirm.strip() != p.meta().get("name"):
        raise HTTPException(400, "Silmek için proje adını aynen yazın.")
    import shutil
    shutil.rmtree(p.dir)
    return RedirectResponse("/", status_code=303)


@app.get("/p/{slug}")
def project_home(slug: str):
    p = get_project(slug)
    ctx = project_ctx(p, "")
    if not ctx["steps_done"]["ontology"]:
        return RedirectResponse(f"/p/{slug}/ontology", status_code=303)
    if not ctx["steps_done"]["graph"]:
        return RedirectResponse(f"/p/{slug}/docs", status_code=303)
    return RedirectResponse(f"/p/{slug}/graph", status_code=303)


# ---------------------------------------------------------------- 1. ontoloji


@app.get("/p/{slug}/ontology", response_class=HTMLResponse)
def ontology_page(request: Request, slug: str):
    p = get_project(slug)
    return page(request, "ontology.html", **project_ctx(p, "ontology"), suggestion=p.suggestion(),
                templates_list=[(k, v[0]) for k, v in TEMPLATES.items()], docs=p.documents())


@app.post("/p/{slug}/ontology")
def ontology_save(request: Request, slug: str, csrf: str = Form(...), domain: str = Form(""),
                  ent_name: list[str] = Form([]), ent_desc: list[str] = Form([]), ent_uri: list[str] = Form([]),
                  rel_name: list[str] = Form([]), rel_desc: list[str] = Form([]), rel_source: list[str] = Form([]),
                  rel_target: list[str] = Form([]), rel_uri: list[str] = Form([])):
    check_csrf(request, csrf)
    p = get_project(slug)

    def at(lst, i):
        return lst[i] if i < len(lst) else ""

    def types(s: str) -> list[str]:
        return [t.strip() for t in s.split(",") if t.strip()]

    ents = [EntityType(name=n.strip()[:60], description=at(ent_desc, i).strip()[:300], uri=at(ent_uri, i) or None)
            for i, n in enumerate(ent_name) if n.strip()]
    rels = [RelationType(name=n.strip()[:60], description=at(rel_desc, i).strip()[:300], source=types(at(rel_source, i)),
                         target=types(at(rel_target, i)), uri=at(rel_uri, i) or None)
            for i, n in enumerate(rel_name) if n.strip()]
    p.save_ontology(Ontology(entity_types=ents, relation_types=rels, domain=domain[:200]))
    return RedirectResponse(f"/p/{slug}/ontology?saved=1", status_code=303)


@app.post("/p/{slug}/ontology/template")
def ontology_template(request: Request, slug: str, csrf: str = Form(...), key: str = Form(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    if key not in TEMPLATES:
        raise HTTPException(400)
    p.save_ontology(TEMPLATES[key][1])
    return RedirectResponse(f"/p/{slug}/ontology?saved=1", status_code=303)


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Dosya çok büyük (en fazla 80 MB)")
    return data


@app.post("/p/{slug}/ontology/suggest")
async def ontology_suggest(request: Request, slug: str, csrf: str = Form(...), doc_id: str = Form(""),
                           hint: str = Form(""), file: UploadFile | None = File(None)):
    check_csrf(request, csrf)
    p = get_project(slug)
    if file is not None and file.filename:
        data = await _read_upload(file)
        try:
            text = extract_text(file.filename, data)
        except UnsupportedFile as exc:
            raise HTTPException(400, str(exc))
        doc = p.add_document(file.filename, data, text)
        doc_id = doc["id"]
    if not doc_id or p.document(doc_id) is None:
        raise HTTPException(400, "Bir belge seçin veya yükleyin.")
    text = p.text(doc_id)

    def run():
        onto = extractor.suggest_ontology(get_llm(), text, hint[:500])
        p.save_suggestion(onto)
        return {"entity_types": len(onto.entity_types), "relation_types": len(onto.relation_types)}

    job = jobs.submit("suggest", "Belgeden ontoloji önerisi", {"slug": slug, "next": f"/p/{slug}/ontology"}, run)
    return RedirectResponse(f"/jobs/{job.id}?next=/p/{slug}/ontology", status_code=303)


@app.post("/p/{slug}/ontology/suggestion")
def ontology_suggestion(request: Request, slug: str, csrf: str = Form(...), action: str = Form(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    sug = p.suggestion()
    if sug and action == "apply":
        p.save_ontology(sug)
    p.save_suggestion(None)
    return RedirectResponse(f"/p/{slug}/ontology", status_code=303)


@app.post("/p/{slug}/ontology/import")
async def ontology_import(request: Request, slug: str, csrf: str = Form(...), file: UploadFile = File(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    data = await _read_upload(file)
    try:
        parsed = standards.parse_ontology_file(file.filename or "ontology.ttl", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    parsed["filename"] = file.filename
    (p.dir / "import_candidates.json").write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
    return RedirectResponse(f"/p/{slug}/ontology/import", status_code=303)


@app.get("/p/{slug}/ontology/import", response_class=HTMLResponse)
def ontology_import_page(request: Request, slug: str):
    p = get_project(slug)
    path = p.dir / "import_candidates.json"
    if not path.exists():
        return RedirectResponse(f"/p/{slug}/ontology", status_code=303)
    return page(request, "ontology_import.html", **project_ctx(p, "ontology"),
                parsed=json.loads(path.read_text(encoding="utf-8")))


@app.post("/p/{slug}/ontology/import/apply")
def ontology_import_apply(request: Request, slug: str, csrf: str = Form(...), cls: list[str] = Form([]),
                          prop: list[str] = Form([])):
    check_csrf(request, csrf)
    p = get_project(slug)
    path = p.dir / "import_candidates.json"
    if path.exists():
        parsed = json.loads(path.read_text(encoding="utf-8"))
        p.save_ontology(standards.merge_selection(p.ontology(), parsed, cls, prop))
        path.unlink()
    return RedirectResponse(f"/p/{slug}/ontology?saved=1", status_code=303)


# ---------------------------------------------------------------- 2. belgeler


@app.get("/p/{slug}/docs", response_class=HTMLResponse)
def docs_page(request: Request, slug: str):
    p = get_project(slug)
    return page(request, "docs.html", **project_ctx(p, "docs"), docs=p.documents(),
                supported=", ".join(sorted(SUPPORTED)))


@app.post("/p/{slug}/docs/upload")
async def docs_upload(request: Request, slug: str, csrf: str = Form(...), files: list[UploadFile] = File(...),
                      process: str = Form("")):
    check_csrf(request, csrf)
    p = get_project(slug)
    errors, added = [], []
    for f in files:
        if not f.filename:
            continue
        data = await _read_upload(f)
        try:
            text = extract_text(f.filename, data)
        except UnsupportedFile as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:
            errors.append(f"{f.filename}: okunamadı ({type(exc).__name__})")
            continue
        if not text.strip():
            errors.append(f"{f.filename}: metin bulunamadı (taranmış PDF olabilir)")
            continue
        added.append(p.add_document(f.filename, data, text)["id"])
    if added and process and p.ontology().is_ready():
        _submit_processing(p, added)
    q = ("?err=" + "|".join(errors)[:600]) if errors else ""
    return RedirectResponse(f"/p/{slug}/docs{q}", status_code=303)


def _submit_processing(p: Project, doc_ids: list[str]):
    for d in doc_ids:
        p.update_document(d, status="queued", error=None)

    def run():
        llm = get_llm()
        done = 0
        for batch in extractor.plan_batches(p, doc_ids, llm):
            try:
                if len(batch) == 1:
                    extractor.process_document(p, batch[0], llm)
                else:
                    extractor.process_batch(p, batch, llm)
                done += len(batch)
            except Exception as exc:
                log.error("Belge(ler) işlenemedi (%s): %s", batch, exc)
                if "iptal" in str(exc):
                    for rest in doc_ids:
                        if (p.document(rest) or {}).get("status") in ("queued", "processing"):
                            p.update_document(rest, status="new")
                    raise
        p.sync_neo4j()
        return {"processed": done, "total": len(doc_ids)}

    return jobs.submit("process", f"{len(doc_ids)} belge işleniyor", {"slug": p.slug, "next": f"/p/{p.slug}/docs"}, run)


@app.post("/p/{slug}/docs/process")
def docs_process(request: Request, slug: str, csrf: str = Form(...), doc_id: list[str] = Form([]),
                 which: str = Form("new")):
    check_csrf(request, csrf)
    p = get_project(slug)
    if not p.ontology().is_ready():
        return RedirectResponse(f"/p/{slug}/ontology", status_code=303)
    docs = p.documents()
    if doc_id:
        ids = [d["id"] for d in docs if d["id"] in doc_id]
    elif which == "all":
        ids = [d["id"] for d in docs]
    else:
        ids = [d["id"] for d in docs if d["status"] in ("new", "error")]
    if ids:
        _submit_processing(p, ids)
    return RedirectResponse(f"/p/{slug}/docs", status_code=303)


@app.post("/p/{slug}/docs/{doc_id}/delete")
def docs_delete(request: Request, slug: str, doc_id: str, csrf: str = Form(...)):
    check_csrf(request, csrf)
    get_project(slug).delete_document(doc_id)
    return RedirectResponse(f"/p/{slug}/docs", status_code=303)


@app.get("/p/{slug}/docs/{doc_id}", response_class=HTMLResponse)
def doc_page(request: Request, slug: str, doc_id: str):
    p = get_project(slug)
    doc = p.document(doc_id)
    if doc is None:
        raise HTTPException(404)
    g = p.graph()
    ids = {doc_id, doc_id + oai.META_SUFFIX, doc_id + ner.NER_SUFFIX}
    ents = [e for e in g["entities"].values() if any(m["doc_id"] in ids for m in e["mentions"])]
    rels = []
    for r in g["relations"].values():
        for s in r["sources"]:
            if s["doc_id"] in ids:
                rels.append({"s": g["entities"].get(r["subject"], {}).get("name", "?"), "p": r["predicate"],
                             "o": g["entities"].get(r["object"], {}).get("name", "?"), **s})
    rejected = [x for x in g["rejected"] if x.get("doc_id") == doc_id]
    return page(request, "doc.html", **project_ctx(p, "docs"), doc=doc, ents=sorted(ents, key=lambda e: (e["type"], e["name"])),
                rels=rels, rejected=rejected)


@app.get("/api/p/{slug}/status")
def status_api(slug: str):
    p = get_project(slug)
    return {"docs": [{k: d[k] for k in ("id", "status", "n_entities", "n_relations", "n_rejected", "error")}
                     for d in p.documents()],
            "busy": jobs.busy(), "manual_pending": bool(broker.pending)}


# ---------------------------------------------------------------- arşivden çekme (OAI-PMH)


@app.get("/api/oai/sets")
def oai_sets(base: str):
    """Arşivi tanır ve setleri (DSpace'te topluluk/koleksiyon) listeler."""
    try:
        client = oai.OaiClient(base)
        info = client.identify()
        sets = client.list_sets()
    except oai.OaiError as exc:
        return JSONResponse({"ok": False, "error": str(exc)})
    return {"ok": True, "name": info.get("name"), "earliest": info.get("earliest"), "sets": sets}


@app.post("/p/{slug}/oai/harvest")
def oai_harvest(request: Request, slug: str, csrf: str = Form(...), base: str = Form(...), set_spec: str = Form(""),
                date_from: str = Form(""), date_until: str = Form(""), limit: int = Form(50),
                meta_graph: str = Form(""), fulltext: str = Form(""), process: str = Form("")):
    check_csrf(request, csrf)
    p = get_project(slug)
    try:
        oai.check_base_url(base)
    except oai.OaiError as exc:
        return RedirectResponse(f"/p/{slug}/docs?err={exc}", status_code=303)
    limit = max(1, min(int(limit or 50), 5000))
    for v in (date_from, date_until):
        if v and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            return RedirectResponse(f"/p/{slug}/docs?err=Tarih YYYY-AA-GG biçiminde olmalı", status_code=303)

    def run():
        res = oai.harvest(p, base, set_spec.strip(), date_from, date_until, limit,
                          build_metadata_graph=bool(meta_graph), fulltext=bool(fulltext))
        new_ids = res["added"] + res["updated"]
        if process and new_ids and p.ontology().is_ready():
            llm = get_llm()
            for d in new_ids:
                p.update_document(d, status="queued")
            extractor.process_many(p, new_ids, llm)
            p.sync_neo4j()
        return {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}

    title = f"Arşivden çekiliyor ({urlparse(base).netloc})"
    job = jobs.submit("oai", title, {"slug": slug, "next": f"/p/{slug}/docs"}, run)
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


# ---------------------------------------------------------------- 3. graf


@app.get("/p/{slug}/graph", response_class=HTMLResponse)
def graph_page(request: Request, slug: str):
    p = get_project(slug)
    g = p.graph()
    n_linked = sum(1 for e in g["entities"].values() if e.get("wikidata"))
    n_wd_rel = sum(1 for r in g["relations"].values() if any(s["doc_id"] == wikidata.WIKIDATA_SOURCE for s in r["sources"]))
    return page(request, "graph.html", **project_ctx(p, "graph"), n_linked=n_linked, n_wd_rel=n_wd_rel)


def _source_name(docs: dict[str, str], doc_id: str) -> str:
    if doc_id == wikidata.WIKIDATA_SOURCE:
        return "Wikidata"
    if doc_id.endswith(oai.META_SUFFIX):
        return docs.get(doc_id[:-len(oai.META_SUFFIX)], doc_id) + " (arşiv metadata)"
    if doc_id.endswith(ner.NER_SUFFIX):
        return docs.get(doc_id[:-len(ner.NER_SUFFIX)], doc_id) + " (varlık modeli)"
    return docs.get(doc_id, doc_id)


def _origin(sources: list[dict[str, Any]]) -> str:
    if all(s["doc_id"] == wikidata.WIKIDATA_SOURCE for s in sources):
        return "wikidata"
    if all(s["doc_id"].endswith(oai.META_SUFFIX) for s in sources):
        return "metadata"
    return "document"


def _graph_payload(p: Project) -> dict[str, Any]:
    """Tam ayrıntılı graf (yol bulma ve dışa aktarım için)."""
    g = p.graph()
    docs = {d["id"]: d["filename"] for d in p.documents()}
    nodes = []
    for key, e in g["entities"].items():
        nodes.append(_node_detail(key, e, docs))
    edges = [_edge_detail(r, docs) for r in g["relations"].values()]
    return {"nodes": nodes, "edges": edges, "types": p.ontology().type_names()}


def _node_detail(key: str, e: dict[str, Any], docs: dict[str, str]) -> dict[str, Any]:
    return {"id": key, "label": e["name"], "type": e["type"], "description": e.get("description"),
            "wikidata": e.get("wikidata"), "alternatives": e.get("wikidata_alternatives") or [],
            "n_mentions": len(e["mentions"]),
            "mentions": [{"doc": _source_name(docs, m["doc_id"]), "text": m["text"]} for m in e["mentions"]][:20]}


def _edge_detail(r: dict[str, Any], docs: dict[str, str]) -> dict[str, Any]:
    return {"id": r["id"], "from": r["subject"], "to": r["object"], "label": r["predicate"], "origin": _origin(r["sources"]),
            "n_sources": len(r["sources"]),
            "sources": [{"doc": _source_name(docs, s["doc_id"]), "sentence": s["sentence"],
                         "confidence": s.get("confidence")} for s in r["sources"]][:10]}


# Graf haritası için hafif veri: yalnızca çizim için gerekenler (ayrıntılar tıklanınca ayrıca çekilir),
# grafın sürümüne göre bellekte tutulur ve ETag ile tarayıcı önbelleğinden döner.
_light_cache: dict[str, tuple[str, bytes]] = {}


def _graph_version(p: Project) -> str:
    parts = []
    for name in ("graph.json", "documents.json", "ontology.json", "layout.json"):
        try:
            st = (p.dir / name).stat()
            parts.append(f"{st.st_mtime_ns:x}-{st.st_size:x}")
        except OSError:
            parts.append("0")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _graph_light(p: Project) -> tuple[str, bytes]:
    ver = _graph_version(p)
    hit = _light_cache.get(p.slug)
    if hit and hit[0] == ver:
        return hit
    g = p.graph()
    layout = (_read_layout(p).get("pos") or {})
    nodes = []
    for key, e in g["entities"].items():
        n: dict[str, Any] = {"id": key, "label": e["name"], "type": e["type"]}
        pos = layout.get(key)
        if pos:
            n["x"], n["y"] = pos
        nodes.append(n)
    edges = [{"id": r["id"], "from": r["subject"], "to": r["object"], "label": r["predicate"], "origin": _origin(r["sources"])}
             for r in g["relations"].values()]
    body = json.dumps({"version": ver, "nodes": nodes, "edges": edges, "types": p.ontology().type_names()},
                      ensure_ascii=False, separators=(",", ":")).encode()
    _light_cache[p.slug] = (ver, body)
    return ver, body


@app.get("/api/p/{slug}/graph")
def graph_api(request: Request, slug: str, full: bool = False):
    p = get_project(slug)
    if full:
        return JSONResponse(_graph_payload(p))
    ver, body = _graph_light(p)
    etag = f'"{ver}"'
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(body, media_type="application/json", headers=headers)


@app.get("/api/p/{slug}/node")
def node_api(slug: str, id: str):
    p = get_project(slug)
    e = p.graph()["entities"].get(id)
    if not e:
        raise HTTPException(404, "Varlık bulunamadı")
    return _node_detail(id, e, {d["id"]: d["filename"] for d in p.documents()})


@app.get("/api/p/{slug}/edge")
def edge_api(slug: str, id: str):
    p = get_project(slug)
    r = next((r for r in p.graph()["relations"].values() if r["id"] == id), None)
    if not r:
        raise HTTPException(404, "İlişki bulunamadı")
    return _edge_detail(r, {d["id"]: d["filename"] for d in p.documents()})


MAX_LAYOUT_NODES = 50000


def _read_layout(p: Project) -> dict[str, Any]:
    try:
        return json.loads((p.dir / "layout.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.post("/api/p/{slug}/layout")
async def layout_save(request: Request, slug: str):
    """Tarayıcının hesapladığı düğüm konumlarını saklar: sonraki açılışlarda yerleşim yeniden hesaplanmaz."""
    p = get_project(slug)
    try:
        data = await request.json()
    except ValueError:
        raise HTTPException(400, "Geçersiz veri")
    check_csrf(request, str(data.get("csrf", "")))
    keys = p.graph()["entities"].keys()
    pos = {}
    for k, v in (data.get("pos") or {}).items():
        if k in keys and isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            pos[k] = [round(float(v[0]), 1), round(float(v[1]), 1)]
    if len(pos) > MAX_LAYOUT_NODES:
        raise HTTPException(413, "Çok fazla düğüm")
    merged = {**(_read_layout(p).get("pos") or {}), **pos} if data.get("merge") else pos
    merged = {k: v for k, v in merged.items() if k in keys}
    tmp = p.dir / ".layout.json.tmp"
    tmp.write_text(json.dumps({"pos": merged}, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p.dir / "layout.json")
    return {"ok": True, "n": len(merged), "version": _graph_version(p)}


# ---------------------------------------------------------------- arama


def _search(p: Project, q: str, entity: str, origin: str, limit: int) -> dict[str, Any]:
    g = p.graph()
    if entity and entity not in g["entities"]:
        entity = ""
    res = gsearch.search(g, q, entity=entity or None, origin=origin if origin in ("document", "metadata", "wikidata") else "",
                         wikidata_source=wikidata.WIKIDATA_SOURCE, meta_suffix=oai.META_SUFFIX, limit=limit)
    docs = {d["id"]: d["filename"] for d in p.documents()}
    ents = g["entities"]
    rows = []
    for r in res["relations"]:
        s, o = ents.get(r["subject"], {}), ents.get(r["object"], {})
        rows.append({"id": r["id"], "predicate": r["predicate"], "origin": r["origin"], "match": r["match"],
                     "subject": {"key": r["subject"], "name": s.get("name", r["subject"]), "type": s.get("type", "")},
                     "object": {"key": r["object"], "name": o.get("name", r["object"]), "type": o.get("type", "")},
                     "sources": [{"doc": _source_name(docs, x["doc_id"]),
                                  "doc_id": x["doc_id"].removesuffix(oai.META_SUFFIX).removesuffix(ner.NER_SUFFIX) if x["doc_id"] != wikidata.WIKIDATA_SOURCE else "",
                                  "sentence": x.get("sentence", ""), "confidence": x.get("confidence")} for x in r["sources"]]})
    entities = [{"key": e["key"], "name": e["name"], "type": e["type"], "description": e.get("description"),
                 "degree": e["degree"]} for e in res["entities"]]
    return {"query": q, "tokens": res["tokens"], "partial": res["partial"], "total": res["total"],
            "n_phrase": res["n_phrase"],
            "entity": ({"key": entity, "name": ents[entity]["name"], "type": ents[entity]["type"]} if entity else None),
            "relations": rows, "entities": entities}


def _highlight(text: str, toks: list[str]) -> Markup:
    out, last = [], 0
    for a, b in gsearch.spans(text, toks):
        out.append(escape(text[last:a]))
        out.append(Markup("<mark>") + escape(text[a:b]) + Markup("</mark>"))
        last = b
    out.append(escape(text[last:]))
    return Markup("").join(out)


@app.get("/p/{slug}/search", response_class=HTMLResponse)
def search_page(request: Request, slug: str, q: str = "", entity: str = "", origin: str = "", limit: int = 100):
    p = get_project(slug)
    limit = max(10, min(limit, 2000))
    res = _search(p, q, entity, origin, limit) if (q.strip() or entity) else None
    examples: dict[str, list[str]] = {"entities": [], "predicates": []}
    if res is None:
        g = p.graph()
        deg: dict[str, int] = {}
        preds: dict[str, int] = {}
        for r in g["relations"].values():
            for k in (r["subject"], r["object"]):
                deg[k] = deg.get(k, 0) + 1
            if not r["predicate"].startswith("WD:"):
                preds[r["predicate"]] = preds.get(r["predicate"], 0) + 1
        top = sorted(deg, key=lambda k: -deg[k])[:8]
        examples = {"entities": [g["entities"][k]["name"] for k in top if k in g["entities"]],
                    "predicates": [x.replace("_", " ").replace("İ", "i").lower() for x in sorted(preds, key=lambda k: -preds[k])[:6]]}
    return page(request, "search.html", **project_ctx(p, "search"), res=res, q=q, origin=origin, limit=limit,
                examples=examples, hl=lambda t: _highlight(t, res["tokens"] if res else []))


@app.get("/api/p/{slug}/search")
def search_api(slug: str, q: str = "", entity: str = "", origin: str = "", limit: int = 100):
    return JSONResponse(_search(get_project(slug), q, entity, origin, max(1, min(limit, 2000))))


@app.get("/p/{slug}/search.csv")
def search_csv(slug: str, q: str = "", entity: str = "", origin: str = ""):
    res = _search(get_project(slug), q, entity, origin, 100000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ozne", "ozne_turu", "iliski", "nesne", "nesne_turu", "kaynak", "kaynak_cumle", "guven"])
    for r in res["relations"]:
        for x in r["sources"] or [{}]:
            w.writerow([r["subject"]["name"], r["subject"]["type"], r["predicate"], r["object"]["name"], r["object"]["type"],
                        x.get("doc", ""), x.get("sentence", ""), x.get("confidence", "")])
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{slug}-arama.csv"'})


@app.get("/api/p/{slug}/path")
def path_api(slug: str, a: str, b: str, include_wikidata: bool = True):
    """İki varlık arasındaki en kısa bağlantı (yön gözetmeden) — "bunlar nasıl bağlantılı?" sorusu için."""
    data = _graph_payload(get_project(slug))
    adj: dict[str, list[tuple[str, dict]]] = {}
    for e in data["edges"]:
        if e["origin"] == "wikidata" and not include_wikidata:
            continue
        adj.setdefault(e["from"], []).append((e["to"], e))
        adj.setdefault(e["to"], []).append((e["from"], e))
    prev: dict[str, tuple[str, dict] | None] = {a: None}
    q = deque([a])
    while q:
        n = q.popleft()
        if n == b:
            break
        for nb, e in adj.get(n, []):
            if nb not in prev:
                prev[nb] = (n, e)
                q.append(nb)
    if b not in prev:
        return {"found": False, "edges": []}
    path, cur = [], b
    while prev[cur] is not None:
        n, e = prev[cur]
        path.append(e)
        cur = n
    return {"found": True, "edges": list(reversed(path))}


@app.post("/p/{slug}/wikidata/{action}")
def wikidata_action(request: Request, slug: str, action: str, csrf: str = Form(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    if action == "link":
        job = jobs.submit("wikidata", "Varlıkları Wikidata ile eşle", {"slug": slug, "next": f"/p/{slug}/graph"},
                          lambda: (wikidata.link_entities(p), p.sync_neo4j())[0])
    elif action == "enrich":
        job = jobs.submit("wikidata", "Wikidata'dan ilişki ekle", {"slug": slug, "next": f"/p/{slug}/graph"},
                          lambda: (wikidata.enrich(p), p.sync_neo4j())[0])
    elif action == "clear":
        p.remove_contributions(wikidata.WIKIDATA_SOURCE)
        p.sync_neo4j()
        return RedirectResponse(f"/p/{slug}/graph", status_code=303)
    else:
        raise HTTPException(404)
    return RedirectResponse(f"/jobs/{job.id}?next=/p/{slug}/graph", status_code=303)


@app.post("/p/{slug}/entity/link")
def entity_link(request: Request, slug: str, csrf: str = Form(...), key: str = Form(...), qid: str = Form("")):
    check_csrf(request, csrf)
    p = get_project(slug)
    wikidata.set_link(p, key, qid or None)
    return JSONResponse({"ok": True})


@app.get("/p/{slug}/export/{fmt}")
def export(slug: str, fmt: str):
    p = get_project(slug)
    g = p.graph()
    name = p.slug
    if fmt == "json":
        return Response(json.dumps({"ontology": p.ontology().model_dump(), **g}, ensure_ascii=False, indent=1),
                        media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
    if fmt == "entities.csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "ad", "tür", "açıklama", "wikidata", "belge_sayısı"])
        for key, e in g["entities"].items():
            w.writerow([e["id"], e["name"], e["type"], e.get("description") or "", (e.get("wikidata") or {}).get("qid", ""),
                        len({m["doc_id"] for m in e["mentions"]})])
        return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}_varliklar.csv"'})
    if fmt == "relations.csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["özne", "ilişki", "nesne", "kaynak", "kaynak_cümle"])
        docs = {d["id"]: d["filename"] for d in p.documents()}
        for r in g["relations"].values():
            for s in r["sources"]:
                w.writerow([g["entities"].get(r["subject"], {}).get("name"), r["predicate"],
                            g["entities"].get(r["object"], {}).get("name"), docs.get(s["doc_id"], s["doc_id"]), s["sentence"]])
        return Response("﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}_iliskiler.csv"'})
    if fmt == "ttl":
        return Response(_turtle(p), media_type="text/turtle; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}.ttl"'})
    raise HTTPException(404)


def _turtle(p: Project) -> str:
    """RDF (Turtle) dışa aktarım: standart URI'si olan türler/ilişkiler o URI'yle, Wikidata eşlemeleri owl:sameAs ile."""
    from rdflib import OWL, RDF, RDFS, Graph, Literal, Namespace, URIRef

    from ..okg.ontology import ascii_ident

    base = Namespace(f"urn:kg:{p.slug}:")
    onto = p.ontology()
    t_uri = {t.name: URIRef(t.uri) if t.uri else base[f"type/{ascii_ident(t.name)}"] for t in onto.entity_types}
    r_uri = {r.name: URIRef(r.uri) if r.uri else base[f"rel/{ascii_ident(r.name)}"] for r in onto.relation_types}
    g = Graph()
    g.bind("kg", base)
    data = p.graph()
    ids = {}
    for key, e in data["entities"].items():
        node = base[f"entity/{e['id']}"]
        ids[key] = node
        g.add((node, RDF.type, t_uri.get(e["type"], base[f"type/{ascii_ident(e['type'])}"])))
        g.add((node, RDFS.label, Literal(e["name"])))
        if e.get("description"):
            g.add((node, RDFS.comment, Literal(e["description"], lang="tr")))
        if e.get("wikidata"):
            g.add((node, OWL.sameAs, URIRef(f"http://www.wikidata.org/entity/{e['wikidata']['qid']}")))
    for r in data["relations"].values():
        if r["subject"] in ids and r["object"] in ids:
            pred = r_uri.get(r["predicate"], base[f"rel/{ascii_ident(r['predicate'])}"])
            g.add((ids[r["subject"]], pred, ids[r["object"]]))
    return g.serialize(format="turtle")


# ---------------------------------------------------------------- işler ve elle yanıt


# ---------------------------------------------------------------- hızlı varlık modeli (NER)


def _ner_job(p: Project, kind: str, title: str, fn) -> RedirectResponse:
    nxt = f"/p/{p.slug}/ner"
    job = jobs.submit(kind, title, {"slug": p.slug, "next": nxt}, fn)
    return RedirectResponse(f"/jobs/{job.id}?next={nxt}", status_code=303)


@app.get("/p/{slug}/ner", response_class=HTMLResponse)
def ner_page(request: Request, slug: str):
    p = get_project(slug)
    cfg = ner.load_config(p)
    names = {d["id"]: d["filename"] for d in p.documents()}
    ds_rows = [{"doc_id": r["doc_id"], "name": names.get(r["doc_id"], r["doc_id"]), "n": len(r["entities"]),
                "rejected": len(r.get("rejected", [])), "source": r.get("source", "")} for r in ner.load_dataset(p)]
    return page(request, "ner.html", **project_ctx(p, "ner"), cfg=cfg, st=ner.status(p), presets=ner.DOMAIN_PRESETS,
                ns=ner.get_settings(cfg), ds_rows=ds_rows, mem=ner.memory_info(),
                gmap=ner.gliner_label_map(cfg) if cfg["labels"] else {},
                matched=ner.match_preset(cfg.get("domain") or "") if cfg.get("domain") else None)


@app.post("/p/{slug}/ner/labels")
def ner_labels(request: Request, slug: str, csrf: str = Form(...), name: list[str] = Form([]), desc: list[str] = Form([]),
               prompt: list[str] = Form([]), use_desc: str = Form(""), threshold: float = Form(0.5), base_model: str = Form("")):
    check_csrf(request, csrf)
    p = get_project(slug)
    cfg = ner.load_config(p)
    at = lambda lst, i: lst[i] if i < len(lst) else ""
    cfg["labels"] = [{"name": n, "description": at(desc, i), "prompt": at(prompt, i)} for i, n in enumerate(name)]
    cfg["gliner_use_description"] = bool(use_desc)
    cfg["threshold"] = min(0.95, max(0.05, threshold))
    if base_model.strip():
        cfg["base_model"] = base_model.strip()
    ner.save_config(p, cfg)
    return RedirectResponse(f"/p/{slug}/ner#labels", status_code=303)


@app.post("/p/{slug}/ner/labels/import")
async def ner_labels_import(request: Request, slug: str, csrf: str = Form(...), text: str = Form(""),
                            file: UploadFile | None = File(None), with_relations: str = Form(""), mode: str = Form("merge")):
    """Tür listesini JSON'dan yükler (ontoloji biçimi: domain, entity_types, relation_types)."""
    check_csrf(request, csrf)
    p = get_project(slug)
    raw: str | bytes = text
    if file is not None and file.filename:
        raw = await _read_upload(file)
    if not (raw.strip() if isinstance(raw, str) else raw):
        return RedirectResponse(f"/p/{slug}/ner?err=JSON dosyası seç veya metni yapıştır#labels", status_code=303)
    try:
        parsed = ner.parse_labels_json(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        return RedirectResponse(f"/p/{slug}/ner?err=JSON okunamadı: {str(exc)[:200]}", status_code=303)
    cfg = ner.load_config(p)
    cfg["labels"] = ner.merge_labels(cfg["labels"], parsed["labels"], mode)
    if parsed["domain"] and (mode == "replace" or not cfg.get("domain")):
        cfg["domain"] = parsed["domain"][:200]
    ner.save_config(p, cfg)
    if with_relations and (parsed["relation_types"] or any(l.get("uri") for l in parsed["labels"])):
        ner.merge_relation_types(p, parsed["relation_types"], parsed["labels"])
    return RedirectResponse(f"/p/{slug}/ner#labels", status_code=303)


@app.get("/p/{slug}/ner/labels.json")
def ner_labels_export(slug: str):
    p = get_project(slug)
    body = json.dumps(ner.labels_json(p, ner.load_config(p)), ensure_ascii=False, indent=2)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{slug}-varlik-turleri.json"'})


@app.post("/p/{slug}/ner/suggest")
def ner_suggest(request: Request, slug: str, csrf: str = Form(...), domain: str = Form(""), how: str = Form("preset"),
                preset: str = Form(""), mode: str = Form("merge")):
    check_csrf(request, csrf)
    p = get_project(slug)
    cfg = ner.load_config(p)
    cfg["domain"] = domain.strip()[:200]
    key = preset if preset in ner.DOMAIN_PRESETS else ner.match_preset(domain)
    if how == "preset":
        if not key:
            ner.save_config(p, cfg)
            return RedirectResponse(f"/p/{slug}/ner?err=" + "Bu alan için hazır liste yok; 'Dil modeline sor'u dene veya listeden seç.", status_code=303)
        cfg["labels"] = ner.merge_labels(cfg["labels"], [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS[key]["labels"]], mode)
        ner.save_config(p, cfg)
        return RedirectResponse(f"/p/{slug}/ner#labels", status_code=303)
    if not domain.strip():
        raise HTTPException(400, "Alan adını yaz (ör. elektrik mühendisliği)")
    ner.save_config(p, cfg)
    sample = "\n\n".join(p.text(d["id"])[:1500] for d in p.documents()[:5])

    def run():
        labels = ner.suggest_labels(get_llm(), domain.strip(), sample)
        c = ner.load_config(p)
        c["labels"] = ner.merge_labels(c["labels"], [{"name": l.name, "description": l.description or ""} for l in labels], mode)
        ner.save_config(p, c)
        return {"labels": len(labels)}

    return _ner_job(p, "ner-suggest", "Varlık türü önerisi", run)


async def _ner_settings_form(request: Request, p: Project, step: str) -> tuple[str, dict[str, Any]]:
    """Adım formunu okur, ayarları kaydeder; (eylem, birleşik ayarlar) döndürür. İşaret kutuları için gizli alan
    "_bools" formdaki tüm kutuların adlarını taşır (işaretsiz kutular forma hiç gelmez)."""
    form = await request.form()
    check_csrf(request, str(form.get("csrf", "")))
    values: dict[str, Any] = {k: v for k, v in form.items() if k not in ("csrf", "action", "_bools")}
    for k in str(form.get("_bools", "")).split(","):
        if k:
            values[k] = k in form
    merged = ner.update_settings(p, step, values)
    if "threshold" in form:
        cfg = ner.load_config(p)
        try:
            cfg["threshold"] = min(0.95, max(0.05, float(form["threshold"])))
            ner.save_config(p, cfg)
        except ValueError:
            pass
    return str(form.get("action", "run")), merged


def _uses_gliner(p: Project, step: str) -> bool:
    if not ner.gliner_available():
        return False
    if step == "train":
        kind = ner.get_settings(ner.load_config(p))["train"]["kind"]
        return kind in ("gliner", "auto")
    return ner.status(p).metrics.get("model_kind") == "gliner"


def _ner_heavy(p: Project, step: str) -> Any:
    """GLiNER kullanan adımları ayrı süreçte çalıştırır (bellek taşarsa web sunucusu düşmesin); sözlük modeli süreç içinde."""
    if not _uses_gliner(p, step):
        return {"train": ner.train, "apply": ner.apply, "evaluate": ner.evaluate_current}[step](p)
    wlog = logging.getLogger("kg.ner.worker")
    noisy = ("HTTP Request", "Warning: You are sending", "warnings.warn", "FutureWarning", "UserWarning")
    res = ner.run_in_subprocess(p, step, on_line=lambda l: None if any(n in l for n in noisy) else wlog.info(l))
    if step == "apply":
        p.sync_neo4j()
    return res


@app.post("/p/{slug}/ner/annotate")
async def ner_annotate(request: Request, slug: str):
    p = get_project(slug)
    action, s = await _ner_settings_form(request, p, "annotate")
    if action == "save":
        return RedirectResponse(f"/p/{slug}/ner?ok=Ayarlar kaydedildi#annotate", status_code=303)
    if not ner.load_config(p)["labels"]:
        raise HTTPException(400, "Önce varlık türlerini belirle")

    def run():
        llm = get_llm()
        return ner.annotate(p, llm, use_batch_api=bool(s["batch_api"]) and hasattr(llm, "client"))

    what = f"toplam {s['n']} belgeye tamamlanıyor" if s["count_mode"] == "total" else f"{s['n']} yeni belge"
    return _ner_job(p, "ner-annotate", f"Eğitim verisi: {what} dil modeliyle etiketleniyor", run)


@app.post("/p/{slug}/ner/train")
async def ner_train(request: Request, slug: str):
    p = get_project(slug)
    action, _ = await _ner_settings_form(request, p, "train")
    if action == "save":
        return RedirectResponse(f"/p/{slug}/ner?ok=Ayarlar kaydedildi#model", status_code=303)
    return _ner_job(p, "ner-train", "Varlık modeli eğitiliyor", lambda: _ner_heavy(p, "train"))


@app.post("/p/{slug}/ner/apply")
async def ner_apply(request: Request, slug: str):
    p = get_project(slug)
    action, _ = await _ner_settings_form(request, p, "apply")
    if action == "save":
        return RedirectResponse(f"/p/{slug}/ner?ok=Ayarlar kaydedildi#apply", status_code=303)
    return _ner_job(p, "ner-apply", "Varlık modeli belgelere uygulanıyor", lambda: _ner_heavy(p, "apply"))


@app.post("/p/{slug}/ner/evaluate")
def ner_evaluate(request: Request, slug: str, csrf: str = Form(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    return _ner_job(p, "ner-evaluate", "Model test belgelerinde değerlendiriliyor", lambda: _ner_heavy(p, "evaluate"))


@app.post("/p/{slug}/ner/threshold")
def ner_threshold(request: Request, slug: str, csrf: str = Form(...), threshold: float = Form(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    cfg = ner.load_config(p)
    cfg["threshold"] = min(0.95, max(0.05, threshold))
    ner.save_config(p, cfg)
    return RedirectResponse(f"/p/{slug}/ner?ok=Eşik {cfg['threshold']} olarak kaydedildi#model", status_code=303)


@app.get("/api/p/{slug}/ner/preview")
def ner_preview(slug: str, step: str = "annotate", selection: str = "", source: str = "", keyword: str = "",
                min_chars: int = -1, n: int = -1, limit: int = -1, scope: str = "", count_mode: str = "", skip_done: str = "",
                per_request: int = -1, doc_chars: int = -1, max_chars: int = -1):
    """Formdaki seçimle kaç belgenin işleneceğini (kaydetmeden) hesaplar."""
    p = get_project(slug)
    if step not in ("annotate", "apply"):
        raise HTTPException(400)
    s = ner.get_settings(ner.load_config(p))[step]
    s.update(ner.clean_settings(step, {k: v for k, v in {"selection": selection, "source": source, "keyword": keyword,
                                                         "min_chars": min_chars, "n": n, "limit": limit, "scope": scope,
                                                         "count_mode": count_mode, "skip_done": skip_done or None,
                                                         "per_request": per_request, "doc_chars": doc_chars,
                                                         "max_chars": max_chars}.items()
                                       if v not in ("", -1, None)}))
    labeled = {r["doc_id"] for r in ner.load_dataset(p)}
    if step == "annotate":
        want = max(0, s["n"] - len(labeled)) if s["count_mode"] == "total" else s["n"]
        ids = ner.select_docs(p, exclude=labeled, selection=s["selection"], source=s["source"], keyword=s["keyword"],
                              min_chars=s["min_chars"], seed=s["seed"])
        est = ner.estimate_requests(p, ids[:want], s, manual=llm_status()["manual"])
        return {"available": len(ids), "will_process": min(want, len(ids)), "in_dataset": len(labeled), **est}
    exclude = set() if s["scope"] == "all" else labeled
    only = (lambda d: not d.get("ner")) if s["scope"] == "new" or s["skip_done"] else None
    ids = ner.select_docs(p, exclude=exclude, selection=s["selection"], source=s["source"], keyword=s["keyword"],
                          min_chars=s["min_chars"], only=only)
    return {"available": len(ids), "will_process": min(s["limit"], len(ids)) if s["limit"] else len(ids)}


@app.post("/p/{slug}/ner/dataset/remove")
def ner_dataset_remove(request: Request, slug: str, csrf: str = Form(...), doc_id: list[str] = Form([]),
                       keep: str = Form("")):
    check_csrf(request, csrf)
    p = get_project(slug)
    if keep.strip():
        try:
            n = ner.remove_from_dataset(p, keep=max(0, int(keep)))
        except ValueError:
            raise HTTPException(400, "Geçersiz sayı")
    else:
        n = ner.remove_from_dataset(p, doc_ids=doc_id)
    return RedirectResponse(f"/p/{slug}/ner?ok={n} belge veri setinden çıkarıldı#annotate", status_code=303)


@app.post("/p/{slug}/ner/clear")
def ner_clear(request: Request, slug: str, csrf: str = Form(...), what: str = Form(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    d = ner.ner_dir(p)
    if what == "labels":
        cfg = ner.load_config(p)
        cfg["labels"] = []
        ner.save_config(p, cfg)
        return RedirectResponse(f"/p/{slug}/ner#labels", status_code=303)
    if what == "dataset":
        (d / "dataset.jsonl").unlink(missing_ok=True)
        (d / "state.json").unlink(missing_ok=True)
    elif what == "graph":
        ner.clear_ner(p)
        p.sync_neo4j()
    elif what == "model":
        import shutil
        shutil.rmtree(d / "model", ignore_errors=True)
        for f in ("gazetteer.json", "metrics.json"):
            (d / f).unlink(missing_ok=True)
    return RedirectResponse(f"/p/{slug}/ner", status_code=303)


@app.post("/p/{slug}/ner/model-upload")
async def ner_model_upload(request: Request, slug: str, csrf: str = Form(...), file: UploadFile = File(...)):
    check_csrf(request, csrf)
    p = get_project(slug)
    data = await file.read()
    try:
        ner.install_model_zip(p, data)
    except (ValueError, OSError) as exc:
        return RedirectResponse(f"/p/{slug}/ner?err={exc}", status_code=303)
    except Exception:
        return RedirectResponse(f"/p/{slug}/ner?err=Zip açılamadı", status_code=303)
    return RedirectResponse(f"/p/{slug}/ner#model", status_code=303)


@app.post("/api/p/{slug}/ner/try")
async def ner_try(request: Request, slug: str):
    p = get_project(slug)
    data = await request.json()
    check_csrf(request, str(data.get("csrf", "")))
    text = str(data.get("text", ""))[:20000]
    import asyncio
    if ner.status(p).metrics.get("model_kind") == "gliner" and ner.gliner_available():
        try:  # GLiNER ayrı süreçte: web sunucusu modeli belleğe almaz
            return await asyncio.to_thread(ner.run_in_subprocess, p, "try", stdin=text, timeout=300)
        except Exception as exc:
            raise HTTPException(500, str(exc)[:500])
    model = ner.trained_model(p)
    if model is None:
        raise HTTPException(400, "Henüz eğitilmiş model yok")
    cfg = ner.load_config(p)
    preds = await asyncio.to_thread(model.predict, [text], ner.label_names(cfg), float(cfg.get("threshold", 0.5)))
    return {"entities": preds[0]}


def _gliner_export(p: Project, rows: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = ner.load_config(p)
    train_rows, test_rows = ner.split_dataset(rows, ner.get_settings(cfg)["train"]["test_ratio"])
    lm = ner.gliner_label_map(cfg)

    def conv(rs):
        return [{**ex, "ner": [[a, b, lm.get(lab, lab)] for a, b, lab in ex["ner"]]}
                for r in rs for ex in ner.to_gliner(r["text"], r["entities"])]

    return {"labels": [lm[n] for n in ner.label_names(cfg)], "label_map": lm,
            "train": conv(train_rows), "test": conv(test_rows)}


def _colab_notebook(p: Project, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Veri seti gömülü, tek tıkla çalışan Colab not defteri. Veri base64 olarak gömülür: kopyala-yapıştır veya
    karakter kodlaması sorunları (Türkçe harfler, özel işaretler) olmaz; dosya yükleme adımı da gerekmez."""
    import base64

    cfg = ner.load_config(p)
    st = ner.get_settings(cfg)["train"]
    data_b64 = base64.b64encode(json.dumps(_gliner_export(p, rows), ensure_ascii=False).encode("utf-8")).decode("ascii")
    script = (HERE.parent.parent / "scripts" / "train_gliner.py").read_text(encoding="utf-8")
    base = cfg.get("base_model") or ner.DEFAULT_BASE_MODEL

    def md(t):
        return {"cell_type": "markdown", "metadata": {}, "source": t}

    def code(t):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}

    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "cells": [
                md(f"# GLiNER egitimi: {p.meta().get('name', p.slug)}\n\n"
                   "1. Menu: Calisma zamani -> Calisma zamani turunu degistir -> T4 GPU.\n"
                   "2. Calisma zamani -> Tumunu calistir (veya hucreleri sirayla calistir).\n"
                   "3. Son hucre model.zip dosyasini indirir; web arayuzunde 3. adimdaki "
                   "'Baska makinede (GPU) egittigin modeli yukle' ile yukle.\n\n"
                   f"Veri seti bu not defterine gomuludur ({len(rows)} belge); dosya yuklemene gerek yok."),
                code("!pip -q install gliner==0.2.29 accelerate sentencepiece protobuf"),
                code("%%writefile train_gliner.py\n" + script),
                code("import base64\n"
                     f"DATA_B64 = '{data_b64}'\n"
                     "open('data.json', 'wb').write(base64.b64decode(DATA_B64))\n"
                     "print('data.json hazir')"),
                code(f"!python train_gliner.py data.json --base {base} --epochs {st['epochs']} --batch {st['batch_size']} "
                     f"--lr {st['lr']} --others-lr {st['others_lr']} --out model"),
                code("!cd model && zip -qr ../model.zip .\nfrom google.colab import files\nfiles.download('model.zip')"),
            ]}


@app.get("/p/{slug}/ner/export/{fmt}")
def ner_export(slug: str, fmt: str):
    p = get_project(slug)
    rows = ner.load_dataset(p)
    if fmt == "dataset.jsonl":
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        return Response(body, media_type="application/x-ndjson",
                        headers={"Content-Disposition": f'attachment; filename="{slug}-ner-dataset.jsonl"'})
    if fmt == "train_gliner.py":  # betiği kopyala-yapıştırla değil dosya olarak indir (boşluklar '·' olarak kopyalanmasın)
        return Response((HERE.parent.parent / "scripts" / "train_gliner.py").read_bytes(), media_type="text/x-python",
                        headers={"Content-Disposition": 'attachment; filename="train_gliner.py"'})
    if fmt == "colab.ipynb":
        return Response(json.dumps(_colab_notebook(p, rows), ensure_ascii=True), media_type="application/x-ipynb+json",
                        headers={"Content-Disposition": f'attachment; filename="{slug}-gliner-colab.ipynb"'})
    if fmt == "gliner.json":
        return Response(json.dumps(_gliner_export(p, rows), ensure_ascii=False), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{slug}-gliner-train.json"'})
    raise HTTPException(404)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str, next: str = ""):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "İş bulunamadı")
    next = next or (job.params or {}).get("next", "")
    safe_next = next if next.startswith("/") and not next.startswith("//") else ""
    return page(request, "job.html", job=job, next=safe_next)


@app.get("/api/jobs/{job_id}")
def job_api(job_id: str, since: int = 0):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    d = job.public(with_log=False)
    d["log"] = job.log[since:]
    d["log_len"] = len(job.log)
    d["manual_pending"] = bool(broker.pending and broker.pending.job_id == job_id)
    return JSONResponse(d)


# ---------------------------------------------------------------- ayarlar (dil modeli)


def _mask(v: str | None) -> str:
    if not v:
        return ""
    return ("•" * 6) + v[-4:] if len(v) > 8 else "•" * len(v)


def _settings_view() -> dict[str, Any]:
    rt = runtime_settings()
    rows = {}
    for k in RUNTIME_KEYS:
        val, from_ui = setting(k), k in rt
        rows[k] = {"value": "" if k in SECRET_KEYS else (val or ""), "masked": _mask(val) if k in SECRET_KEYS else "",
                   "set": bool(val), "source": "arayüz" if from_ui else ("ortam" if env(k) else "")}
    return rows


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return page(request, "settings.html", s=_settings_view(), gemini_default=DEFAULT_GEMINI_MODEL)


@app.post("/settings")
async def settings_save(request: Request):
    form = await request.form()
    check_csrf(request, str(form.get("csrf", "")))
    updates: dict[str, str | None] = {}
    backend = str(form.get("LLM_BACKEND", "")).strip()
    if backend in ("manual", "anthropic", "gemini", "openai_compat"):
        updates["LLM_BACKEND"] = backend
    for k in RUNTIME_KEYS:
        if k == "LLM_BACKEND" or k not in form:
            continue
        v = str(form.get(k, "")).strip()
        if k in SECRET_KEYS:
            if form.get(f"clear_{k}"):
                updates[k] = None
            elif v:  # boş bırakılan anahtar alanı mevcut anahtarı korur
                updates[k] = v
        else:
            updates[k] = v or None
    save_runtime_settings(updates)
    return RedirectResponse("/settings?ok=1", status_code=303)


class _Ping(BaseModel):
    ok: bool
    model: str


@app.post("/api/settings/test")
async def settings_test(request: Request):
    data = await request.json()
    check_csrf(request, str(data.get("csrf", "")))
    import asyncio

    def run():
        llm = get_llm()
        if getattr(llm, "manual", False):
            return {"ok": True, "message": "Kopyala–yapıştır modu: API gerekmez."}
        t0 = time.time()
        out = llm.complete_json("Kısa bir bağlantı testi. Yalnızca istenen JSON'u ver.",
                                'ok alanını true, model alanını kendi model adın yap.', _Ping)
        return {"ok": bool(out.ok), "message": f"Yanıt alındı ({time.time() - t0:.1f} sn) · {getattr(llm, 'model_name', '')}"}

    try:
        return await asyncio.wait_for(asyncio.to_thread(run), timeout=90)
    except Exception as exc:
        return {"ok": False, "message": f"{type(exc).__name__}: {str(exc)[:400]}"}


@app.get("/manual", response_class=HTMLResponse)
def manual_page(request: Request):
    return page(request, "manual.html", req=broker.pending)


@app.get("/api/manual")
def manual_api():
    req = broker.pending
    return {"pending": req.id if req else None, "error": req.error if req else None,
            "job_id": req.job_id if req else None}


@app.get("/manual/{req_id}/prompt.md")
def manual_prompt_download(req_id: str):
    req = broker.pending
    if req is None or req.id != req_id:
        raise HTTPException(404, "Bekleyen istem yok")
    return Response(req.prompt, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="istem_{req.seq:02d}.md"'})


@app.post("/manual/{req_id}")
def manual_submit(request: Request, req_id: str, csrf: str = Form(...), answer: str = Form(...)):
    check_csrf(request, csrf)
    req = broker.pending
    if req is None or req.id != req_id or not broker.submit(req_id, answer):
        return RedirectResponse("/manual", status_code=303)
    attempts = req.attempts
    end = time.monotonic() + 5
    while time.monotonic() < end and broker.pending is req and req.attempts == attempts:
        time.sleep(0.1)
    if broker.pending is None and req.job_id:
        # İş bir sonraki istemi istemeden bitecekse kısa süre bekleyip doğrudan hedef sayfaya dön
        job = jobs.get(req.job_id)
        end = time.monotonic() + 3
        while job and time.monotonic() < end and job.status in ("running", "queued") and broker.pending is None:
            time.sleep(0.1)
        if broker.pending is not None:
            return RedirectResponse("/manual", status_code=303)
        nxt = (job.params or {}).get("next") if job else None
        if job and job.status == "done" and nxt:
            return RedirectResponse(nxt, status_code=303)
        return RedirectResponse(f"/jobs/{req.job_id}", status_code=303)
    return RedirectResponse("/manual", status_code=303)


@app.post("/manual/{req_id}/cancel")
def manual_cancel(request: Request, req_id: str, csrf: str = Form(...)):
    check_csrf(request, csrf)
    req = broker.pending
    broker.cancel(req_id)
    return RedirectResponse(f"/jobs/{req.job_id}" if req and req.job_id else "/", status_code=303)


__all__ = ["app"]
