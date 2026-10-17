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

from ..config import ensure_dirs, env, load_dotenv
from ..llm import get_llm, set_manual_broker
from ..okg import extractor, oai, search as gsearch, standards, wikidata
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
    backend = (env("LLM_BACKEND", "anthropic") or "anthropic").lower()
    if backend == "manual":
        return {"backend": backend, "ready": True, "label": "Kopyala–yapıştır (Claude.ai vb.)", "manual": True}
    if backend == "anthropic":
        ok = bool(env("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN"))
        return {"backend": backend, "ready": ok, "label": f"Anthropic API · {env('LLM_MODEL', 'claude-opus-5-5')}", "manual": False}
    return {"backend": backend, "ready": bool(env("OPENAI_COMPAT_BASE_URL")), "label": "Yerel model", "manual": False}


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
    ids = {doc_id, doc_id + oai.META_SUFFIX}
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
                                  "doc_id": x["doc_id"].removesuffix(oai.META_SUFFIX) if x["doc_id"] != wikidata.WIKIDATA_SOURCE else "",
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
