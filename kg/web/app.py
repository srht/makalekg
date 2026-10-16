"""Web arayüzü (FastAPI + Jinja2).

Sayfalar: pano, iddialar, hipotezler, soru sor, spot-check, inceleme kuyruğu, raporlar, işler.
Uzun işler (ingest/infer/backtest/ask) arka plan iş kuyruğunda çalışır; sayfa logları canlı gösterir.

Güvenlik: tüm sayfalar HTTP Basic ile korunur (APP_USERNAME / APP_PASSWORD). Şifre ayarlanmadıysa
uygulama hiçbir sayfayı sunmaz (yalnızca /healthz). POST formları CSRF jetonu ister.
"""
from __future__ import annotations

import base64
import hmac
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import markdown as md
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from .. import service
from ..config import DATA_DIR, REPORTS_DIR, ensure_dirs, env, load_dotenv
from ..llm import set_manual_broker
from .jobs import JobManager
from .guide import build_guide, save_state
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
    if _store is not None:
        _store.close()


app = FastAPI(title="Kemik–Beyin KG", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

CSRF_SECRET = (env("APP_SECRET") or secrets.token_hex(32)).encode()


def csrf_token(user: str) -> str:
    return hmac.new(CSRF_SECRET, user.encode(), "sha256").hexdigest()


def render_md(text: str) -> Markup:
    # Raporlar yerel olarak üretilir; yine de ham HTML'yi kaçırarak dönüştür
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    return Markup(md.markdown(safe, extensions=["tables", "fenced_code"]))


templates.env.filters["md"] = render_md
templates.env.filters["pct"] = lambda v: "—" if v is None else f"{v:.0%}"
templates.env.filters["num"] = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"


# ---------------------------------------------------------------- graf erişimi

_store = None
_snap_cache: dict[str, Any] = {"t": 0.0, "snap": None, "error": None}
SNAP_TTL = 15.0


def backend() -> str:
    return (env("GRAPH_BACKEND", "neo4j") or "neo4j").lower()


def read_store():
    """Okuma için: Neo4j'de paylaşılan sürücü, bellek-içi depoda diskten taze kopya."""
    global _store
    if backend() == "memory":
        return service.open_store("memory")
    if _store is None:
        _store = service.open_store("neo4j")
    return _store


def job_store():
    return service.open_store(backend())


def snapshot(force: bool = False):
    now = time.monotonic()
    if not force and _snap_cache["snap"] is not None and now - _snap_cache["t"] < SNAP_TTL:
        return _snap_cache["snap"], None
    try:
        snap = read_store().snapshot()
        _snap_cache.update(t=now, snap=snap, error=None)
        return snap, None
    except Exception as exc:
        log.warning("Graf okunamadı: %s", exc)
        _snap_cache.update(error=f"{type(exc).__name__}: {exc}")
        return _snap_cache["snap"], _snap_cache["error"]


def invalidate(_job=None) -> None:
    _snap_cache.update(t=0.0)


jobs = JobManager(on_finish=invalidate)
broker = WebBroker(jobs)
set_manual_broker(broker)


def run_with_store(fn, *args, **kwargs):
    def runner():
        store = job_store()
        try:
            return fn(store, *args, **kwargs)
        finally:
            store.close()
    return runner


# ---------------------------------------------------------------- kimlik doğrulama


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if request.url.path in ("/healthz",):
        return await call_next(request)
    if env("KG_WEB_NO_AUTH") == "1":
        request.state.user = "local"
        return await call_next(request)
    password = env("APP_PASSWORD")
    username = env("APP_USERNAME", "admin")
    if not password:
        return HTMLResponse("<h1>APP_PASSWORD ayarlanmadı</h1><p>Güvenlik için arayüz şifresiz açılmaz. "
                            "Dokploy'da Environment sekmesinde APP_PASSWORD tanımlayın.</p>", status_code=503)
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
                        headers={"WWW-Authenticate": 'Basic realm="kg-bone-brain", charset="UTF-8"'})
    request.state.user = username
    return await call_next(request)


def check_csrf(request: Request, token: str) -> None:
    if not secrets.compare_digest(token or "", csrf_token(request.state.user)):
        raise HTTPException(403, "Geçersiz form jetonu; sayfayı yenileyip tekrar deneyin.")


def page(request: Request, name: str, **ctx) -> HTMLResponse:
    current = jobs.current
    return templates.TemplateResponse(request, name, {
        "csrf": csrf_token(request.state.user), "current_job": current, "queue_busy": jobs.busy(),
        "manual_pending": broker.pending,
        "nav": request.url.path, **ctx})


# ---------------------------------------------------------------- sayfalar


@app.get("/healthz")
def healthz():
    return {"ok": True}


def config_status() -> dict[str, Any]:
    llm_backend = (env("LLM_BACKEND", "anthropic") or "anthropic").lower()
    if llm_backend == "anthropic":
        llm_ready, model = bool(env("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN")), env("LLM_MODEL", "claude-opus-5-5")
    elif llm_backend == "manual":
        llm_ready, model = True, env("MANUAL_MODEL_NAME", "manual (sohbet arayüzü)")
    else:
        llm_ready, model = bool(env("OPENAI_COMPAT_BASE_URL")), env("OPENAI_COMPAT_MODEL")
    return {"llm_backend": llm_backend, "llm_ready": llm_ready, "manual": llm_backend == "manual",
            "llm_model": model,
            "graph_backend": backend(), "contact_email": bool(env("CONTACT_EMAIL")),
            "seed_pdf": (DATA_DIR / "uploads" / "seed.pdf").exists()}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    snap, error = snapshot()
    stats = service.stats(snap) if snap else None
    runs = []
    try:
        runs = service.ingest_runs(read_store())[:5]
    except Exception:
        pass
    sets = list((service.sources_config().get("seed_sets") or {}).keys())
    cfg = config_status()
    spot = service.spotcheck_summary()
    phase1_checks = None
    if snap and stats and stats["claims"] > 0:
        try:
            from ..validate import acceptance
            phase1_checks = acceptance.phase1(snap, service.seed_doi())
        except Exception as exc:  # rehber, kabul kontrolü hata verse de açılmalı
            log.warning("Faz 1 kabul kontrolü hesaplanamadı: %s", exc)
    guide = build_guide(stats, cfg, spot, runs[0].get("id") if runs else None, phase1_checks, jobs)
    return page(request, "dashboard.html", stats=stats, error=error, cfg=cfg, runs=runs, guide=guide,
                recent_jobs=jobs.list()[:6], sets=sets, spot=spot, latest_run=runs[0].get("id") if runs else None)


@app.post("/guide/approve")
def guide_approve(request: Request, csrf: str = Form(...), what: str = Form(...), run_id: str = Form("")):
    """Kullanıcının açık onayı: Faz 1 sonuçlarını kabul etme veya spot-check sonucuyla devam etme."""
    check_csrf(request, csrf)
    if what == "phase1":
        save_state(phase1_approved=True)
    elif what == "spotcheck":
        save_state(spotcheck_accepted_run=run_id)
    elif what == "reset_phase1":
        save_state(phase1_approved=False)
    else:
        raise HTTPException(400)
    return RedirectResponse("/", status_code=303)


@app.get("/claims", response_class=HTMLResponse)
def claims(request: Request, q: str = ""):
    snap, error = snapshot()
    rows = service.claims_table(snap, q) if snap else []
    return page(request, "claims.html", rows=rows, q=q, error=error)


@app.get("/hypotheses", response_class=HTMLResponse)
def hypotheses(request: Request, status: str = ""):
    snap, error = snapshot()
    rows = service.hypotheses_table(snap) if snap else []
    if status:
        rows = [r for r in rows if r["status"] == status]
    return page(request, "hypotheses.html", rows=rows, status=status, error=error)


@app.get("/hypotheses/{hid}", response_class=HTMLResponse)
def hypothesis_detail(request: Request, hid: str):
    snap, error = snapshot()
    row = next((r for r in service.hypotheses_table(snap) if r["id"] == hid), None) if snap else None
    if row is None:
        raise HTTPException(404, "Hipotez bulunamadı")
    return page(request, "hypothesis.html", h=row, error=error)


@app.get("/ask", response_class=HTMLResponse)
def ask_page(request: Request):
    asked = [j for j in jobs.list() if j.kind == "ask"][:10]
    return page(request, "ask.html", asked=asked)


@app.post("/ask")
def ask_submit(request: Request, question: str = Form(...), csrf: str = Form(...)):
    check_csrf(request, csrf)
    question = question.strip()[:2000]
    if not question:
        return RedirectResponse("/ask", status_code=303)
    job = jobs.submit("ask", f"Soru: {question[:80]}", {"question": question},
                      run_with_store(service.ask, question, backend=backend()))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.get("/spotcheck", response_class=HTMLResponse)
def spotcheck_page(request: Request):
    return page(request, "spotcheck.html", sample=service.spotcheck_sample(), summary=service.spotcheck_summary())


@app.post("/spotcheck")
async def spotcheck_submit(request: Request):
    form = await request.form()
    check_csrf(request, str(form.get("csrf", "")))
    sample = service.spotcheck_sample()
    if not sample:
        raise HTTPException(400, "Örnek yok; önce bir spot-check örneği üretin.")
    items = []
    for i, it in enumerate(sample["items"]):
        label = form.get(f"label_{i}")
        if label in ("correct", "partial", "wrong"):
            items.append({"edge_id": it["edge_id"], "related_edge_ids": it.get("related_edge_ids", []),
                          "label": label, "note": str(form.get(f"note_{i}", ""))[:500]})
    # Graf yazması diğer işlerle çakışmasın diye kuyruktan geçer
    job = jobs.submit("spotcheck_apply", f"Spot-check sonuçları ({len(items)} kenar)", {"n": len(items)},
                      run_with_store(service.spotcheck_apply, {"run_id": sample.get("run_id"), "items": items}))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request):
    return page(request, "review.html", rows=service.review_queue())


@app.get("/reports", response_class=HTMLResponse)
def reports_page(request: Request):
    files = service.report_files()
    has_spot_html = (REPORTS_DIR / "spotcheck.html").exists()
    return page(request, "reports.html", files=files, has_spot_html=has_spot_html)


def _safe_report(rel: str) -> Path:
    if not re.fullmatch(r"[\w\-./]+", rel) or ".." in rel:
        raise HTTPException(404)
    path = (REPORTS_DIR / rel).resolve()
    if REPORTS_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404)
    return path


@app.get("/reports/view/{rel:path}", response_class=HTMLResponse)
def report_view(request: Request, rel: str):
    path = _safe_report(rel)
    if path.suffix == ".html":
        return HTMLResponse(path.read_text(encoding="utf-8"))
    return page(request, "report.html", report_name=rel, body=path.read_text(encoding="utf-8"))


@app.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request):
    return page(request, "jobs.html", rows=jobs.list())


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "İş bulunamadı")
    return page(request, "job.html", job=job)


@app.get("/api/jobs/{job_id}")
def job_api(job_id: str, since: int = 0):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    d = job.public(with_log=False)
    d["log"] = job.log[since:]
    d["log_len"] = len(job.log)
    if job.kind == "ask" and job.status == "done" and isinstance(job.result, dict):
        d["answer_html"] = str(render_md(job.result.get("answer") or ""))
    d["manual_pending"] = bool(broker.pending and broker.pending.job_id == job_id)
    return JSONResponse(d)


# ---------------------------------------------------------------- elle (kopyala-yapıştır) mod


@app.get("/manual", response_class=HTMLResponse)
def manual_page(request: Request):
    return page(request, "manual.html", req=broker.pending, cfg=config_status())


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
                    headers={"Content-Disposition": f'attachment; filename="istem_{req.seq:02d}_{req.kind}.md"'})


@app.post("/manual/{req_id}")
def manual_submit(request: Request, req_id: str, csrf: str = Form(...), answer: str = Form(...)):
    check_csrf(request, csrf)
    req = broker.pending
    if req is None or req.id != req_id or not broker.submit(req_id, answer):
        return RedirectResponse("/manual", status_code=303)
    # İşçinin yanıtı doğrulamasını kısa süre bekle: hata varsa aynı sayfada göster
    attempts = req.attempts
    end = time.monotonic() + 5
    while time.monotonic() < end and broker.pending is req and req.attempts == attempts:
        time.sleep(0.1)
    if broker.pending is None and req.job_id:
        return RedirectResponse(f"/jobs/{req.job_id}", status_code=303)
    return RedirectResponse("/manual", status_code=303)


@app.post("/manual/{req_id}/cancel")
def manual_cancel(request: Request, req_id: str, csrf: str = Form(...)):
    check_csrf(request, csrf)
    req = broker.pending
    broker.cancel(req_id)
    return RedirectResponse(f"/jobs/{req.job_id}" if req and req.job_id else "/manual", status_code=303)


# ---------------------------------------------------------------- iş başlatma


def _flag(v: str | None) -> bool:
    return v in ("1", "on", "true", "yes")


@app.post("/jobs/ingest")
async def start_ingest(request: Request, csrf: str = Form(...), phase: int = Form(1), sets: list[str] = Form([]),
                       limit: str = Form(""), force: str = Form(""), pdf: UploadFile | None = File(None)):
    check_csrf(request, csrf)
    pdf_path = None
    if pdf is not None and pdf.filename:
        data = await pdf.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "PDF çok büyük (en fazla 80 MB)")
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, "Dosya PDF değil")
        dest = DATA_DIR / "uploads" / "seed.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        pdf_path = str(dest)
    elif phase == 1 and (DATA_DIR / "uploads" / "seed.pdf").exists():
        pdf_path = str(DATA_DIR / "uploads" / "seed.pdf")
    lim = int(limit) if limit.strip().isdigit() else None
    params = {"phase": phase, "sets": sets, "limit": lim, "force": _flag(force), "pdf": bool(pdf_path)}
    title = "Faz 1: tohum makale" if phase == 1 else f"Faz 2: genişleme ({', '.join(sets) or 'tüm setler'})"
    job = jobs.submit("ingest", title, params, run_with_store(
        service.ingest, phase=phase, pdf=pdf_path, sets=sets or None, limit=lim, force=_flag(force)))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.post("/jobs/infer")
def start_infer(request: Request, csrf: str = Form(...), embed: str = Form(""), agent: str = Form(""),
                literature: str = Form(""), ingest_literature: str = Form(""), task: str = Form("")):
    check_csrf(request, csrf)
    params = {"embed": _flag(embed), "agent": _flag(agent), "literature": _flag(literature),
              "ingest_literature": _flag(ingest_literature), "task": task.strip()[:1000] or None}
    job = jobs.submit("infer", "Hipotez üretimi", params, run_with_store(service.infer, backend=backend(), **params))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.post("/jobs/backtest")
def start_backtest(request: Request, csrf: str = Form(...), cutoff: int = Form(2024), k: int = Form(20)):
    check_csrf(request, csrf)
    job = jobs.submit("backtest", f"Backtest (kesim {cutoff})", {"cutoff": cutoff, "k": k},
                      run_with_store(service.backtest, cutoff, k))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.post("/jobs/spotcheck")
def start_spotcheck(request: Request, csrf: str = Form(...)):
    check_csrf(request, csrf)
    job = jobs.submit("spotcheck", "Yeni spot-check örneği", {}, run_with_store(service.spotcheck_build))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


@app.post("/jobs/accept")
def start_accept(request: Request, csrf: str = Form(...), phase: int = Form(...)):
    check_csrf(request, csrf)
    job = jobs.submit("accept", f"Faz {phase} kabul kontrolü", {"phase": phase},
                      run_with_store(service.acceptance_check, phase))
    return RedirectResponse(f"/jobs/{job.id}", status_code=303)


__all__ = ["app"]
