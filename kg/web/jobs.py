"""Arka plan iş kuyruğu: ingest / infer / backtest gibi uzun işleri tek bir işçi iş parçacığında sırayla
çalıştırır, loglarını yakalar ve geçmişi data/jobs.json'a yazar. Aynı anda tek iş çalışır; bu hem
graf yazmalarını hem LLM maliyetini öngörülebilir tutar."""
from __future__ import annotations

import json
import logging
import queue
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..config import DATA_DIR

MAX_LOG_LINES = 2000
MAX_HISTORY = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: str
    title: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed | interrupted
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None

    def public(self, with_log: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not with_log:
            d.pop("log")
        return d


class _JobLogHandler(logging.Handler):
    def __init__(self, manager: "JobManager"):
        super().__init__(level=logging.INFO)
        self.manager = manager
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        job = self.manager.current
        if job is None or threading.get_ident() != self.manager.worker_ident:
            return
        try:
            job.log.append(self.format(record))
            if len(job.log) > MAX_LOG_LINES:
                del job.log[: len(job.log) - MAX_LOG_LINES]
        except Exception:  # log yakalama asla işi düşürmemeli
            pass


class JobManager:
    def __init__(self, on_finish: Callable[[Job], None] | None = None):
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.q: queue.Queue[tuple[Job, Callable[[], Any]]] = queue.Queue()
        self.current: Job | None = None
        self.worker_ident: int | None = None
        self.on_finish = on_finish
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.path = DATA_DIR / "jobs.json"
        self._load()
        logging.getLogger().addHandler(_JobLogHandler(self))
        self.thread = threading.Thread(target=self._worker, name="kg-job-worker", daemon=True)
        self.thread.start()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for d in json.loads(self.path.read_text(encoding="utf-8")):
                job = Job(**d)
                if job.status in ("queued", "running"):
                    job.status, job.error = "interrupted", "sunucu yeniden başlatıldı"
                self.jobs[job.id] = job
                self.order.append(job.id)
        except (json.JSONDecodeError, TypeError):
            pass

    def _save(self) -> None:
        """Geçmişi atomik yazar; istek ve işçi iş parçacıkları aynı anda çağırabilir."""
        try:
            with self.save_lock:
                with self.lock:
                    keep = self.order[-MAX_HISTORY:]
                    data = [self.jobs[i].public() for i in keep if i in self.jobs]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(f".{self.path.name}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
                tmp.replace(self.path)
        except OSError as exc:  # geçmiş yazılamasa da işler çalışmaya devam eder
            logging.getLogger("kg.web.jobs").warning("İş geçmişi yazılamadı: %s", exc)

    def submit(self, kind: str, title: str, params: dict[str, Any], fn: Callable[[], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, title=title, params=params)
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.q.put((job, fn))
        self._save()
        return job

    def list(self) -> list[Job]:
        with self.lock:
            return [self.jobs[i] for i in reversed(self.order) if i in self.jobs]

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def busy(self) -> bool:
        return self.current is not None or not self.q.empty()

    def _worker(self) -> None:
        self.worker_ident = threading.get_ident()
        log = logging.getLogger("kg.web.jobs")
        while True:
            job, fn = self.q.get()
            self.current = job
            job.status, job.started_at = "running", _now()
            self._save()
            log.info("İş başladı: %s", job.title)
            try:
                job.result = fn()
                job.status = "done"
                log.info("İş tamamlandı")
            except Exception as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                log.error("İş başarısız: %s\n%s", job.error, traceback.format_exc())
            finally:
                job.finished_at = _now()
                self.current = None
                self._save()
                if self.on_finish:
                    try:
                        self.on_finish(job)
                    except Exception:
                        pass
