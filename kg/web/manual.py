"""Elle (kopyala-yapıştır) LLM modu için web aracısı.

Arka plan işi bir LLM yanıtına ihtiyaç duyduğunda istemi burada bekletir ve durur. Kullanıcı /manual
sayfasında istemi kopyalar, sohbet arayüzüne (ör. Claude.ai) yapıştırır, yanıtı geri yapıştırır.
Yanıt doğrulanır; geçersizse hata gösterilir ve aynı istem için yeni yanıt beklenir.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..llm import LLMError

KIND_TR = {"ExtractionResult": "İddia çıkarımı", "VerificationResult": "Öz-doğrulama",
           "AgentCandidates": "Aday hipotezler", "answer": "Soru yanıtı"}


@dataclass
class PendingRequest:
    id: str
    kind: str
    prompt: str
    job_id: str | None
    job_title: str | None
    seq: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    error: str | None = None
    attempts: int = 0

    @property
    def kind_tr(self) -> str:
        return KIND_TR.get(self.kind, self.kind)


class WebBroker:
    def __init__(self, jobs=None):
        self.jobs = jobs
        self.pending: PendingRequest | None = None
        self._cond = threading.Condition()
        self._answer: str | None = None
        self._cancelled = False
        self._seq: dict[str | None, int] = {}

    def ask(self, kind: str, prompt: str, parse: Callable[[str], Any]) -> Any:
        job = self.jobs.current if self.jobs else None
        job_id = job.id if job else None
        self._seq[job_id] = self._seq.get(job_id, 0) + 1
        req = PendingRequest(uuid.uuid4().hex[:12], kind, prompt, job_id, job.title if job else None, self._seq[job_id])
        with self._cond:
            self.pending, self._answer, self._cancelled = req, None, False
        if job:
            job.status = "waiting"
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._answer is not None or self._cancelled)
                    if self._cancelled:
                        raise LLMError("kullanıcı elle yanıt adımını iptal etti")
                    text, self._answer = self._answer, None
                try:
                    return parse(text)
                except ValueError as exc:
                    req.attempts += 1
                    req.error = str(exc)[:3000]
        finally:
            with self._cond:
                if self.pending is req:
                    self.pending = None
            if job and job.status == "waiting":
                job.status = "running"

    def submit(self, req_id: str, text: str) -> bool:
        with self._cond:
            if self.pending is None or self.pending.id != req_id:
                return False
            self.pending.error = None
            self._answer = text
            self._cond.notify_all()
            return True

    def cancel(self, req_id: str) -> bool:
        with self._cond:
            if self.pending is None or self.pending.id != req_id:
                return False
            self._cancelled = True
            self._cond.notify_all()
            return True
