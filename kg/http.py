"""Önbellekli, nazik HTTP istemcisi (Europe PMC, Semantic Scholar, Crossref, MeSH, UniProt...)."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .config import DATA_DIR, env

log = logging.getLogger(__name__)


PROJECT_URL = "https://github.com/srht/makalekg"
# Wikimedia kuralı: istemciler seri ve yavaş istek atmalı (en az ~1 sn arayla)
HOST_INTERVALS = {"www.wikidata.org": 1.0, "query.wikidata.org": 2.0}
MAX_RETRY_AFTER = 120.0


class RateLimited(ConnectionError):
    """Sunucu (429/503) sınırlamayı tekrar denemelerden sonra da sürdürdü."""

    def __init__(self, url: str, retry_after: float | None = None):
        host = httpx.URL(url).host
        super().__init__(f"{host} çok fazla istek dedi (429)" +
                         (f"; yaklaşık {int(retry_after)} sn sonra tekrar deneyin" if retry_after else ""))
        self.retry_after = retry_after


def user_agent() -> str:
    """Wikimedia User-Agent politikası: uygulama adı/sürümü + iletişim (URL veya e-posta)."""
    contact = env("CONTACT_EMAIL")
    return (f"KnowledgeGraphKurucu/1.0 ({PROJECT_URL}" + (f"; {contact}" if contact else "") +
            f") httpx/{httpx.__version__}")


def _retry_after(r: httpx.Response) -> float | None:
    value = r.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


class Http:
    def __init__(self, cache_dir: Path | None = None, min_interval: float = 0.34, offline: bool = False,
                 max_attempts: int = 5):
        self.cache_dir = cache_dir or DATA_DIR / "cache" / "http"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": user_agent()})
        self.min_interval = min_interval
        self.offline = offline
        self.max_attempts = max_attempts
        self._last: dict[str, float] = {}

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def _pace(self, host: str) -> None:
        interval = max(self.min_interval, HOST_INTERVALS.get(host, 0.0))
        wait = interval - (time.monotonic() - self._last.get(host, 0))
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
            as_text: bool = False, use_cache: bool = True, method: str = "GET") -> Any:
        """method="POST" parametreleri form gövdesinde yollar (uzun SPARQL sorguları için)."""
        path = self._cache_path(url, params)
        if use_cache and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return cached["body"]
        if self.offline:
            raise ConnectionError(f"çevrimdışı mod ve önbellekte yok: {url}")
        host = httpx.URL(url).host
        body: Any = None
        last_retry_after = None
        r: httpx.Response | None = None
        for attempt in range(self.max_attempts):
            self._pace(host)
            try:
                if method == "POST":
                    r = self.client.post(url, data=params, headers=headers)
                else:
                    r = self.client.get(url, params=params, headers=headers)
                self._last[host] = time.monotonic()
                if r.status_code == 404:
                    body = None
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    last_retry_after = _retry_after(r)
                    if last_retry_after and last_retry_after > MAX_RETRY_AFTER:
                        raise RateLimited(url, last_retry_after)
                    delay = last_retry_after or min(60.0, 2.0 ** (attempt + 2))
                    log.warning("%s %d yanıtı verdi; %.0f sn bekleyip tekrar denenecek (%d/%d)",
                                host, r.status_code, delay, attempt + 1, self.max_attempts)
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                body = r.text if as_text else r.json()
                break
            except (httpx.TransportError, json.JSONDecodeError) as exc:
                log.warning("HTTP hatası %s (%s), tekrar deneniyor", url, exc)
                time.sleep(min(60.0, 2.0 ** (attempt + 1)))
        else:
            if r is not None and getattr(r, "status_code", None) in (429, 503):
                raise RateLimited(url, last_retry_after)
            raise ConnectionError(f"istek başarısız: {url}")
        if use_cache:
            path.write_text(json.dumps({"url": url, "params": params, "body": body}), encoding="utf-8")
        return body

    def download(self, url: str, dest: Path) -> Path | None:
        if dest.exists():
            return dest
        if self.offline:
            return None
        r = self.client.get(url)
        if r.status_code != 200:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest
