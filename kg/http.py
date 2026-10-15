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


class Http:
    def __init__(self, cache_dir: Path | None = None, min_interval: float = 0.34, offline: bool = False):
        self.cache_dir = cache_dir or DATA_DIR / "cache" / "http"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        contact = env("CONTACT_EMAIL")
        ua = "kg-bone-brain/0.1" + (f" (mailto:{contact})" if contact else "")
        self.client = httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": ua})
        self.min_interval = min_interval
        self.offline = offline
        self._last: dict[str, float] = {}

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
            as_text: bool = False, use_cache: bool = True) -> Any:
        path = self._cache_path(url, params)
        if use_cache and path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return cached["body"]
        if self.offline:
            raise ConnectionError(f"çevrimdışı mod ve önbellekte yok: {url}")
        host = httpx.URL(url).host
        wait = self.min_interval - (time.monotonic() - self._last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        body: Any = None
        for attempt in range(4):
            try:
                r = self.client.get(url, params=params, headers=headers)
                self._last[host] = time.monotonic()
                if r.status_code == 404:
                    body = None
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** (attempt + 1))
                    continue
                r.raise_for_status()
                body = r.text if as_text else r.json()
                break
            except (httpx.TransportError, json.JSONDecodeError) as exc:
                log.warning("HTTP hatası %s (%s), tekrar deneniyor", url, exc)
                time.sleep(2 ** (attempt + 1))
        else:
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
