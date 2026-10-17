"""Ayar yükleme: YAML dosyaları + .env (gizli anahtarlar yalnızca .env'de)."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
# Konteynerde kalıcı birimlere yönlendirmek için ortam değişkeniyle değiştirilebilir
DATA_DIR = Path(os.environ.get("KG_DATA_DIR") or ROOT / "data")
REPORTS_DIR = Path(os.environ.get("KG_REPORTS_DIR") or ROOT / "reports")


def load_dotenv(path: Path | None = None) -> None:
    """Basit .env okuyucu; ortamda zaten tanımlı değişkenleri ezmez."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


# Arayüzden girilen ayarlar (ör. dil modeli seçimi, API anahtarları) data/settings.json'da tutulur ve ortam
# değişkenlerinden önce gelir; böylece Dokploy'da yeniden dağıtmadan değiştirilebilir.
RUNTIME_KEYS = ("LLM_BACKEND", "LLM_MODEL", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL",
                "OPENAI_COMPAT_BASE_URL", "OPENAI_COMPAT_API_KEY", "OPENAI_COMPAT_MODEL", "MANUAL_MODEL_NAME")
SECRET_KEYS = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_COMPAT_API_KEY")
_runtime_cache: dict[str, Any] = {"mtime": None, "data": {}}


def _runtime_path() -> Path:
    return DATA_DIR / "settings.json"


def runtime_settings() -> dict[str, str]:
    path = _runtime_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return {}
    if _runtime_cache["mtime"] != (str(path), mtime):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        _runtime_cache.update(mtime=(str(path), mtime), data={k: str(v) for k, v in data.items() if k in RUNTIME_KEYS})
    return dict(_runtime_cache["data"])


def save_runtime_settings(updates: dict[str, str | None]) -> None:
    """None veya boş değer o ayarı siler (ortam değişkenine geri dönülür). Dosya yalnız sahibince okunur (0600)."""
    data = runtime_settings()
    for k, v in updates.items():
        if k not in RUNTIME_KEYS:
            continue
        if v in (None, ""):
            data.pop(k, None)
        else:
            data[k] = str(v).strip()
    path = _runtime_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".settings.json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def setting(key: str, default: str | None = None) -> str | None:
    """Önce arayüz ayarı (data/settings.json), sonra ortam değişkeni."""
    value = runtime_settings().get(key) if key in RUNTIME_KEYS else None
    return value if value not in (None, "") else env(key, default)


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def scoring_config() -> dict[str, Any]:
    return load_yaml("scoring.yaml")


def rules_config() -> dict[str, Any]:
    return load_yaml("rules.yaml")


def sources_config() -> dict[str, Any]:
    return load_yaml("sources.yaml")


def ensure_dirs() -> None:
    for sub in ("raw", "extractions", "cache"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
