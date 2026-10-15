"""Ayar yükleme: YAML dosyaları + .env (gizli anahtarlar yalnızca .env'de)."""
from __future__ import annotations

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
