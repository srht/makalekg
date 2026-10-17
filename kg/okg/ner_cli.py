"""Hızlı varlık modeli boru hattı — komut satırı.

  python -m kg.okg.ner_cli labels   <proje> --domain elektrik            # hazır tür listesi (veya --llm)
  python -m kg.okg.ner_cli annotate <proje> --n 200 --per-request 10     # dil modeliyle eğitim verisi (Anthropic'te Batch API)
  python -m kg.okg.ner_cli train    <proje> [--kind gliner|sozluk] [--epochs 5]
  python -m kg.okg.ner_cli apply    <proje> [--all] [--threshold 0.5]  # kalan belgelere uygula
  python -m kg.okg.ner_cli run      <proje> --domain elektrik --n 200    # hepsi sırayla
  python -m kg.okg.ner_cli status   <proje>

Proje web arayüzündeki adres adıdır (/p/<proje>). Kopyala–yapıştır modunda istemler terminalde gösterilir.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..config import load_dotenv
from ..llm import get_llm
from . import ner
from .store import Project


def _project(slug: str) -> Project:
    p = Project(slug)
    if not p.exists():
        sys.exit(f"Proje bulunamadı: {slug}")
    return p


def cmd_labels(p: Project, a: argparse.Namespace) -> None:
    cfg = ner.load_config(p)
    if a.domain:
        cfg["domain"] = a.domain
    if a.llm:
        labels = [{"name": l.name, "description": l.description or ""} for l in ner.suggest_labels(get_llm(), cfg["domain"])]
    else:
        key = a.preset or ner.match_preset(cfg["domain"])
        if not key:
            sys.exit(f"Hazır liste yok. Seçenekler: {', '.join(ner.DOMAIN_PRESETS)} (veya --llm)")
        labels = [{"name": n, "description": d} for n, d in ner.DOMAIN_PRESETS[key]["labels"]]
    cfg["labels"] = labels
    ner.save_config(p, cfg)
    for l in labels:
        print(f"- {l['name']}: {l['description']}")


def cmd_annotate(p: Project, a: argparse.Namespace) -> None:
    llm = get_llm()
    extra = {k: v for k, v in {"count_mode": "total" if a.total else None, "source": a.source,
                               "keyword": a.keyword}.items() if v is not None}
    res = ner.annotate(p, llm, n=a.n, per_request=a.per_request,
                       use_batch_api=None if not a.no_batch_api else False, **extra)
    print(json.dumps(res, ensure_ascii=False))
    return res


def cmd_train(p: Project, a: argparse.Namespace) -> None:
    m = ner.train(p, kind=a.kind, epochs=a.epochs, test_ratio=a.test_ratio,
                  **({"batch_size": a.batch_size} if a.batch_size else {}))
    print(json.dumps({k: m.get(k) for k in ("model_kind", "n_train", "n_test", "base", "test")}, ensure_ascii=False, indent=1))
    return m


def cmd_apply(p: Project, a: argparse.Namespace) -> None:
    extra = {k: v for k, v in {"scope": "all" if a.all else None, "limit": a.limit}.items() if v is not None}
    res = ner.apply(p, threshold=a.threshold, **extra)
    print(json.dumps(res, ensure_ascii=False))
    return res


def cmd_try(p: Project, _a: argparse.Namespace) -> dict:
    """stdin'deki metni eğitilmiş modelle etiketler (web arayüzündeki "Dene" için ayrı süreçte çalışır)."""
    model = ner.trained_model(p)
    if model is None:
        raise SystemExit("Henüz eğitilmiş model yok")
    cfg = ner.load_config(p)
    text = sys.stdin.read()[:20000]
    out = {"entities": model.predict([text], ner.label_names(cfg), float(cfg.get("threshold", 0.5)))[0]}
    print(json.dumps(out, ensure_ascii=False))
    return out


def cmd_status(p: Project, _a: argparse.Namespace) -> None:
    st = ner.status(p)
    print(json.dumps(st.__dict__, ensure_ascii=False, indent=1, default=str))


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stderr)
    ap = argparse.ArgumentParser(prog="python -m kg.okg.ner_cli", description="Hızlı varlık modeli boru hattı")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("labels", "annotate", "train", "apply", "run", "status", "try"):
        s = sub.add_parser(name)
        s.add_argument("project")
        s.add_argument("--result-file", default=None, help="sonucu JSON olarak bu dosyaya yaz")
        if name in ("labels", "run"):
            s.add_argument("--domain", default="")
            s.add_argument("--preset", default="", choices=[""] + list(ner.DOMAIN_PRESETS))
            s.add_argument("--llm", action="store_true", help="türleri dil modelinden öner")
        if name in ("annotate", "run"):
            s.add_argument("--n", type=int, default=None, help="belge sayısı (varsayılan: kayıtlı ayar, 200)")
            s.add_argument("--per-request", type=int, default=None)
            s.add_argument("--total", action="store_true", help="--n veri setinin toplam hedefi olsun")
            s.add_argument("--source", choices=["all", "archive", "upload"], default=None)
            s.add_argument("--keyword", default=None)
            s.add_argument("--no-batch-api", action="store_true", help="Anthropic Batch API yerine anlık istekler")
        if name in ("train", "run"):
            s.add_argument("--kind", default=None, choices=["auto", "gliner", "sozluk"])
            s.add_argument("--epochs", type=int, default=None)
            s.add_argument("--batch-size", type=int, default=None)
            s.add_argument("--test-ratio", type=float, default=None)
        if name in ("apply", "run"):
            s.add_argument("--all", action="store_true")
            s.add_argument("--limit", type=int, default=None, help="en fazla kaç belgeye uygulansın")
            s.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args(argv)
    p = _project(a.project)
    if a.cmd == "run":
        if a.domain or a.preset or a.llm or not ner.load_config(p)["labels"]:
            cmd_labels(p, a)
        cmd_annotate(p, a)
        cmd_train(p, a)
        cmd_apply(p, a)
    else:
        res = {"labels": cmd_labels, "annotate": cmd_annotate, "train": cmd_train, "apply": cmd_apply,
               "status": cmd_status, "try": cmd_try}[a.cmd](p, a)
        if a.result_file:
            Path(a.result_file).write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
