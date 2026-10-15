"""kg — Kemik–Beyin Ekseni Knowledge Graph komut satırı.

  kg ingest --phase 1 [--pdf data/seed.pdf]     tohum makale: çıkarım + yükleme + spotcheck + tutarsızlıklar
  kg ingest --phase 2 [--set bone_brain]        genişleme: referanslar + atıf yapanlar + sorgular (≤ 200)
  kg infer [--embed] [--agent] [--literature]   hipotez üretimi → HYPOTHESIS kenarları + reports/hypotheses
  kg ask "soru"                                 uzman olmayan kullanıcı için düz Türkçe yanıt
  kg spotcheck [--results dosya.json]           spotcheck.html üret / işaretlenmiş sonucu uygula
  kg backtest --cutoff 2024                     backtest raporu (reports/backtest.md)
  kg accept --phase N                           faz kabul kriterlerini kontrol et
  kg unresolved                                 normalize edilemeyen varlıklar ve bulanık eşleşme önerileri
  kg web [--port 8000]                          web arayüzü
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kg.config import load_dotenv


def _store(args):
    from kg.service import open_store
    return open_store(getattr(args, "backend", None))


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def cmd_ingest(args) -> int:
    from kg import service

    store = _store(args)
    try:
        _print(service.ingest(store, phase=args.phase, pdf=args.pdf, sets=args.set, limit=args.limit,
                              offline=args.offline, force=args.force))
    finally:
        store.close()
    return 0


def cmd_infer(args) -> int:
    from kg import service

    store = _store(args)
    try:
        _print(service.infer(store, embed=args.embed, agent=args.agent, task=args.task, literature=args.literature,
                             ingest_literature=args.ingest_literature, max_literature=args.max_literature,
                             backend=args.backend))
        print("kartlar: reports/hypotheses/")
    finally:
        store.close()
    return 0


def cmd_ask(args) -> int:
    from kg import service

    store = _store(args)
    try:
        print(service.ask(store, args.question, backend=args.backend)["answer"])
    finally:
        store.close()
    return 0


def cmd_spotcheck(args) -> int:
    from kg import service

    store = _store(args)
    try:
        if args.results:
            summary = service.spotcheck_apply(store, json.loads(Path(args.results).read_text(encoding="utf-8")))
            _print(summary)
            if not summary["passed"]:
                print("Hata oranı %10'u aştı: çıkarım promptu düzeltilmeli; partinin kenarları needs_review işaretlendi.")
        else:
            _print(service.spotcheck_build(store, n=args.n, run_id=args.run, seed=args.seed))
    finally:
        store.close()
    return 0


def cmd_backtest(args) -> int:
    from kg import service

    store = _store(args)
    try:
        result = service.backtest(store, args.cutoff, k=args.k)
        _print({k: v for k, v in result.items() if k != "top"})
        print("rapor: reports/backtest.md")
    finally:
        store.close()
    return 0


def cmd_accept(args) -> int:
    from kg import service

    store = _store(args)
    try:
        checks = service.acceptance_check(store, args.phase)
    finally:
        store.close()
    for c in checks:
        print({True: "GEÇTİ ", False: "KALDI ", None: "BEKLİYOR"}[c["passed"]], c["criterion"], "—", c["detail"])
    return 0 if all(c["passed"] for c in checks) else 1


def cmd_unresolved(args) -> int:
    from kg.pipeline import unresolved_entities

    store = _store(args)
    try:
        for row in unresolved_entities(store):
            print(json.dumps(row, ensure_ascii=False))
    finally:
        store.close()
    return 0


def cmd_web(args) -> int:
    import uvicorn
    uvicorn.run("kg.web.app:app", host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips="*")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["neo4j", "memory"], help="graf deposu (varsayılan: .env GRAPH_BACKEND)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="boru hattını çalıştır")
    s.add_argument("--phase", type=int, choices=[1, 2], default=1)
    s.add_argument("--pdf", help="tohum makale PDF'i (Faz 1)")
    s.add_argument("--set", action="append", help="yalnızca bu tohum seti (Faz 2), birden çok verilebilir")
    s.add_argument("--limit", type=int, help="en fazla bu kadar makale işle")
    s.add_argument("--offline", action="store_true", help="ağ erişimi olmadan (yalnızca önbellek/yerel PDF)")
    s.add_argument("--force", action="store_true", help="önbellekteki çıkarımları yok say, yeniden çıkar")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("infer", help="hipotez üret")
    s.add_argument("--embed", action="store_true", help="PyKEEN bağlantı tahmini (pip install -e '.[embed]')")
    s.add_argument("--agent", action="store_true", help="LLM ajanını üretici olarak ekle")
    s.add_argument("--task", default="En az test edilmiş ama en güçlü bağlı molekülleri bul ve hipotez öner.")
    s.add_argument("--literature", action="store_true", help="Europe PMC literatür kontrolü")
    s.add_argument("--ingest-literature", action="store_true", help="bulunan makaleleri boru hattına ekle")
    s.add_argument("--max-literature", type=int, default=20)
    s.set_defaults(fn=cmd_infer)

    s = sub.add_parser("ask", help="grafa soru sor")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("spotcheck", help="çıkarım doğruluğu kontrolü")
    s.add_argument("--results", help="spotcheck.html'den indirilen JSON")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--run", help="ingest run id (varsayılan: son)")
    s.add_argument("--seed", type=int)
    s.set_defaults(fn=cmd_spotcheck)

    s = sub.add_parser("backtest", help="backtest")
    s.add_argument("--cutoff", type=int, default=2024, help="kesim yılı (bu yıldan önceki makalelerle kurulur)")
    s.add_argument("--k", type=int, default=20)
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("accept", help="faz kabul kriterleri")
    s.add_argument("--phase", type=int, choices=[1, 2, 3, 4], required=True)
    s.set_defaults(fn=cmd_accept)

    s = sub.add_parser("unresolved", help="normalize edilemeyen varlıklar")
    s.set_defaults(fn=cmd_unresolved)

    s = sub.add_parser("web", help="web arayüzünü başlat")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_web)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
