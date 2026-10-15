"""kg — Kemik–Beyin Ekseni Knowledge Graph komut satırı.

  kg ingest --phase 1 [--pdf data/seed.pdf]     tohum makale: çıkarım + yükleme + spotcheck + tutarsızlıklar
  kg ingest --phase 2 [--set bone_brain]        genişleme: referanslar + atıf yapanlar + sorgular (≤ 200)
  kg infer [--embed] [--agent] [--literature]   hipotez üretimi → HYPOTHESIS kenarları + reports/hypotheses
  kg ask "soru"                                 uzman olmayan kullanıcı için düz Türkçe yanıt
  kg spotcheck [--results dosya.json]           spotcheck.html üret / işaretlenmiş sonucu uygula
  kg backtest --cutoff 2024                     backtest raporu (reports/backtest.md)
  kg accept --phase N                           faz kabul kriterlerini kontrol et
  kg unresolved                                 normalize edilemeyen varlıklar ve bulanık eşleşme önerileri
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kg.config import load_dotenv, sources_config
from kg.discover import norm_doi


def _store(args):
    from kg.graph import get_store
    return get_store(getattr(args, "backend", None))


def cmd_ingest(args) -> int:
    from kg.llm import get_llm
    from kg.pipeline import Pipeline
    from kg.validate import acceptance
    from kg.validate.spotcheck import build_spotcheck

    store = _store(args)
    pipe = Pipeline(store, get_llm(), offline=args.offline, force=args.force)
    if args.phase == 1:
        papers, cites = pipe.phase1_papers(args.pdf), []
    else:
        papers, cites = pipe.expansion_papers(args.set or None)
    if args.limit:
        papers = papers[: args.limit]
    summary = pipe.ingest(papers, cites)
    html, items = build_spotcheck(store, n=10, run_id=summary.run.id)
    info = summary.as_dict()
    print(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"\nspotcheck: {html} ({len(items)} kenar) · tutarsızlıklar: reports/inconsistencies.md")
    snap = store.snapshot()
    if args.phase == 1:
        seed = norm_doi(sources_config()["seed_sets"]["bone_brain"][0]["doi"])
        report = acceptance.write_report(1, acceptance.phase1(snap, seed), info)
    else:
        report = acceptance.write_report(2, acceptance.phase2(snap, summary.normalization_rate), info)
    print(f"kabul raporu: {report}")
    store.close()
    return 0


def cmd_infer(args) -> int:
    from kg.infer.run import run_inference
    from kg.validate import acceptance

    store = _store(args)
    agent_fn = literature_fn = reingest_fn = None
    if args.agent or args.literature:
        from kg.llm import get_llm
        llm = get_llm()
    if args.agent:
        from kg.infer.agent import GraphAgent
        agent_fn = lambda snap: GraphAgent(llm, snap, _cypher(args)).propose(args.task)
    if args.literature:
        from kg.discover import Discoverer
        from kg.infer.literature import literature_search
        from kg.pipeline import Pipeline
        disc = Discoverer()
        literature_fn = lambda snap, h: [r["doi"] for r in literature_search(disc, snap, h.subject_key, h.object_key)]
        if args.ingest_literature:
            pipe = Pipeline(store, llm)
            reingest_fn = lambda dois: pipe.ingest_dois(dois[: args.max_literature])
    scored = run_inference(store, use_embed=args.embed, agent_fn=agent_fn, literature_fn=literature_fn,
                           reingest_fn=reingest_fn)
    for s in scored[:20]:
        h = s.hypothesis
        print(f"{h.score:.3f}  {h.subject_key} -{h.predicate}-> {h.object_key}  [{h.status.value}]")
    print(f"\n{len(scored)} hipotez · kartlar: reports/hypotheses/")
    report = acceptance.write_report(3, acceptance.phase3(store.snapshot()))
    print(f"kabul raporu: {report}")
    store.close()
    return 0


def _cypher(args):
    if (getattr(args, "backend", None) or "").lower() == "memory":
        return None
    from kg.config import env
    if (env("GRAPH_BACKEND", "neo4j") or "").lower() != "neo4j":
        return None
    from kg.infer.agent import ReadOnlyCypher
    return ReadOnlyCypher()


def cmd_ask(args) -> int:
    from kg.infer.agent import GraphAgent
    from kg.llm import get_llm

    store = _store(args)
    agent = GraphAgent(get_llm(), store.snapshot(), _cypher(args))
    print(agent.ask(args.question))
    store.close()
    return 0


def cmd_spotcheck(args) -> int:
    from kg.validate.spotcheck import apply_results, build_spotcheck

    store = _store(args)
    if args.results:
        summary = apply_results(store, json.loads(Path(args.results).read_text(encoding="utf-8")))
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        if not summary["passed"]:
            print("Hata oranı %10'u aştı: çıkarım promptu düzeltilmeli; partinin kenarları needs_review işaretlendi.")
    else:
        path, items = build_spotcheck(store, n=args.n, run_id=args.run, seed=args.seed)
        print(f"{path} ({len(items)} kenar)")
    store.close()
    return 0


def cmd_backtest(args) -> int:
    from kg.validate import acceptance
    from kg.validate.backtest import run_backtest

    store = _store(args)
    result = run_backtest(store.snapshot(), args.cutoff, k=args.k)
    print(json.dumps({k: v for k, v in result.items() if k not in ("top", "calibration")}, ensure_ascii=False, indent=1))
    print("rapor: reports/backtest.md")
    acceptance.write_report(4, acceptance.phase4())
    store.close()
    return 0


def cmd_accept(args) -> int:
    from kg.validate import acceptance

    store = _store(args)
    snap = store.snapshot()
    if args.phase == 1:
        checks = acceptance.phase1(snap, norm_doi(sources_config()["seed_sets"]["bone_brain"][0]["doi"]))
    elif args.phase == 2:
        checks = acceptance.phase2(snap)
    elif args.phase == 3:
        checks = acceptance.phase3(snap)
    else:
        checks = acceptance.phase4()
    path = acceptance.write_report(args.phase, checks)
    for c in checks:
        print({True: "GEÇTİ ", False: "KALDI ", None: "BEKLİYOR"}[c["passed"]], c["criterion"], "—", c["detail"])
    print(f"rapor: {path}")
    store.close()
    return 0 if all(c["passed"] for c in checks) else 1


def cmd_unresolved(args) -> int:
    from kg.pipeline import unresolved_entities

    store = _store(args)
    for row in unresolved_entities(store):
        print(json.dumps(row, ensure_ascii=False))
    store.close()
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
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
