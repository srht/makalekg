"""Uygulama servis katmanı: CLI ve web arayüzü aynı işlemleri buradan çağırır."""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from .config import DATA_DIR, REPORTS_DIR, env, sources_config
from .discover import norm_doi
from .graph import GraphSnapshot, GraphStore, get_store

log = logging.getLogger(__name__)


def seed_doi() -> str:
    return norm_doi(sources_config()["seed_sets"]["bone_brain"][0]["doi"])


def open_store(backend: str | None = None) -> GraphStore:
    return get_store(backend)


# ---------------------------------------------------------------- işlemler


def ingest(store: GraphStore, phase: int = 1, pdf: str | None = None, sets: list[str] | None = None,
           limit: int | None = None, offline: bool = False, force: bool = False) -> dict[str, Any]:
    from .llm import get_llm
    from .pipeline import Pipeline
    from .validate import acceptance
    from .validate.spotcheck import build_spotcheck

    pipe = Pipeline(store, get_llm(), offline=offline, force=force)
    if phase == 1:
        papers, cites = pipe.phase1_papers(pdf), []
    else:
        papers, cites = pipe.expansion_papers(sets or None)
    if limit:
        papers = papers[:limit]
    log.info("%d makale işlenecek", len(papers))
    summary = pipe.ingest(papers, cites)
    html, items = build_spotcheck(store, n=10, run_id=summary.run.id)
    info = summary.as_dict()
    snap = store.snapshot()
    if phase == 1:
        report = acceptance.write_report(1, acceptance.phase1(snap, seed_doi()), info)
    else:
        report = acceptance.write_report(2, acceptance.phase2(snap, summary.normalization_rate), info)
    log.info("Bitti: %d makale, %d iddia, %d reddedilen", summary.run.n_papers, summary.run.n_claims,
             summary.run.n_rejected)
    return {**info, "spotcheck": str(html), "spotcheck_items": len(items), "acceptance_report": str(report)}


def _cypher(backend: str | None):
    if (backend or env("GRAPH_BACKEND", "neo4j") or "").lower() != "neo4j":
        return None
    from .infer.agent import ReadOnlyCypher
    return ReadOnlyCypher()


def infer(store: GraphStore, embed: bool = False, agent: bool = False, task: str | None = None,
          literature: bool = False, ingest_literature: bool = False, max_literature: int = 20,
          backend: str | None = None) -> dict[str, Any]:
    from .infer.run import run_inference
    from .validate import acceptance

    agent_fn = literature_fn = reingest_fn = None
    llm = None
    if agent or ingest_literature:
        from .llm import get_llm
        llm = get_llm()
    if agent:
        from .infer.agent import GraphAgent
        task = task or "En az test edilmiş ama en güçlü bağlı molekülleri bul ve hipotez öner."
        agent_fn = lambda snap: GraphAgent(llm, snap, _cypher(backend)).propose(task)
    if literature:
        from .discover import Discoverer
        from .infer.literature import literature_search
        disc = Discoverer()
        literature_fn = lambda snap, h: [r["doi"] for r in literature_search(disc, snap, h.subject_key, h.object_key)]
        if ingest_literature:
            from .pipeline import Pipeline
            pipe = Pipeline(store, llm)
            reingest_fn = lambda dois: pipe.ingest_dois(dois[:max_literature])
    scored = run_inference(store, use_embed=embed, agent_fn=agent_fn, literature_fn=literature_fn,
                           reingest_fn=reingest_fn)
    report = acceptance.write_report(3, acceptance.phase3(store.snapshot()))
    log.info("%d hipotez üretildi", len(scored))
    return {"n_hypotheses": len(scored), "acceptance_report": str(report),
            "top": [{"subject": s.hypothesis.subject_key, "predicate": s.hypothesis.predicate,
                     "object": s.hypothesis.object_key, "score": s.hypothesis.score,
                     "status": s.hypothesis.status.value} for s in scored[:20]]}


def ask(store: GraphStore, question: str, backend: str | None = None) -> dict[str, Any]:
    from .infer.agent import GraphAgent
    from .llm import get_llm

    answer = GraphAgent(get_llm(), store.snapshot(), _cypher(backend)).ask(question)
    return {"question": question, "answer": answer}


def backtest(store: GraphStore, cutoff: int = 2024, k: int = 20) -> dict[str, Any]:
    from .validate import acceptance
    from .validate.backtest import run_backtest

    result = run_backtest(store.snapshot(), cutoff, k=k)
    acceptance.write_report(4, acceptance.phase4())
    return {k_: v for k_, v in result.items() if k_ not in ("calibration",)}


def spotcheck_build(store: GraphStore, n: int = 10, run_id: str | None = None, seed: int | None = None) -> dict[str, Any]:
    from .validate.spotcheck import build_spotcheck

    path, items = build_spotcheck(store, n=n, run_id=run_id, seed=seed)
    return {"path": str(path), "n": len(items)}


def spotcheck_apply(store: GraphStore, results: dict[str, Any]) -> dict[str, Any]:
    from .validate.spotcheck import apply_results

    return apply_results(store, results)


def acceptance_check(store: GraphStore, phase: int) -> list[dict[str, Any]]:
    from .validate import acceptance

    snap = store.snapshot()
    if phase == 1:
        checks = acceptance.phase1(snap, seed_doi())
    elif phase == 2:
        checks = acceptance.phase2(snap)
    elif phase == 3:
        checks = acceptance.phase3(snap)
    else:
        checks = acceptance.phase4()
    acceptance.write_report(phase, checks)
    return checks


# ---------------------------------------------------------------- okuma (web için)


def stats(snap: GraphSnapshot) -> dict[str, Any]:
    labels = Counter(n.get("label") for n in snap.nodes.values())
    edges = Counter(e.type for e in snap.edges)
    ents = snap.entities()
    resolved = sum(1 for n in ents.values() if not n.get("unresolved", True))
    return {
        "papers": labels.get("Paper", 0), "experiments": labels.get("Experiment", 0),
        "claims": labels.get("Claim", 0), "entities": len(ents),
        "normalization_rate": (resolved / len(ents)) if ents else None,
        "evidence_edges": edges.get("SUPPORTS", 0) + edges.get("CONTRADICTS", 0),
        "hypotheses": edges.get("HYPOTHESIS", 0),
        "needs_review": sum(1 for e in snap.edges if e.props.get("needs_review")),
        "verified": sum(1 for e in snap.edges if e.props.get("verified_by_human")),
        "by_label": dict(labels), "by_edge": dict(edges),
    }


def claims_table(snap: GraphSnapshot, query: str | None = None, limit: int = 300) -> list[dict[str, Any]]:
    from .score import describe_experiment, evidence_strength, independent_replications

    rows = []
    q = (query or "").lower().strip()
    for key, n in snap.nodes.items():
        if n.get("label") != "Claim":
            continue
        subj = next((snap.name(e.dst) for e in snap.out_edges(key, ["SUBJECT"])), "?")
        obj = next((snap.name(e.dst) for e in snap.out_edges(key, ["OBJECT"])), "?")
        text = n.get("text") or ""
        if q and q not in text.lower() and q not in subj.lower() and q not in obj.lower():
            continue
        evidence = []
        for e, exp, paper in snap.evidence_for_claim(key):
            s = evidence_strength(snap, e.src, key, e.props)
            evidence.append({"relation": e.type, "doi": paper.get("doi"), "year": paper.get("year"),
                             "figure": e.props.get("figure"), "source_sentence": e.props.get("source_sentence"),
                             "p_value": e.props.get("p_value"), "E": s.E,
                             "level": describe_experiment(exp, paper, independent_replications(snap, e.src)),
                             "needs_review": e.props.get("needs_review", False),
                             "verified": e.props.get("verified_by_human", False),
                             "verification": e.props.get("verification")})
        rows.append({"id": key, "text": text, "polarity": n.get("polarity"), "predicate": n.get("predicate"),
                     "subject": subj, "object": obj, "evidence": evidence,
                     "best_E": max((x["E"] for x in evidence), default=0.0)})
    rows.sort(key=lambda r: -r["best_E"])
    return rows[:limit]


def hypotheses_table(snap: GraphSnapshot) -> list[dict[str, Any]]:
    rows = []
    for e in snap.hypotheses():
        p = e.props
        rows.append({"id": e.id, "subject": snap.name(e.src), "object": snap.name(e.dst),
                     "predicate": p.get("predicate"), "score": p.get("score", 0.0),
                     "chain_confidence": p.get("chain_confidence"), "embed_score": p.get("embed_score"),
                     "status": p.get("status"), "weakest_link": p.get("weakest_link"),
                     "rationale": p.get("rationale"), "path": p.get("path") or [],
                     "generators": p.get("generators") or [], "literature_dois": p.get("literature_dois") or [],
                     "created_at": p.get("created_at")})
    rows.sort(key=lambda r: -(r["score"] or 0))
    return rows


def ingest_runs(store: GraphStore) -> list[dict[str, Any]]:
    runs = getattr(store, "runs", None)
    if runs is not None:
        return sorted(runs, key=lambda r: r.get("date", ""), reverse=True)
    try:
        rows = store._run("MATCH (r:IngestRun) RETURN properties(r) AS p ORDER BY r.date DESC LIMIT 20")
        return [r["p"] for r in rows]
    except Exception:
        return []


def review_queue(limit: int = 200) -> list[dict[str, Any]]:
    path = DATA_DIR / "review_queue.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit]


def report_files() -> list[str]:
    """reports/ altındaki markdown raporlarının göreli yolları."""
    if not REPORTS_DIR.exists():
        return []
    return sorted(str(p.relative_to(REPORTS_DIR)) for p in REPORTS_DIR.rglob("*.md"))


def spotcheck_sample() -> dict[str, Any] | None:
    path = DATA_DIR / "spotcheck_sample.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def spotcheck_summary() -> dict[str, Any] | None:
    path = DATA_DIR / "spotcheck_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
