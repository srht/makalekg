"""Çıkarım döngüsü: üreticiler → tek kuyruk → tekilleştirme → skor → literatür kontrolü → yazma + kart.

Üç üretici (kurallar, PyKEEN, LLM ajanı) adaylarını tek kuyrukta birleştirir. Aynı
(özne, ilişki, nesne) üçlüsü varsa skor güncellenir, yeni kenar açılmaz.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ..config import scoring_config
from ..graph import GraphSnapshot, GraphStore
from ..report import write_card, write_index
from ..schema import Hypothesis, HypothesisStatus
from ..score import chain_confidence, final_score
from .embed import EmbeddingPredictor, attach_embedding_scores
from .literature import hypothesis_status
from .rules import Candidate, rule_candidates
from .writer import HypothesisWriter

log = logging.getLogger(__name__)


@dataclass
class ScoredHypothesis:
    hypothesis: Hypothesis
    penalties: dict[str, float]
    warnings: list[str]


def merge_queue(*batches: list[Candidate]) -> list[Candidate]:
    merged: dict[tuple[str, str, str], Candidate] = {}
    for batch in batches:
        for c in batch:
            prev = merged.get(c.triple)
            if prev is None:
                merged[c.triple] = c
                continue
            gens = sorted(set(prev.generators + c.generators))
            better = c if min(l.score.E for l in c.links) > min(l.score.E for l in prev.links) else prev
            better.generators = gens
            better.embed_score = prev.embed_score if prev.embed_score is not None else c.embed_score
            if better is c and prev.rationale not in c.rationale:
                c.rationale = f"{c.rationale}\n\n{prev.rationale}"
            merged[c.triple] = better
    return list(merged.values())


def score_candidate(snap: GraphSnapshot, c: Candidate, cfg: dict[str, Any]) -> ScoredHypothesis:
    chain = chain_confidence(snap, c.links, cfg)
    S = final_score(chain.C, c.embed_score, cfg)
    weak = chain.weakest
    weakest_text = (f"{snap.name(weak.src)} → {snap.name(weak.dst)} (E = {weak.score.E:.3f}; "
                    f"{weak.score.explanation})") if weak else None
    warnings = []
    for l in c.links:
        if l.score.funding_warning:
            warnings.append(f"Halka {snap.name(l.src)} → {snap.name(l.dst)}: makale endüstri fonlu "
                            f"(doi:{l.edge.props.get('source_doi')}); skor düşürülmedi, bilginize.")
        if l.score.tracer_warning:
            warnings.append(f"Halka {snap.name(l.src)} → {snap.name(l.dst)}: floresan grup (NBD/BODIPY) "
                            "molekülün davranışını değiştirebilir.")
        if l.score.paper.get("retracted"):
            warnings.append(f"Halka {snap.name(l.src)} → {snap.name(l.dst)}: kaynak makale GERİ ÇEKİLMİŞ.")
    h = Hypothesis(subject_key=c.subject, object_key=c.object, predicate=c.predicate,
                   path=[l.as_dict(snap) for l in c.links], score=S, chain_confidence=chain.C,
                   embed_score=c.embed_score, rationale=c.rationale, weakest_link=weakest_text,
                   generators=c.generators)
    return ScoredHypothesis(h, chain.penalties, sorted(set(warnings)))


def run_inference(store: GraphStore, *, use_embed: bool = False, agent_fn: Callable[[GraphSnapshot], list[Candidate]] | None = None,
                  literature_fn: Callable[[GraphSnapshot, Hypothesis], list[str]] | None = None,
                  reingest_fn: Callable[[list[str]], None] | None = None, write: bool = True,
                  cards: bool = True, cfg: dict[str, Any] | None = None) -> list[ScoredHypothesis]:
    cfg = cfg or scoring_config()
    snap = store.snapshot()

    batches = [rule_candidates(snap, scoring=cfg)]
    predictor = None
    if use_embed:
        try:
            predictor = EmbeddingPredictor().fit(snap)
            batches.append(predictor.candidates(snap))
        except RuntimeError as exc:
            log.warning("Embedding atlandı: %s", exc)
            predictor = None
    if agent_fn is not None:
        batches.append(agent_fn(snap))
    queue = merge_queue(*batches)
    attach_embedding_scores(predictor, queue)

    scored = [score_candidate(snap, c, cfg) for c in queue]
    scored = [s for s in scored if s.hypothesis.score >= cfg.get("hypothesis_min_score", 0.0)]

    # Literatür kontrolü: bulunan makaleler boru hattına eklenir; durum yalnızca kanıt grafından okunur
    if literature_fn is not None:
        found: dict[int, list[str]] = {}
        for i, s in enumerate(scored):
            found[i] = literature_fn(snap, s.hypothesis)
            s.hypothesis.literature_dois = found[i]
        new_dois = sorted({d for v in found.values() for d in v})
        if new_dois and reingest_fn is not None:
            reingest_fn(new_dois)
            snap = store.snapshot()
    for s in scored:
        status, dois = hypothesis_status(snap, s.hypothesis.subject_key, s.hypothesis.predicate, s.hypothesis.object_key)
        s.hypothesis.status = status
        if dois:
            s.hypothesis.literature_dois = sorted(set(s.hypothesis.literature_dois + dois))

    scored.sort(key=lambda s: s.hypothesis.score, reverse=True)
    if write:
        writer = HypothesisWriter(store)
        for s in scored:
            writer.write(s.hypothesis)
        if hasattr(store, "flush"):
            store.flush()
    if cards:
        rows = []
        for s in scored:
            path = write_card(snap, s.hypothesis, s.penalties, s.warnings)
            rows.append({"title": f"{snap.name(s.hypothesis.subject_key)} → {snap.name(s.hypothesis.object_key)} "
                                  f"({s.hypothesis.predicate})", "score": s.hypothesis.score,
                         "status": s.hypothesis.status.value, "weakest": s.hypothesis.weakest_link, "path": str(path)})
        write_index(rows)
    return scored


__all__ = ["run_inference", "merge_queue", "score_candidate", "ScoredHypothesis", "HypothesisStatus"]
