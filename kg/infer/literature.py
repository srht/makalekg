"""Literatür kontrolü: "bu hipotez test edilmiş mi?"

1. Europe PMC'de özne + nesne adlarıyla arama → grafta olmayan aday makaleler.
2. Bulunan makaleler boru hattına (ingest) eklenir — kanıt grafına ancak boru hattından geçerek girerler.
3. Durum, YALNIZCA kanıt grafına bakılarak belirlenir: özne → nesne arasında doğrudan
   kanıt kenarı onaylanmışsa tested_supported, çürütülmüşse tested_refuted, yoksa untested.
   Sistem kendi tahminini kanıt sanamaz.
"""
from __future__ import annotations

import logging
from typing import Any

from ..discover import Discoverer, norm_doi
from ..graph import GraphSnapshot
from ..schema import EVIDENCE_PREDICATES, HypothesisStatus

log = logging.getLogger(__name__)

OPPOSITE = {"INCREASES": "DECREASES", "DECREASES": "INCREASES"}


def _search_terms(snapshot: GraphSnapshot, key: str) -> str:
    n = snapshot.nodes.get(key, {})
    names = {x for x in (n.get("symbol"), n.get("name"), n.get("measure")) if x}
    return " OR ".join(f'"{x}"' for x in sorted(names)) or f'"{snapshot.name(key)}"'


def literature_search(discoverer: Discoverer, snapshot: GraphSnapshot, subject: str, obj: str,
                      limit: int = 10) -> list[dict[str, Any]]:
    query = f"({_search_terms(snapshot, subject)}) AND ({_search_terms(snapshot, obj)})"
    try:
        results = discoverer.epmc_search(query, page_size=limit)
    except ConnectionError as exc:
        log.warning("Literatür araması başarısız: %s", exc)
        return []
    known = {k for k, v in snapshot.nodes.items() if v.get("label") == "Paper"}
    out = []
    for r in results:
        doi = norm_doi(r.get("doi"))
        if doi and doi not in known:
            out.append({"doi": doi, "title": r.get("title"), "year": r.get("pubYear")})
    return out


def hypothesis_status(snapshot: GraphSnapshot, subject: str, predicate: str, obj: str) -> tuple[HypothesisStatus, list[str]]:
    """Kanıt grafındaki doğrudan özne→nesne kenarlarından durumu türetir. (durum, kanıt DOI'leri)"""
    supported, refuted = [], []
    for e in snapshot.out_edges(subject, EVIDENCE_PREDICATES):
        if e.dst != obj:
            continue
        stance = e.props.get("stance", "affirmed")
        # MEDIATES/ASSOCIATED hipotezleri için herhangi bir onaylanmış doğrudan kenar desteğe sayılır
        same = e.type == predicate or (predicate in ("ASSOCIATED", "MEDIATES") and e.type != OPPOSITE.get(predicate))
        doi = e.props.get("source_doi")
        if stance == "affirmed" and same:
            supported.append(doi)
        elif stance == "negated" or e.type == OPPOSITE.get(predicate):
            refuted.append(doi)
    if refuted and not supported:
        return HypothesisStatus.tested_refuted, sorted(set(refuted))
    if supported:
        return HypothesisStatus.tested_supported, sorted(set(supported + refuted))
    return HypothesisStatus.untested, []
