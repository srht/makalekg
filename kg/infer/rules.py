"""Kural tabanlı zincirler: rules.yaml'daki kalıplara uyan 2–3 adımlı yollar;
A ile C arasında doğrudan kanıt kenarı yoksa aday üretilir.

Yalnızca kanıt kenarları (stance=affirmed) gezilir; HYPOTHESIS kenarları asla yol halkası olamaz.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from ..config import rules_config
from ..graph import Edge, GraphSnapshot
from ..schema import EVIDENCE_PREDICATES
from ..score import Link, relation_score


@dataclass
class Candidate:
    subject: str
    predicate: str
    object: str
    links: list[Link]
    generators: list[str]
    rationale: str
    embed_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


def _type_ok(snapshot: GraphSnapshot, key: str, wanted: str) -> bool:
    return wanted == "*" or snapshot.label(key) == wanted


def best_links(snapshot: GraphSnapshot, cfg: dict[str, Any] | None = None) -> dict[tuple[str, str, str], Link]:
    """Her (src, yüklem, dst) için en güçlü (en yüksek E) onaylanmış kanıt kenarı."""
    best: dict[tuple[str, str, str], Link] = {}
    for e in snapshot.evidence_relations():
        if e.props.get("stance", "affirmed") != "affirmed" or e.props.get("needs_review"):
            continue
        s = relation_score(snapshot, e, cfg)
        k = (e.src, e.type, e.dst)
        if k not in best or s.E > best[k].score.E:
            best[k] = Link(e.src, e.type, e.dst, e, s)
    return best


def has_direct_evidence(snapshot: GraphSnapshot, a: str, c: str) -> bool:
    """A -[*]-> C kanıt kenarı (herhangi bir kanıt yüklemi, herhangi bir stance) var mı?"""
    return any(e.dst == c for e in snapshot.out_edges(a, EVIDENCE_PREDICATES))


def _steps(links_by_src: dict[str, list[Link]], links_by_dst: dict[str, list[Link]], node: str,
           pred: str) -> Iterator[tuple[str, Link]]:
    if pred.endswith("_INV"):
        base = pred[:-4]
        for l in links_by_dst.get(node, []):
            if base == "*" or l.predicate == base:
                yield l.src, Link(l.dst, l.predicate, l.src, l.edge, l.score, inverse=True)
    else:
        for l in links_by_src.get(node, []):
            if pred == "*" or l.predicate == pred:
                yield l.dst, l


def rule_candidates(snapshot: GraphSnapshot, rules: dict[str, Any] | None = None,
                    scoring: dict[str, Any] | None = None) -> list[Candidate]:
    rules = rules or rules_config()
    links = best_links(snapshot, scoring)
    by_src: dict[str, list[Link]] = {}
    by_dst: dict[str, list[Link]] = {}
    for l in links.values():
        by_src.setdefault(l.src, []).append(l)
        by_dst.setdefault(l.dst, []).append(l)

    out: dict[tuple[str, str, str], Candidate] = {}
    for pattern in rules.get("patterns", []):
        types, preds = pattern["node_types"], pattern["predicates"]
        if len(types) != len(preds) + 1 or len(preds) > rules.get("max_path_length", 3):
            continue
        for start in snapshot.entities():
            if not _type_ok(snapshot, start, types[0]):
                continue
            paths: list[tuple[list[str], list[Link]]] = [([start], [])]
            for i, pred in enumerate(preds):
                nxt = []
                for nodes, chain in paths:
                    for node, link in _steps(by_src, by_dst, nodes[-1], pred):
                        if node in nodes or not _type_ok(snapshot, node, types[i + 1]):
                            continue
                        nxt.append((nodes + [node], chain + [link]))
                paths = nxt
            for nodes, chain in paths:
                a, c = nodes[0], nodes[-1]
                if has_direct_evidence(snapshot, a, c):
                    continue
                pred = pattern.get("hypothesis_predicate", "ASSOCIATED")
                key = (a, pred, c)
                rationale = (f"Kalıp '{pattern['name']}' ({pattern.get('description', '')}): " +
                             " ; ".join(f"{snapshot.name(l.src)} -{l.predicate}{'(ters)' if l.inverse else ''}-> "
                                        f"{snapshot.name(l.dst)}" for l in chain) +
                             f". {snapshot.name(a)} ile {snapshot.name(c)} arasında doğrudan kanıt kenarı yok.")
                cand = Candidate(a, pred, c, chain, [f"rules:{pattern['name']}"], rationale)
                prev = out.get(key)
                if prev is None or min(l.score.E for l in chain) > min(l.score.E for l in prev.links):
                    if prev:
                        cand.generators = sorted(set(prev.generators + cand.generators))
                    out[key] = cand
                else:
                    prev.generators = sorted(set(prev.generators + cand.generators))
    return list(out.values())


def shortest_evidence_path(snapshot: GraphSnapshot, a: str, c: str, max_len: int = 3,
                           links: dict[tuple[str, str, str], Link] | None = None) -> list[Link]:
    """Yönsüz BFS ile A–C arası en kısa kanıt yolu (embedding/ajan adayları için yol üretimi)."""
    links = links if links is not None else best_links(snapshot)
    adj: dict[str, list[tuple[str, Link]]] = {}
    for l in links.values():
        adj.setdefault(l.src, []).append((l.dst, l))
        adj.setdefault(l.dst, []).append((l.src, Link(l.dst, l.predicate, l.src, l.edge, l.score, inverse=True)))
    frontier = [(a, [])]
    seen = {a}
    for _ in range(max_len):
        nxt = []
        for node, path in frontier:
            for nb, link in adj.get(node, []):
                if nb == c:
                    return path + [link]
                if nb not in seen:
                    seen.add(nb)
                    nxt.append((nb, path + [link]))
        frontier = nxt
    return []


def validate_path(snapshot: GraphSnapshot, path: list[dict[str, Any]],
                  links: dict[tuple[str, str, str], Link] | None = None) -> list[Link] | None:
    """Ajanın önerdiği yolun her halkası gerçek bir kanıt kenarı olmalı; aksi halde None.
    HYPOTHESIS kenarları kanıt kenarı olmadığı için burada asla geçmez."""
    links = links if links is not None else best_links(snapshot)
    out = []
    for step in path:
        k = (step.get("from"), step.get("predicate"), step.get("to"))
        if k in links:
            out.append(links[k])
            continue
        kr = (step.get("to"), step.get("predicate"), step.get("from"))
        if kr in links:
            l = links[kr]
            out.append(Link(l.dst, l.predicate, l.src, l.edge, l.score, inverse=True))
            continue
        return None
    return out or None


def as_edge(link: Link) -> Edge:
    return link.edge
