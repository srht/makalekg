"""Graf deposu arayüzü, salt-okunur anlık görüntü (snapshot) ve bellek-içi depo.

Neo4j deposu kg/load.py içindedir; ikisi aynı yazma arayüzünü uygular. Bellek-içi depo
(data/graph.json) Neo4j olmadan geliştirme, test ve backtest için kullanılır.

Ayrım: kanıt yazma metotları (add_evidence, add_relation, ...) ve hipotez yazma metodu
(upsert_hypothesis) ayrıdır; hipotezleri yalnızca kg.infer.writer.HypothesisWriter çağırır.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from .config import DATA_DIR
from .schema import (EVIDENCE_PREDICATES, Claim, Entity, EvidenceEdge, Experiment, Hypothesis, IngestRun,
                     Paper, Provenance)

EVIDENCE_EDGE_TYPES = ("SUPPORTS", "CONTRADICTS")
STRUCTURAL_EDGE_TYPES = ("REPORTS", "SUBJECT", "OBJECT", "CITES", "ORTHOLOG_OF", "REPLICATES")
HYPOTHESIS_EDGE = "HYPOTHESIS"


def edge_id(*parts: str) -> str:
    return "e_" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


@dataclass
class Edge:
    type: str
    src: str
    dst: str
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.props.get("edge_id") or edge_id(self.type, self.src, self.dst, str(self.props.get("claim_id", "")))


@dataclass
class GraphSnapshot:
    """Salt-okunur görüntü. Çıkarım, skor ve doğrulama katmanları yalnızca bunu okur."""
    nodes: dict[str, dict[str, Any]]  # key → {"label": ..., ...props}
    edges: list[Edge]

    def __post_init__(self) -> None:
        self._out: dict[str, list[Edge]] = defaultdict(list)
        self._in: dict[str, list[Edge]] = defaultdict(list)
        for e in self.edges:
            self._out[e.src].append(e)
            self._in[e.dst].append(e)

    # --- erişim
    def label(self, key: str) -> str | None:
        n = self.nodes.get(key)
        return n.get("label") if n else None

    def out_edges(self, key: str, types: Iterable[str] | None = None) -> list[Edge]:
        t = set(types) if types else None
        return [e for e in self._out.get(key, []) if t is None or e.type in t]

    def in_edges(self, key: str, types: Iterable[str] | None = None) -> list[Edge]:
        t = set(types) if types else None
        return [e for e in self._in.get(key, []) if t is None or e.type in t]

    def entities(self) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self.nodes.items() if v.get("label") in
                ("Gene", "Intervention", "Model", "CellType", "Outcome", "Disease")}

    def evidence_relations(self) -> list[Edge]:
        """Varlık → varlık kanıt kenarları. HYPOTHESIS ASLA dahil değildir."""
        return [e for e in self.edges if e.type in EVIDENCE_PREDICATES]

    def hypotheses(self) -> list[Edge]:
        return [e for e in self.edges if e.type == HYPOTHESIS_EDGE]

    def evidence_for_claim(self, claim_key: str) -> list[tuple[Edge, dict[str, Any], dict[str, Any]]]:
        """(SUPPORTS/CONTRADICTS kenarı, experiment, paper) üçlüleri."""
        out = []
        for e in self.in_edges(claim_key, EVIDENCE_EDGE_TYPES):
            exp = self.nodes.get(e.src, {})
            paper = {}
            for r in self.in_edges(e.src, ["REPORTS"]):
                paper = self.nodes.get(r.src, {})
            out.append((e, exp, paper))
        return out

    def name(self, key: str) -> str:
        n = self.nodes.get(key, {})
        return n.get("symbol") or n.get("name") or n.get("text") or key

    def filtered(self, keep_dois: set[str]) -> "GraphSnapshot":
        """Yalnızca verilen makalelerden gelen kanıtla yeni bir görüntü (backtest için)."""
        keep_exps = {e.dst for e in self.edges if e.type == "REPORTS" and e.src in keep_dois}
        keep_claims = {e.dst for e in self.edges if e.type in EVIDENCE_EDGE_TYPES and e.src in keep_exps}
        edges = []
        for e in self.edges:
            if e.type == HYPOTHESIS_EDGE:
                continue
            if e.type == "REPORTS" and e.src not in keep_dois:
                continue
            if e.type in EVIDENCE_EDGE_TYPES and e.src not in keep_exps:
                continue
            if e.type in ("SUBJECT", "OBJECT") and e.src not in keep_claims:
                continue
            if e.type in EVIDENCE_PREDICATES and e.props.get("claim_id") not in keep_claims:
                continue
            if e.type == "CITES" and (e.src not in keep_dois or e.dst not in keep_dois):
                continue
            if e.type == "REPLICATES" and (e.src not in keep_exps or e.dst not in keep_exps):
                continue
            edges.append(e)
        used = {e.src for e in edges} | {e.dst for e in edges}
        nodes = {k: v for k, v in self.nodes.items()
                 if (v.get("label") == "Paper" and k in keep_dois) or
                 (v.get("label") not in ("Paper",) and k in used)}
        return GraphSnapshot(nodes=nodes, edges=edges)


class GraphStore(Protocol):
    def setup(self) -> None: ...
    def upsert_paper(self, p: Paper) -> None: ...
    def upsert_experiment(self, e: Experiment) -> None: ...
    def upsert_claim(self, c: Claim) -> None: ...
    def upsert_entity(self, e: Entity) -> None: ...
    def add_evidence(self, exp_id: str, claim_key: str, edge: EvidenceEdge) -> str: ...
    def add_claim_role(self, claim_key: str, entity_key: str, role: str) -> None: ...
    def add_relation(self, subj: str, predicate: str, obj: str, claim_key: str, prov: Provenance,
                     extra: dict[str, Any] | None = None) -> str: ...
    def add_cites(self, citing: str, cited: str) -> None: ...
    def add_ortholog(self, mouse_key: str, human_key: str, method: str) -> None: ...
    def add_replicates(self, exp_a: str, exp_b: str) -> None: ...
    def record_ingest_run(self, run: IngestRun) -> None: ...
    def mark_needs_review(self, edge_ids: list[str]) -> int: ...
    def mark_verified(self, edge_ids: list[str]) -> int: ...
    def upsert_hypothesis(self, h: Hypothesis) -> None: ...
    def snapshot(self) -> GraphSnapshot: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------- bellek-içi depo


class MemoryStore:
    """JSON dosyasına kalıcılaşan, MERGE semantiğine sahip basit depo."""

    def __init__(self, path: Path | None = None, persist: bool = True):
        self.path = path or DATA_DIR / "graph.json"
        self.persist = persist
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, Edge] = {}
        self.runs: list[dict[str, Any]] = []
        if persist and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.nodes = data.get("nodes", {})
            self.edges = {k: Edge(**v) for k, v in data.get("edges", {}).items()}
            self.runs = data.get("ingest_runs", [])

    def setup(self) -> None:
        pass

    def flush(self) -> None:
        if not self.persist:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"nodes": self.nodes,
                "edges": {k: {"type": e.type, "src": e.src, "dst": e.dst, "props": e.props} for k, e in self.edges.items()},
                "ingest_runs": self.runs}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def close(self) -> None:
        self.flush()

    def _merge_node(self, key: str, label: str, props: dict[str, Any]) -> None:
        node = self.nodes.setdefault(key, {"label": label})
        node.update({k: v for k, v in props.items() if v is not None})

    def _merge_edge(self, eid: str, type_: str, src: str, dst: str, props: dict[str, Any]) -> str:
        props = {**props, "edge_id": eid}
        if eid in self.edges:
            existing = self.edges[eid].props
            # insan doğrulaması ve inceleme bayrakları yeniden çalıştırmada korunur
            for keep in ("verified_by_human", "needs_review"):
                if existing.get(keep):
                    props[keep] = existing[keep]
            existing.update(props)
        else:
            self.edges[eid] = Edge(type_, src, dst, props)
        return eid

    def upsert_paper(self, p: Paper) -> None:
        self._merge_node(p.doi, "Paper", p.model_dump(mode="json"))

    def upsert_experiment(self, e: Experiment) -> None:
        self._merge_node(e.id, "Experiment", e.model_dump(mode="json"))
        self._merge_edge(edge_id("REPORTS", e.paper_doi, e.id), "REPORTS", e.paper_doi, e.id, {})

    def upsert_claim(self, c: Claim) -> None:
        self._merge_node(c.id, "Claim", c.model_dump(mode="json"))

    def upsert_entity(self, e: Entity) -> None:
        self._merge_node(e.key, e.type, {**e.model_dump(mode="json"), "key": e.key})

    def add_evidence(self, exp_id: str, claim_key: str, edge: EvidenceEdge) -> str:
        eid = edge_id(edge.relation, exp_id, claim_key)
        return self._merge_edge(eid, edge.relation, exp_id, claim_key, edge.model_dump(mode="json", exclude={"relation"}))

    def add_claim_role(self, claim_key: str, entity_key: str, role: str) -> None:
        self._merge_edge(edge_id(role, claim_key, entity_key), role, claim_key, entity_key, {})

    def add_relation(self, subj: str, predicate: str, obj: str, claim_key: str, prov: Provenance,
                     extra: dict[str, Any] | None = None) -> str:
        if predicate not in EVIDENCE_PREDICATES:
            raise ValueError(f"geçersiz kanıt yüklemi: {predicate}")
        eid = edge_id(predicate, subj, obj, claim_key)
        props = {**prov.model_dump(mode="json"), **(extra or {}), "claim_id": claim_key}
        return self._merge_edge(eid, predicate, subj, obj, props)

    def add_cites(self, citing: str, cited: str) -> None:
        self._merge_edge(edge_id("CITES", citing, cited), "CITES", citing, cited, {})

    def add_ortholog(self, mouse_key: str, human_key: str, method: str) -> None:
        self._merge_edge(edge_id("ORTHOLOG_OF", mouse_key, human_key), "ORTHOLOG_OF", mouse_key, human_key,
                         {"method": method})

    def add_replicates(self, exp_a: str, exp_b: str) -> None:
        self._merge_edge(edge_id("REPLICATES", exp_a, exp_b), "REPLICATES", exp_a, exp_b, {})

    def record_ingest_run(self, run: IngestRun) -> None:
        self.runs = [r for r in self.runs if r.get("id") != run.id] + [run.model_dump(mode="json")]
        self.nodes[f"ingest_run:{run.id}"] = {"label": "IngestRun", **run.model_dump(mode="json")}

    def _set_flag(self, edge_ids: list[str], flag: str) -> int:
        n = 0
        for eid in edge_ids:
            if eid in self.edges:
                self.edges[eid].props[flag] = True
                n += 1
        return n

    def mark_needs_review(self, edge_ids: list[str]) -> int:
        return self._set_flag(edge_ids, "needs_review")

    def mark_verified(self, edge_ids: list[str]) -> int:
        return self._set_flag(edge_ids, "verified_by_human")

    def upsert_hypothesis(self, h: Hypothesis) -> None:
        eid = edge_id(HYPOTHESIS_EDGE, h.subject_key, h.predicate, h.object_key)
        props = h.model_dump(mode="json")
        if eid in self.edges:  # tekilleştirme: aynı üçlü → skor/durum güncellenir, created_at korunur
            props["created_at"] = self.edges[eid].props.get("created_at", props["created_at"])
        self._merge_edge(eid, HYPOTHESIS_EDGE, h.subject_key, h.object_key, props)

    def snapshot(self) -> GraphSnapshot:
        nodes = {k: v for k, v in self.nodes.items() if v.get("label") != "IngestRun"}
        return GraphSnapshot(nodes=json.loads(json.dumps(nodes)),
                             edges=[Edge(e.type, e.src, e.dst, dict(e.props)) for e in self.edges.values()])


def get_store(backend: str | None = None) -> GraphStore:
    from .config import env

    backend = (backend or env("GRAPH_BACKEND", "neo4j")).lower()
    if backend == "memory":
        return MemoryStore()
    if backend == "neo4j":
        from .load import Neo4jStore
        return Neo4jStore.from_env()
    raise ValueError(f"bilinmeyen GRAPH_BACKEND: {backend}")
