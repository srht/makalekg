"""Yükleme: çıkarımları grafa MERGE ile yazar (idempotent) ve Neo4j deposu.

Her çalıştırma bir ingest_run kaydı bırakır (tarih, model, makale sayısı, reddedilen çıkarım sayısı).
"""
from __future__ import annotations

import json
import logging
from itertools import combinations
from typing import Any

from .config import env
from .extract import PaperExtraction
from .graph import (EVIDENCE_EDGE_TYPES, HYPOTHESIS_EDGE, Edge, GraphSnapshot, GraphStore, edge_id)
from .normalize import Normalizer
from .schema import (EVIDENCE_PREDICATES, Claim, ClaimStatus, Design, Entity, EvidenceEdge, Experiment,
                     Hypothesis, IngestRun, Paper, Provenance, experiment_id, normalize_species)

log = logging.getLogger(__name__)

ENTITY_LABELS = ("Gene", "Intervention", "Model", "CellType", "Outcome", "Disease")


# ---------------------------------------------------------------- Neo4j deposu


class Neo4jStore:
    """Neo4j 5 deposu. Tüm düğümler benzersiz `uid` taşır; yazmalar MERGE ile yapılır."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    @classmethod
    def from_env(cls) -> "Neo4jStore":
        return cls(env("NEO4J_URI", "bolt://localhost:7687"), env("NEO4J_USER", "neo4j"),
                   env("NEO4J_PASSWORD", ""), env("NEO4J_DATABASE"))

    def close(self) -> None:
        self.driver.close()

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(cypher, **params)]

    @staticmethod
    def _clean(props: dict[str, Any]) -> dict[str, Any]:
        """Neo4j yalnızca ilkel tipler ve ilkel listeleri saklar."""
        out = {}
        for k, v in props.items():
            if v is None:
                continue
            if isinstance(v, (dict,)) or (isinstance(v, list) and v and isinstance(v[0], (dict, list))):
                out[k + "_json"] = json.dumps(v, ensure_ascii=False, default=str)
            else:
                out[k] = v.value if hasattr(v, "value") else v
        return out

    def setup(self) -> None:
        self._run("CREATE CONSTRAINT node_uid IF NOT EXISTS FOR (n:Node) REQUIRE n.uid IS UNIQUE")
        for label in ("Paper", "Experiment", "Claim", "IngestRun", *ENTITY_LABELS):
            self._run(f"CREATE CONSTRAINT {label.lower()}_uid IF NOT EXISTS FOR (n:{label}) REQUIRE n.uid IS UNIQUE")
        self._run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")

    def _merge_node(self, label: str, uid: str, props: dict[str, Any], extra_label: str = "") -> None:
        self._run(f"MERGE (n:Node {{uid: $uid}}) SET n:{label}{extra_label} SET n += $props",
                  uid=uid, props=self._clean(props))

    def _merge_edge(self, type_: str, src: str, dst: str, eid: str, props: dict[str, Any]) -> str:
        # verified_by_human / needs_review yeniden çalıştırmada ezilmez
        props = self._clean({**props, "edge_id": eid})
        sticky = {k: props.pop(k) for k in ("verified_by_human", "needs_review") if k in props}
        self._run(
            f"MATCH (a:Node {{uid: $src}}), (b:Node {{uid: $dst}}) "
            f"MERGE (a)-[r:{type_} {{edge_id: $eid}}]->(b) SET r += $props "
            "SET r.verified_by_human = coalesce(r.verified_by_human, false) OR $vbh, "
            "    r.needs_review = coalesce(r.needs_review, false) OR $nr",
            src=src, dst=dst, eid=eid, props=props,
            vbh=bool(sticky.get("verified_by_human")), nr=bool(sticky.get("needs_review")))
        return eid

    def upsert_paper(self, p: Paper) -> None:
        self._merge_node("Paper", p.doi, p.model_dump(mode="json"))

    def upsert_experiment(self, e: Experiment) -> None:
        self._merge_node("Experiment", e.id, e.model_dump(mode="json"))
        self._merge_edge("REPORTS", e.paper_doi, e.id, edge_id("REPORTS", e.paper_doi, e.id), {})

    def upsert_claim(self, c: Claim) -> None:
        self._merge_node("Claim", c.id, c.model_dump(mode="json"))

    def upsert_entity(self, e: Entity) -> None:
        self._merge_node(e.type, e.key, {**e.model_dump(mode="json"), "key": e.key}, extra_label=":Entity")

    def add_evidence(self, exp_id: str, claim_key: str, edge: EvidenceEdge) -> str:
        return self._merge_edge(edge.relation, exp_id, claim_key, edge_id(edge.relation, exp_id, claim_key),
                                edge.model_dump(mode="json", exclude={"relation"}))

    def add_claim_role(self, claim_key: str, entity_key: str, role: str) -> None:
        self._merge_edge(role, claim_key, entity_key, edge_id(role, claim_key, entity_key), {})

    def add_relation(self, subj: str, predicate: str, obj: str, claim_key: str, prov: Provenance,
                     extra: dict[str, Any] | None = None) -> str:
        if predicate not in EVIDENCE_PREDICATES:
            raise ValueError(f"geçersiz kanıt yüklemi: {predicate}")
        return self._merge_edge(predicate, subj, obj, edge_id(predicate, subj, obj, claim_key),
                                {**prov.model_dump(mode="json"), **(extra or {}), "claim_id": claim_key})

    def add_cites(self, citing: str, cited: str) -> None:
        self._merge_edge("CITES", citing, cited, edge_id("CITES", citing, cited), {})

    def add_ortholog(self, mouse_key: str, human_key: str, method: str) -> None:
        self._merge_edge("ORTHOLOG_OF", mouse_key, human_key, edge_id("ORTHOLOG_OF", mouse_key, human_key),
                         {"method": method})

    def add_replicates(self, exp_a: str, exp_b: str) -> None:
        self._merge_edge("REPLICATES", exp_a, exp_b, edge_id("REPLICATES", exp_a, exp_b), {})

    def record_ingest_run(self, run: IngestRun) -> None:
        self._merge_node("IngestRun", run.id, run.model_dump(mode="json"))

    def mark_needs_review(self, edge_ids: list[str]) -> int:
        rows = self._run("MATCH ()-[r]->() WHERE r.edge_id IN $ids SET r.needs_review = true RETURN count(r) AS n",
                         ids=edge_ids)
        return rows[0]["n"] if rows else 0

    def mark_verified(self, edge_ids: list[str]) -> int:
        rows = self._run("MATCH ()-[r]->() WHERE r.edge_id IN $ids SET r.verified_by_human = true RETURN count(r) AS n",
                         ids=edge_ids)
        return rows[0]["n"] if rows else 0

    def upsert_hypothesis(self, h: Hypothesis) -> None:
        eid = edge_id(HYPOTHESIS_EDGE, h.subject_key, h.predicate, h.object_key)
        props = h.model_dump(mode="json")
        props["path_text"] = [f"{l.get('from_name')} -{l.get('predicate')}-> {l.get('to_name')}" for l in h.path]
        created = props.pop("created_at")
        self._merge_edge(HYPOTHESIS_EDGE, h.subject_key, h.object_key, eid, props)
        self._run("MATCH ()-[r:HYPOTHESIS {edge_id: $eid}]->() SET r.created_at = coalesce(r.created_at, $c)",
                  eid=eid, c=created)

    def snapshot(self) -> GraphSnapshot:
        nodes: dict[str, dict[str, Any]] = {}
        for row in self._run("MATCH (n:Node) WHERE NOT n:IngestRun "
                             "RETURN n.uid AS uid, labels(n) AS labels, properties(n) AS props"):
            label = next((l for l in row["labels"] if l not in ("Entity", "Node")), row["labels"][0])
            nodes[row["uid"]] = {"label": label, **_unjson(row["props"])}
        edges = [Edge(r["type"], r["src"], r["dst"], _unjson(r["props"])) for r in self._run(
            "MATCH (a:Node)-[r]->(b:Node) "
            "RETURN type(r) AS type, a.uid AS src, b.uid AS dst, properties(r) AS props")]
        return GraphSnapshot(nodes=nodes, edges=edges)


def _unjson(props: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in props.items():
        if k.endswith("_json") and isinstance(v, str):
            out[k[:-5]] = json.loads(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- yükleyici

DESIGN_MAP = {d.value.lower(): d for d in Design}


def _design(value: str | None) -> Design | None:
    return DESIGN_MAP.get((value or "").lower()) if value else None


def relation_stance(polarity: str, evidence_relation: str) -> str:
    """affirmed: iddia edilen ilişki deneyce destekleniyor; negated: ilişki yok/çürütülmüş."""
    return "affirmed" if (polarity == "+") == (evidence_relation == "SUPPORTS") else "negated"


class GraphLoader:
    def __init__(self, store: GraphStore, normalizer: Normalizer):
        self.store = store
        self.normalizer = normalizer

    def load_paper(self, paper: Paper, extraction: PaperExtraction | None, run_id: str) -> int:
        self.store.upsert_paper(paper.model_copy(update={"fulltext": bool(extraction and extraction.fulltext)}))
        if extraction is None or paper.retracted:
            return 0
        n = 0
        for rec in extraction.accepted:
            c = rec.claim
            x = c.experiment
            species = normalize_species(x.species)
            exp = Experiment(id=experiment_id(paper.doi, x.figure, x.panel), paper_doi=paper.doi, figure=x.figure,
                             panel=x.panel, design=_design(x.design), species=species, sex=x.sex,
                             n_per_group=x.n_per_group, blinded=x.blinded, randomized=x.randomized,
                             tracer_type=x.tracer_type, model_name=x.model_name)
            claim = Claim(id=rec.claim_id, text=c.claim, polarity=c.polarity, status=ClaimStatus.evidence,
                          predicate=c.predicate)
            subj = self.normalizer.normalize(c.subject, default_species=species)
            obj = self.normalizer.normalize(c.object, default_species=species)
            figure = " ".join(p for p in (x.figure, x.panel) if p) or None
            prov = dict(source_doi=paper.doi, figure=figure, source_sentence=c.source_sentence,
                        extraction_model=extraction.model, extraction_confidence=c.extraction_confidence,
                        verification=rec.verification, ingest_run_id=run_id)

            self.store.upsert_experiment(exp)
            self.store.upsert_claim(claim)
            self.store.upsert_entity(subj)
            self.store.upsert_entity(obj)
            self.normalizer.register_known([subj, obj])
            self.store.add_evidence(exp.id, claim.id, EvidenceEdge(relation=c.evidence_relation, p_value=c.p_value,
                                                                   effect_size=c.effect_size, **prov))
            self.store.add_claim_role(claim.id, subj.key, "SUBJECT")
            self.store.add_claim_role(claim.id, obj.key, "OBJECT")
            if c.predicate:
                # Kenara tür/tasarım bağlamı da yazılır: zincir cezaları (tür farkı vb.) için gerekir
                extra = {"stance": relation_stance(c.polarity, c.evidence_relation), "species": exp.species,
                         "design": exp.design.value if exp.design else None, "experiment_id": exp.id,
                         "dose_known": (bool(c.subject.dose or c.subject.schedule)
                                        if c.subject.type == "Intervention" else None)}
                self.store.add_relation(subj.key, c.predicate, obj.key, claim.id, Provenance(**prov),
                                        {k: v for k, v in extra.items() if v is not None})
            n += 1
        return n

    def link_orthologs(self) -> int:
        snap = self.store.snapshot()
        genes = [Entity.model_validate({k: v for k, v in n.items() if k not in ("label", "key")})
                 for n in snap.nodes.values() if n.get("label") == "Gene"]
        pairs = self.normalizer.orthologs(genes)
        for m, h, method in pairs:
            self.store.add_ortholog(m, h, method)
        return len(pairs)

    def link_replications(self) -> int:
        """Aynı Claim'i destekleyen, FARKLI laboratuvarlardan deneyler arasında REPLICATES."""
        snap = self.store.snapshot()
        n = 0
        for key, node in snap.nodes.items():
            if node.get("label") != "Claim":
                continue
            supporting = [(e, exp, paper) for e, exp, paper in snap.evidence_for_claim(key) if e.type == "SUPPORTS"]
            for (e1, x1, p1), (e2, x2, p2) in combinations(supporting, 2):
                lab1, lab2 = (p1.get("lab") or "").lower(), (p2.get("lab") or "").lower()
                if not lab1 or not lab2 or lab1 == lab2 or p1.get("doi") == p2.get("doi"):
                    continue
                later, earlier = (e1.src, e2.src) if (p1.get("year") or 0) >= (p2.get("year") or 0) else (e2.src, e1.src)
                self.store.add_replicates(later, earlier)
                n += 1
        return n


def evidence_query_returns_hypotheses(snapshot: GraphSnapshot) -> bool:
    """Faz 3 kabul kriteri: kanıt sorgusu hiçbir HYPOTHESIS kenarı döndürmemeli."""
    rels = snapshot.evidence_relations()
    ev = [e for e in snapshot.edges if e.type in EVIDENCE_EDGE_TYPES]
    return any(e.type == HYPOTHESIS_EDGE for e in rels + ev)
