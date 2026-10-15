"""LLM ajanı: grafı SALT-OKUNUR araçlarla gezer.

- `ask`: uzman olmayan kullanıcının sorusunu, kanıt düzeyini düz Türkçe söyleyerek yanıtlar.
- `propose`: periyodik görev / soru ile tetiklenir, her adayı yol (path) ve gerekçesiyle döner.

Yasaklar: ajanın grafa yazma yetkisi yoktur (yazma HypothesisWriter'dan geçer ve ajan ona erişemez);
önerdiği her yol halkası gerçek bir kanıt kenarına karşı doğrulanır (kendi hipotezini kanıt olarak
kullanamaz); insan klinik önerisi üretmez.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from ..config import env
from ..graph import GraphSnapshot
from ..llm import LLM, Tool
from ..schema import EVIDENCE_PREDICATES
from ..score import describe_experiment, independent_replications
from .rules import Candidate, best_links, has_direct_evidence, validate_path

log = logging.getLogger(__name__)

WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD\s+CSV|FOREACH|GRANT|DENY|REVOKE)\b|"
    r"\bCALL\s+(dbms|db\.(create|drop)|apoc\.(create|merge|refactor|periodic|do|cypher\.(run)?write))", re.I)


class ReadOnlyCypher:
    """Neo4j'e salt-okunur bağlantı: READ erişim modu + yazma anahtar kelimesi filtresi
    (+ .env'de NEO4J_READ_USER tanımlıysa reader rolündeki kullanıcı)."""

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        from neo4j import READ_ACCESS, GraphDatabase

        self._mode = READ_ACCESS
        self.driver = GraphDatabase.driver(
            uri or env("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or env("NEO4J_READ_USER") or env("NEO4J_USER", "neo4j"),
                  password or env("NEO4J_READ_PASSWORD") or env("NEO4J_PASSWORD", "")))

    def query(self, cypher: str, limit: int = 50) -> list[dict[str, Any]]:
        if WRITE_PATTERN.search(cypher):
            raise PermissionError("ajan yalnızca okuma sorgusu çalıştırabilir")
        with self.driver.session(default_access_mode=self._mode) as s:
            return [r.data() for r in s.run(cypher)][:limit]


class AgentLink(BaseModel):
    from_: str = Field(alias="from")
    predicate: str
    to: str

    model_config = {"populate_by_name": True}


class AgentCandidate(BaseModel):
    subject: str = Field(description="Varlık anahtarı (key), ör. Gene:mouse:MGI:...")
    predicate: str = Field(description="INCREASES | DECREASES | MEDIATES | PROTECTS | REQUIRES")
    object: str
    path: list[AgentLink]
    rationale: str = Field(description="Düz Türkçe gerekçe")


class AgentCandidates(BaseModel):
    candidates: list[AgentCandidate]


ASK_SYSTEM = """Sen kemik–beyin ekseni knowledge graph'ı üzerinde çalışan bir araştırma asistanısın.
Kullanıcı uzman değildir. Kurallar:
- Yalnızca araçlardan gelen kanıta dayan; grafta olmayanı söyleme, sayı uydurma.
- Her bulgu için kanıt düzeyini düz Türkçe söyle (ör. "sadece farede, tek laboratuvar, körlemesiz, n = 9–10").
- Kanıt (makalelerden) ile hipotezi (sistemin tahmini) ASLA karıştırma; hipotezden söz edersen açıkça "hipotez, test edilmemiş" de.
- Kaynak olarak makale DOI'si ve şekil numarasını ver.
- İnsan için klinik/tedavi önerisi verme; soru buna yönelikse bunun bir araştırma grafiği olduğunu söyle.
- Yanıtı Türkçe yaz."""

PROPOSE_SYSTEM = """Sen kemik–beyin ekseni knowledge graph'ında hipotez arayan bir ajansın.
Grafı araçlarla gez. Her aday hipotez için grafta GERÇEKTEN var olan kanıt kenarlarından oluşan bir yol ver
(her halka: from anahtarı, yüklem, to anahtarı — araçların döndürdüğü anahtarları birebir kullan).
Doğrudan kanıtı zaten olan ilişkileri önerme. Hipotezleri kanıt olarak kullanma. Klinik öneri üretme.
En fazla 10 aday öner; bitirdiğinde adayları açıkça listele."""


class GraphAgent:
    def __init__(self, llm: LLM, snapshot: GraphSnapshot, cypher: ReadOnlyCypher | None = None):
        self.llm = llm
        self.snap = snapshot
        self.cypher = cypher
        self._links = best_links(snapshot)
        self._names = {k: f"{v.get('symbol') or v.get('name') or k} [{v.get('label')}]"
                       for k, v in snapshot.entities().items()}

    # ------------------------------------------------------------ araçlar (salt-okunur)

    def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        hits = process.extract(query, self._names, scorer=fuzz.WRatio, limit=limit)
        return [{"key": key, "label": self.snap.label(key), "name": self.snap.name(key),
                 "species": self.snap.nodes[key].get("species"), "match": round(score)} for _, score, key in hits]

    def entity_relations(self, key: str) -> list[dict[str, Any]]:
        out = []
        for e in self.snap.out_edges(key, EVIDENCE_PREDICATES) + self.snap.in_edges(key, EVIDENCE_PREDICATES):
            out.append({"from": e.src, "from_name": self.snap.name(e.src), "predicate": e.type,
                        "to": e.dst, "to_name": self.snap.name(e.dst), "stance": e.props.get("stance", "affirmed"),
                        "claim_id": e.props.get("claim_id"), "source_doi": e.props.get("source_doi"),
                        "figure": e.props.get("figure")})
        return out[:60]

    def claim_evidence(self, claim_id: str) -> dict[str, Any]:
        claim = self.snap.nodes.get(claim_id, {})
        ev = []
        for e, exp, paper in self.snap.evidence_for_claim(claim_id):
            ev.append({"relation": e.type, "paper": paper.get("doi"), "year": paper.get("year"),
                       "lab": paper.get("lab"), "funding": paper.get("funding_source"),
                       "retracted": paper.get("retracted"), "p_value": e.props.get("p_value"),
                       "source_sentence": e.props.get("source_sentence"),
                       "evidence_level": describe_experiment(exp, paper, independent_replications(self.snap, e.src))})
        return {"claim": claim.get("text"), "status": claim.get("status"), "evidence": ev}

    def find_paths(self, from_key: str, to_key: str, max_len: int = 3) -> list[list[dict[str, Any]]]:
        adj: dict[str, list[tuple[str, tuple[str, str, str]]]] = {}
        for (s, p, d) in self._links:
            adj.setdefault(s, []).append((d, (s, p, d)))
        paths: list[list[dict[str, Any]]] = []
        stack = [(from_key, [], {from_key})]
        while stack and len(paths) < 10:
            node, path, seen = stack.pop()
            if len(path) >= max_len:
                continue
            for nb, k in adj.get(node, []):
                if nb in seen:
                    continue
                step = path + [{"from": k[0], "predicate": k[1], "to": k[2], "E": self._links[k].score.E}]
                if nb == to_key:
                    paths.append(step)
                else:
                    stack.append((nb, step, seen | {nb}))
        return paths

    def least_tested_strongly_connected(self, limit: int = 10) -> list[dict[str, Any]]:
        """Periyodik görev için: çok bağlantılı ama az sayıda bağımsız kanıtı olan moleküller."""
        rows = []
        for key, node in self.snap.entities().items():
            if node.get("label") != "Gene":
                continue
            rels = self.snap.out_edges(key, EVIDENCE_PREDICATES) + self.snap.in_edges(key, EVIDENCE_PREDICATES)
            if not rels:
                continue
            papers = {e.props.get("source_doi") for e in rels}
            rows.append({"key": key, "name": self.snap.name(key), "degree": len(rels), "n_papers": len(papers)})
        rows.sort(key=lambda r: (-r["degree"], r["n_papers"]))
        return rows[:limit]

    def tools(self) -> list[Tool]:
        s = {"type": "object", "additionalProperties": False}
        tools = [
            Tool("search_entities", "Grafta varlık ara (gen, müdahale, sonuç, model, hücre tipi, hastalık).",
                 {**s, "properties": {"query": {"type": "string"}}, "required": ["query"]}, self.search_entities),
            Tool("entity_relations", "Bir varlığın kanıt kenarları (yalnızca makalelerden çıkarılmış ilişkiler).",
                 {**s, "properties": {"key": {"type": "string"}}, "required": ["key"]}, self.entity_relations),
            Tool("claim_evidence", "Bir iddianın deneyleri, makaleleri ve düz Türkçe kanıt düzeyi.",
                 {**s, "properties": {"claim_id": {"type": "string"}}, "required": ["claim_id"]}, self.claim_evidence),
            Tool("find_paths", "İki varlık arasında en fazla 3 adımlı kanıt yolları.",
                 {**s, "properties": {"from_key": {"type": "string"}, "to_key": {"type": "string"}},
                  "required": ["from_key", "to_key"]}, self.find_paths),
            Tool("least_tested_strongly_connected", "En çok bağlantılı ama en az test edilmiş moleküller.",
                 {**s, "properties": {}, "required": []}, lambda: self.least_tested_strongly_connected()),
        ]
        if self.cypher is not None:
            tools.append(Tool("run_cypher", "Salt-okunur Cypher sorgusu (yazma komutları reddedilir). "
                              "Kanıt kenarları: SUPPORTS, CONTRADICTS, INCREASES, DECREASES, REQUIRES, MEDIATES, PROTECTS. "
                              "HYPOTHESIS kenarları kanıt DEĞİLDİR.",
                              {**s, "properties": {"cypher": {"type": "string"}}, "required": ["cypher"]},
                              lambda cypher: self.cypher.query(cypher)))
        return tools

    # ------------------------------------------------------------ görevler

    def ask(self, question: str) -> str:
        return self.llm.run_tools(ASK_SYSTEM, question, self.tools())

    def propose(self, task: str = "En az test edilmiş ama en güçlü bağlı molekülleri bul ve hipotez öner.") -> list[Candidate]:
        transcript = self.llm.run_tools(PROPOSE_SYSTEM, task, self.tools())
        structured = self.llm.complete_json(
            "Aşağıdaki ajan çıktısındaki aday hipotezleri şemaya dönüştür. Yalnızca metinde geçen adayları ve "
            "anahtarları birebir kullan; yeni aday ekleme.", transcript or "(boş)", AgentCandidates)
        out = []
        for c in structured.candidates:
            if c.subject not in self.snap.nodes or c.object not in self.snap.nodes:
                log.info("Ajan adayı reddedildi (bilinmeyen anahtar): %s → %s", c.subject, c.object)
                continue
            if has_direct_evidence(self.snap, c.subject, c.object):
                continue
            links = validate_path(self.snap, [{"from": l.from_, "predicate": l.predicate, "to": l.to} for l in c.path],
                                  self._links)
            if not links:
                log.info("Ajan adayı reddedildi (yol kanıt kenarlarıyla doğrulanamadı): %s", c)
                continue
            out.append(Candidate(c.subject, c.predicate, c.object, links, ["agent"], c.rationale))
        return out


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, default=str)
