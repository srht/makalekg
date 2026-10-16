"""Ontoloji tabanlı varlık ve ilişki çıkarımı.

Her belge parçası için LLM'e ontoloji + metin verilir; yanıt ontolojiden üretilen şemaya göre doğrulanır.
Sonra deterministik kontroller yapılır ve geçmeyen her öğe sebebiyle "reddedilen" listesine yazılır:
  - varlığın metindeki yazılışı (mention) metinde gerçekten geçmeli,
  - ilişkinin kaynak cümlesi metinde birebir geçmeli,
  - ilişkinin özne/nesnesi çıkarılan varlıklar arasında olmalı ve ontolojideki tür kısıtına uymalı.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ..llm import LLM, LLMError, is_manual
from .documents import chunk_text
from .ontology import Ontology, SuggestedOntology
from .store import Project, norm_name

log = logging.getLogger(__name__)

API_CHUNK = 15000
MANUAL_CHUNK = 120000  # kopyala–yapıştır modunda istem sayısını azaltmak için büyük parçalar

EXTRACT_SYSTEM = """You build a knowledge graph from documents using a FIXED ontology.

Rules:
- Extract only entities whose type is in the ontology, and only relations whose type is in the ontology.
- Extract only what the text explicitly states. Do not use outside knowledge, do not guess.
- entity.mention: the entity exactly as it appears in the text (verbatim, same spelling).
- entity.name: a canonical name (e.g. full name instead of a pronoun or abbreviation, if the text gives it).
  Use the SAME name every time you refer to the same entity.
- relation.subject / relation.object: must be the `name` of an entity in your entities list.
- Respect the allowed subject/object types of each relation type.
- relation.source_sentence: copy the sentence that states the relation VERBATIM from the text (max 400 characters).
- confidence: lower it when the statement is hedged or indirect.
- descriptions in Turkish, one short sentence; null if the text gives nothing.
- If nothing matches the ontology, return empty lists."""

SUGGEST_SYSTEM = """You design ontologies for knowledge graphs. Read the document excerpt and propose an ontology
for building a knowledge graph from documents like this one.
- 5 to 10 entity types, 5 to 12 relation types. Names short and in Turkish (relation names UPPER_SNAKE_CASE, Turkish).
- Each type gets a one-sentence Turkish description.
- For each relation give the allowed subject types (source) and object types (target) using your entity type names;
  empty list means any type.
- Prefer types that occur many times in such documents over rare ones.
- domain: one short Turkish phrase naming the field."""


def _ws(s: str) -> str:
    s = s.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().casefold()


def _type_ok(allowed: list[str], etype: str) -> bool:
    return not allowed or etype in allowed


def extract_chunk(llm: LLM, onto: Ontology, title: str, text: str) -> tuple[list[dict], list[dict], list[dict]]:
    schema = onto.extraction_schema()
    domain = f"Domain: {onto.domain}\n" if onto.domain else ""
    user = f"{domain}{onto.prompt_text()}\n\n=== DOCUMENT: {title} ===\n{text}"
    result = llm.complete_json(EXTRACT_SYSTEM, user, schema)
    corpus = _ws(text)
    entities, relations, rejected = [], [], []
    by_name: dict[str, dict[str, Any]] = {}
    for e in result.entities:
        item = {"type": e.type, "name": e.name.strip(), "mention": e.mention.strip(), "description": e.description}
        if not item["name"]:
            continue
        if _ws(item["mention"]) not in corpus and _ws(item["name"]) not in corpus:
            rejected.append({"kind": "varlık", "item": f"{item['name']} ({item['type']})",
                             "reason": "metinde geçmiyor", "text": item["mention"]})
            continue
        entities.append(item)
        by_name.setdefault(norm_name(item["name"]), item)
    for r in result.relations:
        desc = f"{r.subject} —{r.predicate}→ {r.object}"
        sentence = r.source_sentence.strip()[:400]
        s, o = by_name.get(norm_name(r.subject)), by_name.get(norm_name(r.object))
        reason = None
        if s is None or o is None:
            reason = "özne veya nesne çıkarılan varlıklar arasında yok"
        elif _ws(sentence) not in corpus:
            reason = "kaynak cümle metinde birebir geçmiyor"
        else:
            rt = onto.relation(r.predicate)
            if rt is None:
                reason = "ontolojide olmayan ilişki türü"
            elif not _type_ok(rt.source, s["type"]) or not _type_ok(rt.target, o["type"]):
                reason = f"tür kısıtına uymuyor ({s['type']} → {o['type']})"
        if reason:
            rejected.append({"kind": "ilişki", "item": desc, "reason": reason, "text": sentence})
            continue
        relations.append({"subject_key": Project.entity_key(s["type"], s["name"]), "predicate": r.predicate,
                          "object_key": Project.entity_key(o["type"], o["name"]), "sentence": sentence,
                          "confidence": max(0.0, min(1.0, float(r.confidence)))})
    return entities, relations, rejected


def process_document(project: Project, doc_id: str, llm: LLM) -> dict[str, int]:
    onto = project.ontology()
    if not onto.is_ready():
        raise LLMError("Ontolojide hiç varlık türü yok; önce ontolojiyi tanımlayın.")
    doc = project.document(doc_id)
    if doc is None:
        raise ValueError("belge bulunamadı")
    project.update_document(doc_id, status="processing", error=None)
    try:
        text = project.text(doc_id)
        chunks = chunk_text(text, MANUAL_CHUNK if is_manual(llm) else API_CHUNK)
        ents, rels, rej = [], [], []
        for i, chunk in enumerate(chunks, 1):
            log.info("%s: parça %d/%d işleniyor", doc["filename"], i, len(chunks))
            title = doc["filename"] + (f" (bölüm {i}/{len(chunks)})" if len(chunks) > 1 else "")
            e, r, x = extract_chunk(llm, onto, title, chunk)
            ents += e
            rels += r
            rej += x
        project.remove_contributions(doc_id)
        project.add_extraction(doc_id, ents, rels, rej)
        counts = {"n_entities": len({(e["type"], norm_name(e["name"])) for e in ents}),
                  "n_relations": len(rels), "n_rejected": len(rej)}
        project.update_document(doc_id, status="done", processed_at=_now(), **counts)
        log.info("%s: %d varlık, %d ilişki, %d reddedilen", doc["filename"], *counts.values())
        return counts
    except Exception as exc:
        project.update_document(doc_id, status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        raise


def suggest_ontology(llm: LLM, text: str, hint: str = "") -> Ontology:
    excerpt = text[:MANUAL_CHUNK // 4 if is_manual(llm) else 20000]
    user = (f"Kullanıcının alan açıklaması: {hint}\n\n" if hint else "") + f"=== DOCUMENT EXCERPT ===\n{excerpt}"
    s = llm.complete_json(SUGGEST_SYSTEM, user, SuggestedOntology)
    return Ontology(domain=s.domain, entity_types=s.entity_types, relation_types=s.relation_types).cleaned()


def _now() -> str:
    from .store import now
    return now()
