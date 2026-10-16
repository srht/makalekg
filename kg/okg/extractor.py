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


def _attribute(texts: dict[str, str], needle: str) -> list[str]:
    n = _ws(needle)
    return [d for d, t in texts.items() if n and n in t]


def process_batch(project: Project, doc_ids: list[str], llm: LLM) -> None:
    """Kısa belgeleri (ör. arşiv özetleri) tek istemde işler; her öğe metninde geçtiği belgeye atanır."""
    onto = project.ontology()
    docs = {d["id"]: d for d in project.documents() if d["id"] in doc_ids}
    for d in doc_ids:
        project.update_document(d, status="processing", error=None)
    try:
        raw = {d: project.text(d) for d in doc_ids}
        combined = "\n\n".join(f"### {docs[d]['filename']}\n{raw[d]}" for d in doc_ids)
        log.info("%d kısa belge tek istemde işleniyor", len(doc_ids))
        ents, rels, rej = extract_chunk(llm, onto, f"{len(doc_ids)} kısa belge", combined)
        norm = {d: _ws(t) for d, t in raw.items()}
        per_doc: dict[str, tuple[list, list, list]] = {d: ([], [], []) for d in doc_ids}
        for e in ents:
            for d in _attribute(norm, e["mention"]) or _attribute(norm, e["name"]):
                per_doc[d][0].append(e)
        ent_by_key = {Project.entity_key(e["type"], e["name"]): e for e in ents}
        for r in rels:
            for d in _attribute(norm, r["sentence"]):
                per_doc[d][1].append(r)
                for k in (r["subject_key"], r["object_key"]):  # ilişkinin uçları o belgede de anılsın
                    if k in ent_by_key and ent_by_key[k] not in per_doc[d][0]:
                        per_doc[d][0].append(ent_by_key[k])
        for x in rej:
            targets = _attribute(norm, x.get("text") or "") or doc_ids[:1]
            per_doc[targets[0]][2].append(x)
        for d, (e, r, x) in per_doc.items():
            project.remove_contributions(d)
            project.add_extraction(d, e, r, x)
            project.update_document(d, status="done", processed_at=_now(),
                                    n_entities=len({(i["type"], norm_name(i["name"])) for i in e}),
                                    n_relations=len(r), n_rejected=len(x))
    except Exception as exc:
        for d in doc_ids:
            project.update_document(d, status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        raise


def plan_batches(project: Project, doc_ids: list[str], llm: LLM) -> list[list[str]]:
    """Kısa belgeleri istem boyutunu aşmayacak gruplara toplar; uzunlar tek başına işlenir."""
    limit = MANUAL_CHUNK if is_manual(llm) else API_CHUNK
    small_max = limit // 3
    chars = {d["id"]: d["chars"] for d in project.documents()}
    batches, cur, size = [], [], 0
    for d in doc_ids:
        c = chars.get(d, 0)
        if c > small_max:
            batches.append([d])
            continue
        if cur and size + c > limit:
            batches.append(cur)
            cur, size = [], 0
        cur.append(d)
        size += c + 100
    if cur:
        batches.append(cur)
    return batches


def process_many(project: Project, doc_ids: list[str], llm: LLM) -> int:
    done = 0
    for batch in plan_batches(project, doc_ids, llm):
        if len(batch) == 1:
            process_document(project, batch[0], llm)
        else:
            process_batch(project, batch, llm)
        done += len(batch)
    return done


def suggest_ontology(llm: LLM, text: str, hint: str = "") -> Ontology:
    excerpt = text[:MANUAL_CHUNK // 4 if is_manual(llm) else 20000]
    user = (f"Kullanıcının alan açıklaması: {hint}\n\n" if hint else "") + f"=== DOCUMENT EXCERPT ===\n{excerpt}"
    s = llm.complete_json(SUGGEST_SYSTEM, user, SuggestedOntology)
    return Ontology(domain=s.domain, entity_types=s.entity_types, relation_types=s.relation_types).cleaned()


def _now() -> str:
    from .store import now
    return now()
