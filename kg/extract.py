"""LLM çıkarımı + öz-doğrulama.

1. Her bölüm (ve içinde atıf yapılan şekil açıklamaları bağlam olarak) sabit JSON şemasıyla
   işlenir: Claim, Experiment, varlıklar.
2. Deterministik kontrol: kaynak cümle metinde birebir geçmeli.
3. İkinci LLM çağrısı her çıkarımı kaynak cümlesiyle karşılaştırır: supported / partial / unsupported.
   unsupported olanlar grafa girmez, data/review_queue.jsonl dosyasına yazılır.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from .config import DATA_DIR
from .fetch import Document, Section, figure_key
from .llm import LLM, LLMError
from .schema import (SOURCE_SENTENCE_MAX, ExtractedClaim, ExtractionResult, VerificationLabel,
                     VerificationResult, claim_id, utcnow)

log = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 12000

EXTRACTION_SYSTEM = """You extract experimental claims from biomedical papers into a knowledge graph.

Rules (strict):
- Extract ONLY what the text explicitly states. Do not infer, generalise, or add background knowledge.
- One claim = one experimental finding, written as a single normalised English sentence.
- source_sentence MUST be copied VERBATIM from the PRIMARY TEXT or the FIGURE LEGENDS given below
  (exact characters, at most 300 characters; if the sentence is longer, copy its first 300 characters).
- Any field the text does not state is null. Never guess numbers: n_per_group, p_value, effect_size,
  sex, blinded, randomized stay null unless written. blinded=false only if the text says so explicitly.
- Keep species and context separate: mouse in vivo, cell culture (in vitro), ex vivo and human findings
  are different experiments. A mouse gene (e.g. Piezo1, species=mouse) and a human gene (PIEZO1,
  species=human) are different entities; write symbols exactly as in the text.
- experiment.figure / panel: the figure the finding is shown in (e.g. "Fig. 3", panel "d-h"), or null.
- evidence_relation: SUPPORTS if the experiment supports the claim as written; CONTRADICTS if the
  experiment reports evidence against a stated hypothesis (e.g. a null/negative result for an expected effect).
- polarity: "+" if the claim asserts an effect/relationship exists, "-" if it asserts absence of an effect.
- predicate (subject → object): INCREASES, DECREASES, REQUIRES (subject's effect needs object),
  MEDIATES (subject mediates/causes the effect on object), PROTECTS (subject protects/improves object
  outcome). null if none fits.
- Entity types: Gene (genes/proteins), Intervention (mechanical loading, drug, gene deletion, antibody,
  serum transfer...), Model (injury/disease model, e.g. CCI-TBI, MCAO), CellType, Outcome (a measured
  readout such as pole test, Morris water maze, MAP2, Nissl, survival), Disease.
- tracer_type: 3H, 13C, deuterium, NBD, BODIPY if a tracer is used, "none" if explicitly none, else null.
- extraction_confidence (0-1): lower it whenever you are unsure about any field.
- Do not extract claims from the introduction that only cite other work.
- If there are no experimental claims, return an empty list."""

VERIFY_SYSTEM = """You audit knowledge-graph extractions against their source sentences.
For each item decide whether the source sentence (with the provided context) states the claim and
the structured fields:
- supported: the sentence clearly states the claim; subject, object, predicate, polarity and numbers match.
- partial: the core claim is stated but some field is overstated, imprecise or not in the sentence.
- unsupported: the sentence does not state the claim, or numbers/direction/species contradict it.
Be strict. Give a short reason. Return one item per input index."""


class ExtractionRecord(BaseModel):
    doi: str
    section: str
    claim: ExtractedClaim
    claim_id: str
    verification: VerificationLabel
    verification_reason: str = ""
    verbatim: bool = True


class PaperExtraction(BaseModel):
    doi: str
    model: str
    created_at: str
    fulltext: bool
    accepted: list[ExtractionRecord]
    rejected: list[ExtractionRecord]


def _norm_ws(s: str) -> str:
    s = s.replace(" ", " ").replace(" ", " ").replace("‐", "-").replace("–", "-").replace("−", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def chunk_section(section: Section, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(section.text) <= max_chars:
        return [section.text]
    parts, buf = [], ""
    for para in re.split(r"\n+", section.text):
        if len(buf) + len(para) + 1 > max_chars and buf:
            parts.append(buf)
            buf = ""
        while len(para) > max_chars:  # tek paragraf çok uzunsa cümle sınırında böl
            cut = para.rfind(". ", 0, max_chars) + 1 or max_chars
            parts.append(para[:cut])
            para = para[cut:].strip()
        buf = f"{buf}\n{para}" if buf else para
    if buf:
        parts.append(buf)
    return parts


def referenced_legends(text: str, doc: Document) -> list[Section]:
    keys = {figure_key(m.group(0)) for m in re.finditer(
        r"(?:extended\s+data\s+|supplementary\s+)?fig(?:ure)?s?\.?\s*s?\d+", text, re.I)}
    keys.discard(None)
    return [s for s in doc.legends() if figure_key(s.figure) in keys]


class Extractor:
    def __init__(self, llm: LLM, out_dir: Path | None = None, review_queue: Path | None = None):
        self.llm = llm
        self.out_dir = out_dir or DATA_DIR / "extractions"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.review_queue = review_queue or DATA_DIR / "review_queue.jsonl"

    def _units(self, doc: Document) -> Iterable[tuple[str, str, list[Section]]]:
        """(bölüm adı, ana metin, bağlam legend'leri) birimleri."""
        legends = doc.legends()
        for sec in doc.sections:
            if sec.kind == "legend":
                continue
            if sec.kind == "body" and sec.name.lower().startswith(("introduction", "references")):
                continue
            if sec.kind == "abstract" and doc.fulltext:
                continue  # tam metin varken özet, şekilsiz yinelenen deneyler üretir
            for i, chunk in enumerate(chunk_section(sec)):
                name = sec.name if i == 0 else f"{sec.name} (part {i + 1})"
                yield name, chunk, referenced_legends(chunk, doc)
        # Legend'ler kendi başına da işlenir (n ve p değerleri çoğunlukla burada)
        batch, size = [], 0
        for leg in legends:
            if size + len(leg.text) > MAX_CHUNK_CHARS and batch:
                yield "Figure legends", "\n\n".join(s.text for s in batch), []
                batch, size = [], 0
            batch.append(leg)
            size += len(leg.text)
        if batch:
            yield "Figure legends", "\n\n".join(s.text for s in batch), []

    def _prompt(self, doc: Document, name: str, text: str, context: list[Section]) -> str:
        ctx = "\n\n".join(s.text for s in context)
        return (f"Paper DOI: {doc.doi}\nSection: {name}\n"
                f"Full text available: {doc.fulltext}\n\n"
                f"=== PRIMARY TEXT ===\n{text}\n\n"
                f"=== FIGURE LEGENDS (context) ===\n{ctx or '(none)'}\n")

    def extract_document(self, doc: Document) -> PaperExtraction:
        accepted: list[ExtractionRecord] = []
        rejected: list[ExtractionRecord] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        corpus = _norm_ws(doc.full_text)

        for name, text, context in self._units(doc):
            try:
                result = self.llm.complete_json(EXTRACTION_SYSTEM, self._prompt(doc, name, text, context), ExtractionResult)
            except LLMError as exc:
                log.error("Çıkarım başarısız (%s / %s): %s", doc.doi, name, exc)
                continue
            claims = []
            for c in result.claims:
                c.source_sentence = c.source_sentence.strip()[:SOURCE_SENTENCE_MAX]
                key = (claim_id(c.claim), c.experiment.figure, c.experiment.panel)
                if key in seen:
                    continue
                seen.add(key)
                claims.append(c)
            if not claims:
                continue
            labels = self._verify(claims, text, context)
            for c, (label, reason) in zip(claims, labels):
                verbatim = _norm_ws(c.source_sentence) in corpus
                if not verbatim:
                    label, reason = VerificationLabel.unsupported, "kaynak cümle metinde birebir geçmiyor; " + reason
                rec = ExtractionRecord(doi=doc.doi, section=name, claim=c, claim_id=claim_id(c.claim),
                                       verification=label, verification_reason=reason, verbatim=verbatim)
                (rejected if label == VerificationLabel.unsupported else accepted).append(rec)

        extraction = PaperExtraction(doi=doc.doi, model=self.llm.model_name, created_at=utcnow(),
                                     fulltext=doc.fulltext, accepted=accepted, rejected=rejected)
        self._save(extraction)
        return extraction

    def _verify(self, claims: list[ExtractedClaim], text: str, context: list[Section]) -> list[tuple[VerificationLabel, str]]:
        items = [{"index": i, "claim": c.claim, "polarity": c.polarity, "predicate": c.predicate,
                  "subject": c.subject.display_name(), "object": c.object.display_name(),
                  "experiment": c.experiment.model_dump(exclude_none=True),
                  "p_value": c.p_value, "effect_size": c.effect_size,
                  "source_sentence": c.source_sentence} for i, c in enumerate(claims)]
        ctx = "\n\n".join(s.text for s in context)
        prompt = (f"=== CONTEXT (section text) ===\n{text}\n\n=== FIGURE LEGENDS ===\n{ctx or '(none)'}\n\n"
                  f"=== ITEMS ===\n{json.dumps(items, ensure_ascii=False, indent=1)}")
        try:
            result = self.llm.complete_json(VERIFY_SYSTEM, prompt, VerificationResult)
        except LLMError as exc:
            log.error("Öz-doğrulama başarısız: %s — tüm öğeler inceleme kuyruğuna", exc)
            return [(VerificationLabel.unsupported, "öz-doğrulama çalışmadı")] * len(claims)
        by_index = {it.index: (it.label, it.reason) for it in result.items}
        return [by_index.get(i, (VerificationLabel.unsupported, "doğrulayıcı bu öğeyi döndürmedi"))
                for i in range(len(claims))]

    def _save(self, extraction: PaperExtraction) -> None:
        slug = re.sub(r"[^a-z0-9]+", "_", extraction.doi.lower()).strip("_")
        (self.out_dir / f"{slug}.json").write_text(extraction.model_dump_json(indent=1), encoding="utf-8")
        # Aynı DOI için önceki kayıtları değiştir (idempotent)
        existing = []
        if self.review_queue.exists():
            existing = [l for l in self.review_queue.read_text(encoding="utf-8").splitlines()
                        if l.strip() and json.loads(l).get("doi") != extraction.doi]
        if existing or extraction.rejected:
            with open(self.review_queue, "w", encoding="utf-8") as fh:
                for line in existing:
                    fh.write(line + "\n")
                for rec in extraction.rejected:
                    fh.write(rec.model_dump_json() + "\n")


def load_extraction(doi: str, out_dir: Path | None = None) -> PaperExtraction | None:
    out_dir = out_dir or DATA_DIR / "extractions"
    slug = re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")
    path = out_dir / f"{slug}.json"
    return PaperExtraction.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
