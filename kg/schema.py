"""Pydantic modelleri: graf düğümleri/kenarları ve LLM çıkarım şeması.

İlke: kanıt (SUPPORTS/CONTRADICTS ve bir Claim'den türetilen varlık kenarları) ile
hipotez (HYPOTHESIS) asla aynı modelde/kenar tipinde karışmaz.
"""
from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SOURCE_SENTENCE_MAX = 300

# ---------------------------------------------------------------- sabitler


class Design(str, Enum):
    in_vivo = "in vivo"
    in_vitro = "in vitro"
    ex_vivo = "ex vivo"
    rct_human = "RCT human"
    observational_human = "observational human"


class ClaimStatus(str, Enum):
    evidence = "evidence"
    hypothesis = "hypothesis"
    refuted = "refuted"


class HypothesisStatus(str, Enum):
    untested = "untested"
    tested_supported = "tested_supported"
    tested_refuted = "tested_refuted"


class VerificationLabel(str, Enum):
    supported = "supported"
    partial = "partial"
    unsupported = "unsupported"


ENTITY_TYPES = ("Gene", "Intervention", "Model", "CellType", "Outcome", "Disease")
EntityType = Literal["Gene", "Intervention", "Model", "CellType", "Outcome", "Disease"]

# Varlık → varlık kanıt yüklemleri (yalnızca bir Claim'den türetilir, claim_id taşır).
# PROTECTS, rules.yaml'daki "koruyucu etki" kalıpları için eklenmiştir.
EVIDENCE_PREDICATES = ("INCREASES", "DECREASES", "REQUIRES", "MEDIATES", "PROTECTS")
Predicate = Literal["INCREASES", "DECREASES", "REQUIRES", "MEDIATES", "PROTECTS"]
EvidenceRelation = Literal["SUPPORTS", "CONTRADICTS"]
Polarity = Literal["+", "-"]
FundingSource = Literal["academic", "industry", "mixed"]
TracerType = Literal["3H", "13C", "deuterium", "NBD", "BODIPY", "none"]

SPECIES_ALIASES = {
    "mouse": "mouse", "mice": "mouse", "murine": "mouse", "mus musculus": "mouse",
    "rat": "rat", "rats": "rat", "rattus norvegicus": "rat",
    "human": "human", "humans": "human", "homo sapiens": "human", "patient": "human", "patients": "human",
    "pig": "pig", "pigs": "pig", "swine": "pig", "porcine": "pig", "minipig": "pig",
    "primate": "primate", "monkey": "primate", "macaque": "primate", "non-human primate": "primate",
    "cell line": "cell_line", "cell_line": "cell_line", "cells": "cell_line",
}


def normalize_species(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    return SPECIES_ALIASES.get(v, v or None)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" .;")


def claim_id(text: str) -> str:
    """Claim kimliği = hash(normalize text)."""
    return "c_" + hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def experiment_id(doi: str, figure: str | None, panel: str | None) -> str:
    fig = normalize_text(figure or "nofig").replace("figure", "fig").replace("fig.", "fig").replace(" ", "")
    pan = normalize_text(panel or "").replace(" ", "")
    return f"{doi.lower()}#{fig}{('/' + pan) if pan else ''}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- düğümler


class Paper(BaseModel):
    doi: str
    title: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    lab: Optional[str] = None  # sorumlu yazar/laboratuvar veya kurum
    affiliation: Optional[str] = None
    retracted: bool = False
    funding_source: Optional[FundingSource] = None
    fulltext: bool = False
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    seed_set: Optional[str] = None
    role: Optional[str] = None

    @field_validator("doi")
    @classmethod
    def _lower_doi(cls, v: str) -> str:
        return v.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")


class Experiment(BaseModel):
    id: str
    paper_doi: str
    figure: Optional[str] = None
    panel: Optional[str] = None
    design: Optional[Design] = None
    species: Optional[str] = None
    sex: Optional[str] = None
    n_per_group: Optional[list[int]] = None
    blinded: Optional[bool] = None
    randomized: Optional[bool] = None
    tracer_type: Optional[TracerType] = None
    model_name: Optional[str] = None


class Claim(BaseModel):
    id: str
    text: str
    polarity: Polarity = "+"
    status: ClaimStatus = ClaimStatus.evidence
    predicate: Optional[Predicate] = None


class Entity(BaseModel):
    type: EntityType
    name: str
    species: Optional[str] = None  # Gene için zorunlu anlamda: fare geni ≠ insan geni
    symbol: Optional[str] = None
    uniprot_id: Optional[str] = None
    xref: Optional[str] = None  # MGI:..., HGNC:..., MESH:..., CL:..., CHEBI:...
    intervention_type: Optional[str] = None
    dose: Optional[str] = None
    schedule: Optional[str] = None
    severity: Optional[str] = None
    measure: Optional[str] = None
    direction: Optional[str] = None
    unresolved: bool = True
    match_suggestion: Optional[str] = None

    @property
    def key(self) -> str:
        """Graf kimliği. Gen/protein türe özgüdür; diğerleri ad (veya xref) ile."""
        if self.xref:
            base = self.xref
        else:
            base = normalize_text(self.symbol or self.name)
        if self.type == "Gene":
            return f"Gene:{self.species or 'unknown'}:{base}"
        return f"{self.type}:{base}"


class Provenance(BaseModel):
    """Her kenarın taşıdığı ortak alanlar — kaynaksız kenar yazılmaz."""
    source_doi: str
    figure: Optional[str] = None
    source_sentence: str
    extraction_model: str
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    verified_by_human: bool = False
    verification: Optional[VerificationLabel] = None
    needs_review: bool = False
    ingest_run_id: Optional[str] = None

    @field_validator("source_sentence")
    @classmethod
    def _limit(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("source_sentence boş olamaz")
        return v[:SOURCE_SENTENCE_MAX]


class EvidenceEdge(Provenance):
    """Experiment -[SUPPORTS|CONTRADICTS]-> Claim"""
    relation: EvidenceRelation
    p_value: Optional[str] = None
    effect_size: Optional[str] = None


class Hypothesis(BaseModel):
    """Yalnızca çıkarım katmanının yazdığı HYPOTHESIS kenarı."""
    subject_key: str
    object_key: str
    predicate: str
    path: list[dict[str, Any]]  # her halka: {from, predicate, to, claim_id, source_doi, figure, E}
    score: float
    chain_confidence: float
    embed_score: Optional[float] = None
    rationale: str
    status: HypothesisStatus = HypothesisStatus.untested
    weakest_link: Optional[str] = None
    generators: list[str] = Field(default_factory=list)
    literature_dois: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject_key, self.predicate, self.object_key)


class IngestRun(BaseModel):
    id: str
    date: str = Field(default_factory=utcnow)
    model: str
    n_papers: int = 0
    n_claims: int = 0
    n_rejected: int = 0
    n_partial: int = 0
    dois: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- LLM çıkarım şeması


class ExtractedEntity(BaseModel):
    type: EntityType
    name: Optional[str] = Field(None, description="Metinde geçtiği biçimiyle ad")
    symbol: Optional[str] = Field(None, description="Gen/protein sembolü, metinde yazdığı gibi")
    species: Optional[str] = Field(None, description="mouse | rat | human | pig | primate | cell_line; bilinmiyorsa null")
    measure: Optional[str] = Field(None, description="Outcome için ölçüm (ör. pole test, MWM)")
    direction: Optional[str] = Field(None, description="Outcome için yön: improved | worsened | unchanged")
    intervention_type: Optional[str] = Field(None, description="mechanical | drug | gene | antibody | serum | other")
    dose: Optional[str] = None
    schedule: Optional[str] = None
    severity: Optional[str] = None

    def display_name(self) -> str:
        return self.symbol or self.name or self.measure or "?"


class ExtractedExperiment(BaseModel):
    figure: Optional[str] = Field(None, description="ör. 'Fig. 3' veya 'Extended Data Fig. 2'")
    panel: Optional[str] = Field(None, description="ör. 'd-h'")
    design: Optional[Literal["in vivo", "in vitro", "ex vivo", "RCT human", "observational human"]] = None
    species: Optional[str] = None
    sex: Optional[str] = None
    n_per_group: Optional[list[int]] = Field(None, description="Metinde yazan grup büyüklükleri; yazmıyorsa null")
    blinded: Optional[bool] = None
    randomized: Optional[bool] = None
    tracer_type: Optional[TracerType] = None
    model_name: Optional[str] = Field(None, description="ör. CCI-TBI, MCAO, porcine CCI")


class ExtractedClaim(BaseModel):
    claim: str = Field(description="Tek cümlelik, normalize edilmiş iddia (İngilizce)")
    polarity: Polarity
    predicate: Optional[Predicate] = Field(None, description="subject → object ilişkisi")
    evidence_relation: EvidenceRelation = Field(description="Deney iddiayı destekliyor mu, çürütüyor mu")
    subject: ExtractedEntity
    object: ExtractedEntity
    experiment: ExtractedExperiment
    source_sentence: str = Field(description="Kaynak cümle, metinden BİREBİR kopya")
    p_value: Optional[str] = None
    effect_size: Optional[str] = None
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    claims: list[ExtractedClaim]


class VerificationItem(BaseModel):
    index: int
    label: VerificationLabel
    reason: str


class VerificationResult(BaseModel):
    items: list[VerificationItem]


# ---------------------------------------------------------------- JSON şema yardımcıları


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic şemasını yapılandırılmış çıktı için katı biçime çevirir:
    $ref'ler açılır, her nesnede tüm alanlar required ve additionalProperties=false."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            out = {k: resolve(v) for k, v in node.items() if k not in ("default", "title")}
            if out.get("type") == "object" and "properties" in out:
                out["required"] = list(out["properties"].keys())
                out["additionalProperties"] = False
            # sayı sınırları yapılandırılmış çıktıda desteklenmeyebilir; doğrulamayı Pydantic yapar
            for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                out.pop(k, None)
            return out
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)
