"""Boru hattı: keşif → tam metin → bölümleme → LLM çıkarımı → öz-doğrulama → normalizasyon → yükleme.

İdempotent: her makale DOI ile anahtarlanır; ham metin ve çıkarımlar data/ altında önbelleklenir,
graf yazmaları MERGE ile yapılır. Tekrar çalıştırmak çift kayıt üretmez.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT, ensure_dirs, sources_config
from .discover import Discoverer, norm_doi
from .extract import Extractor, load_extraction
from .fetch import Document, Fetcher
from .graph import GraphStore
from .http import Http
from .llm import LLM
from .load import GraphLoader
from .normalize import Normalizer
from .schema import Entity, IngestRun, Paper, VerificationLabel
from .validate import consistency

log = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    run: IngestRun
    papers: list[str] = field(default_factory=list)
    skipped_retracted: list[str] = field(default_factory=list)
    abstract_only: list[str] = field(default_factory=list)
    entities_total: int = 0
    entities_resolved: int = 0
    inconsistencies: int = 0
    orthologs: int = 0
    replications: int = 0

    @property
    def normalization_rate(self) -> float:
        return self.entities_resolved / self.entities_total if self.entities_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {**self.run.model_dump(), "skipped_retracted": self.skipped_retracted,
                "abstract_only": self.abstract_only, "normalization_rate": round(self.normalization_rate, 3),
                "inconsistencies": self.inconsistencies, "orthologs": self.orthologs,
                "replications": self.replications}


def new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


class Pipeline:
    def __init__(self, store: GraphStore, llm: LLM, offline: bool = False, force: bool = False):
        ensure_dirs()
        self.store = store
        self.llm = llm
        self.force = force
        self.http = Http(offline=offline)
        self.discoverer = Discoverer(self.http)
        self.fetcher = Fetcher(self.http)
        self.extractor = Extractor(llm)
        self.normalizer = Normalizer(self.http, offline=offline)
        self.loader = GraphLoader(store, self.normalizer)
        self.offline = offline

    # ------------------------------------------------------------ makale listeleri

    def phase1_papers(self, pdf: str | None = None) -> list[tuple[Paper, str | None]]:
        seed = sources_config()["seed_sets"]["bone_brain"][0]
        doi = norm_doi(seed["doi"])
        paper = self._metadata(doi, "bone_brain", seed.get("role", "seed"))
        if pdf:
            return [(paper, str(Path(pdf).resolve()))]
        pdf_path = seed.get("pdf")
        return [(paper, str(ROOT / pdf_path) if pdf_path else None)]

    def expansion_papers(self, sets: list[str] | None = None) -> tuple[list[tuple[Paper, str | None]], list[tuple[str, str]]]:
        cfg = sources_config()
        exp = cfg.get("expansion", {})
        out, cites = [], []
        for name, seeds in cfg["seed_sets"].items():
            if sets and name not in sets:
                continue
            papers, edges = self.discoverer.discover(
                name, seeds, (cfg.get("query_terms") or {}).get(name, []), hops=exp.get("hops", 1),
                max_papers=exp.get("max_papers_per_set", 200), include_references=exp.get("include_references", True),
                include_citations=exp.get("include_citations", True),
                max_query_results=exp.get("max_query_results_per_term", 25))
            pdfs = {norm_doi(s.get("doi")): s.get("pdf") for s in seeds if s.get("doi")}
            out += [(p, pdfs.get(p.doi)) for p in papers]
            cites += edges
        return out, cites

    def _metadata(self, doi: str, seed_set: str | None, role: str | None) -> Paper:
        if self.offline:
            return Paper(doi=doi, seed_set=seed_set, role=role)
        try:
            return self.discoverer.paper_metadata(doi, seed_set, role)
        except Exception as exc:  # meta veri olmadan da devam edilir
            log.warning("Meta veri alınamadı (%s): %s", doi, exc)
            return Paper(doi=doi, seed_set=seed_set, role=role)

    # ------------------------------------------------------------ çalıştırma

    def ingest(self, papers: list[tuple[Paper, str | None]], cites: list[tuple[str, str]] | None = None,
               documents: dict[str, Document] | None = None, run_id: str | None = None) -> IngestSummary:
        self.store.setup()
        self.normalizer.register_known(
            Entity.model_validate({k: v for k, v in n.items() if k not in ("label", "key")})
            for n in self.store.snapshot().entities().values())
        run = IngestRun(id=run_id or new_run_id(), model=self.llm.model_name)
        summary = IngestSummary(run=run)
        all_inconsistencies = consistency.load_report()

        for paper, pdf in papers:
            log.info("İşleniyor: %s", paper.doi)
            if paper.retracted:
                self.store.upsert_paper(paper)
                summary.skipped_retracted.append(paper.doi)
                continue
            doc = (documents or {}).get(paper.doi) or self.fetcher.fetch(paper.doi, pmcid=paper.pmcid, pdf_path=pdf)
            if not doc.sections:
                log.warning("Metin bulunamadı: %s", paper.doi)
                self.store.upsert_paper(paper)
                continue
            if not doc.fulltext:
                summary.abstract_only.append(paper.doi)
            extraction = None if self.force else load_extraction(paper.doi)
            if extraction is None:
                extraction = self.extractor.extract_document(doc)
            n = self.loader.load_paper(paper, extraction, run.id)
            run.n_papers += 1
            run.n_claims += n
            run.n_rejected += len(extraction.rejected)
            run.n_partial += sum(1 for r in extraction.accepted if r.verification == VerificationLabel.partial)
            run.dois.append(paper.doi)
            summary.papers.append(paper.doi)
            found = consistency.check_document(doc, extraction)
            all_inconsistencies = consistency.merge_reports(all_inconsistencies, found, paper.doi)

        for citing, cited in cites or []:
            self.store.add_cites(citing, cited)
        summary.orthologs = self.loader.link_orthologs()
        summary.replications = self.loader.link_replications()
        self.store.record_ingest_run(run)
        self.normalizer.save()

        snap = self.store.snapshot()
        ents = [n for n in snap.entities().values()]
        summary.entities_total = len(ents)
        summary.entities_resolved = sum(1 for n in ents if not n.get("unresolved", True))
        consistency.write_report(all_inconsistencies)
        summary.inconsistencies = len(all_inconsistencies)
        if hasattr(self.store, "flush"):
            self.store.flush()
        return summary

    def ingest_dois(self, dois: list[str]) -> IngestSummary:
        """Literatür kontrolünden gelen makaleleri boru hattına ekler."""
        papers = [(self._metadata(d, None, "literature_check"), None) for d in dois]
        return self.ingest(papers)


def unresolved_entities(store: GraphStore) -> list[dict[str, Any]]:
    snap = store.snapshot()
    return [{"key": k, **{f: n.get(f) for f in ("label", "name", "symbol", "species", "match_suggestion")}}
            for k, n in snap.entities().items() if n.get("unresolved", True)]


__all__ = ["Pipeline", "IngestSummary", "unresolved_entities", "Entity"]
