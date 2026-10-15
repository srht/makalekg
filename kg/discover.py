"""Keşif: tohum makaleler → referanslar + atıf yapanlar + anahtar kavram sorguları (1–2 atlama).

Kaynaklar: Europe PMC REST API, Semantic Scholar Graph API, Crossref (meta veri, fon kaynağı,
retraction durumu; Retraction Watch verisi Crossref üzerinden "updated-by" alanında gelir).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from .config import env
from .http import Http
from .schema import Paper

log = logging.getLogger(__name__)

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
S2 = "https://api.semanticscholar.org/graph/v1"
CROSSREF = "https://api.crossref.org/works"

INDUSTRY_PATTERN = re.compile(
    r"\b(inc|ltd|llc|gmbh|corp|corporation|co\.|plc|ag|s\.a\.|pharma\w*|therapeutics|biotech\w*|"
    r"laborator(?:y|ies) inc|infraredx|amgen|pfizer|novartis|roche|merck|sanofi|astrazeneca|"
    r"regeneron|esperion|lilly|bayer|takeda|abbvie|gsk|glaxo\w*|medtronic|abbott)\b",
    re.I,
)


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.strip().lower().removeprefix("https://doi.org/").removeprefix("doi:")


def classify_funding(funder_names: Iterable[str]) -> str | None:
    names = [n for n in funder_names if n]
    if not names:
        return None
    industry = [bool(INDUSTRY_PATTERN.search(n)) for n in names]
    if all(industry):
        return "industry"
    if any(industry):
        return "mixed"
    return "academic"


class Discoverer:
    def __init__(self, http: Http | None = None):
        self.http = http or Http()
        key = env("SEMANTIC_SCHOLAR_API_KEY")
        self.s2_headers = {"x-api-key": key} if key else None

    # ------------------------------------------------------------ Europe PMC

    def epmc_search(self, query: str, page_size: int = 25) -> list[dict[str, Any]]:
        data = self.http.get(f"{EPMC}/search", params={
            "query": query, "format": "json", "resultType": "core", "pageSize": page_size,
        })
        return ((data or {}).get("resultList") or {}).get("result") or []

    def epmc_by_doi(self, doi: str) -> dict[str, Any] | None:
        results = self.epmc_search(f'DOI:"{doi}"', page_size=1)
        return results[0] if results else None

    def epmc_links(self, record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
        """kind: references | citations"""
        src, pid = record.get("source"), record.get("id")
        if not src or not pid:
            return []
        data = self.http.get(f"{EPMC}/{src}/{pid}/{kind}", params={"format": "json", "pageSize": 1000})
        if not data:
            return []
        if kind == "references":
            return (data.get("referenceList") or {}).get("reference") or []
        return (data.get("citationList") or {}).get("citation") or []

    def resolve_title(self, title: str, year: int | None = None, journal: str | None = None) -> str | None:
        """sources.yaml'da yalnızca başlıkla verilen tohumların DOI'sini çözer."""
        q = f'TITLE:"{title}"' + (f" AND PUB_YEAR:{year}" if year else "")
        for rec in self.epmc_search(q, page_size=5):
            if rec.get("doi"):
                if journal and journal.lower().split()[0] not in (rec.get("journalTitle") or "").lower() \
                        and journal.lower() not in (rec.get("journalInfo", {}).get("journal", {}).get("isoabbreviation") or "").lower():
                    continue
                return norm_doi(rec["doi"])
        return None

    # ------------------------------------------------------------ Semantic Scholar

    def s2_links(self, doi: str, kind: str, limit: int = 1000) -> list[str]:
        field = "citedPaper" if kind == "references" else "citingPaper"
        data = self.http.get(f"{S2}/paper/DOI:{doi}/{kind}", params={"fields": "externalIds", "limit": limit},
                             headers=self.s2_headers)
        dois = []
        for item in (data or {}).get("data") or []:
            ext = (item.get(field) or {}).get("externalIds") or {}
            if ext.get("DOI"):
                dois.append(norm_doi(ext["DOI"]))
        return dois

    # ------------------------------------------------------------ Crossref

    def crossref(self, doi: str) -> dict[str, Any] | None:
        data = self.http.get(f"{CROSSREF}/{doi}")
        return (data or {}).get("message")

    @staticmethod
    def is_retracted(crossref_msg: dict[str, Any] | None, epmc: dict[str, Any] | None) -> bool:
        if crossref_msg:
            for upd in crossref_msg.get("updated-by") or []:
                if (upd.get("type") or "").lower() in ("retraction", "removal", "withdrawal"):
                    return True
            for rel in (crossref_msg.get("relation") or {}).get("is-retracted-by", []) or []:
                if rel:
                    return True
            if any("retracted" in (t or "").lower()[:12] for t in crossref_msg.get("title") or []):
                return True
        if epmc:
            types = ((epmc.get("pubTypeList") or {}).get("pubType")) or []
            if any("retracted publication" in t.lower() for t in types):
                return True
        return False

    # ------------------------------------------------------------ meta veri

    def paper_metadata(self, doi: str, seed_set: str | None = None, role: str | None = None) -> Paper:
        doi = norm_doi(doi)
        epmc = None
        cr = None
        try:
            epmc = self.epmc_by_doi(doi)
        except ConnectionError as exc:
            log.warning("Europe PMC erişilemedi (%s): %s", doi, exc)
        try:
            cr = self.crossref(doi)
        except ConnectionError as exc:
            log.warning("Crossref erişilemedi (%s): %s", doi, exc)

        title = (epmc or {}).get("title") or ((cr or {}).get("title") or [None])[0]
        year = None
        if epmc and epmc.get("pubYear"):
            year = int(epmc["pubYear"])
        elif cr:
            parts = ((cr.get("issued") or {}).get("date-parts") or [[None]])[0]
            year = parts[0] if parts and parts[0] else None
        journal = ((epmc or {}).get("journalInfo") or {}).get("journal", {}).get("title") \
            or ((cr or {}).get("container-title") or [None])[0]

        lab, affiliation = self._lab(epmc, cr)
        funders = [f.get("name", "") for f in (cr or {}).get("funder") or []]
        if epmc:
            funders += [g.get("agency", "") for g in ((epmc.get("grantsList") or {}).get("grant") or [])]

        return Paper(
            doi=doi, title=title, year=year, journal=journal, lab=lab, affiliation=affiliation,
            retracted=self.is_retracted(cr, epmc), funding_source=classify_funding(funders),
            pmid=(epmc or {}).get("pmid"), pmcid=(epmc or {}).get("pmcid"),
            seed_set=seed_set, role=role,
        )

    @staticmethod
    def _lab(epmc: dict[str, Any] | None, cr: dict[str, Any] | None) -> tuple[str | None, str | None]:
        """Laboratuvar = son yazar (genellikle sorumlu yazar) + kurumu. REPLICATES için kullanılır."""
        authors = ((epmc or {}).get("authorList") or {}).get("author") or []
        if authors:
            last = authors[-1]
            aff = None
            details = (last.get("authorAffiliationDetailsList") or {}).get("authorAffiliation") or []
            if details:
                aff = details[0].get("affiliation")
            aff = aff or last.get("affiliation")
            return last.get("lastName") or last.get("fullName"), aff
        cr_authors = (cr or {}).get("author") or []
        if cr_authors:
            last = cr_authors[-1]
            affs = [a.get("name") for a in last.get("affiliation") or [] if a.get("name")]
            return last.get("family"), (affs[0] if affs else None)
        return None, None

    # ------------------------------------------------------------ genişleme

    def neighbors(self, doi: str) -> tuple[list[str], list[str]]:
        """(referanslar, atıf yapanlar) DOI listeleri: Europe PMC + Semantic Scholar birleşimi."""
        refs: list[str] = []
        cites: list[str] = []
        try:
            rec = self.epmc_by_doi(doi)
            if rec:
                refs += [norm_doi(r.get("doi")) for r in self.epmc_links(rec, "references") if r.get("doi")]
                cites += [norm_doi(c.get("doi")) for c in self.epmc_links(rec, "citations") if c.get("doi")]
        except ConnectionError as exc:
            log.warning("Europe PMC komşuları alınamadı: %s", exc)
        try:
            refs += self.s2_links(doi, "references")
            cites += self.s2_links(doi, "citations")
        except ConnectionError as exc:
            log.warning("Semantic Scholar komşuları alınamadı: %s", exc)
        return _dedupe(refs), _dedupe(cites)

    def discover(self, seed_set: str, seeds: list[dict[str, Any]], query_terms: list[str],
                 hops: int = 1, max_papers: int = 200, include_references: bool = True,
                 include_citations: bool = True, max_query_results: int = 25) -> tuple[list[Paper], list[tuple[str, str]]]:
        """Makale listesi ve CITES kenarları (citing_doi, cited_doi) döner. Tohumlar her zaman ilk sıradadır."""
        ordered: list[tuple[str, str | None]] = []  # (doi, role)
        seen: set[str] = set()
        cites_edges: list[tuple[str, str]] = []

        def add(doi: str | None, role: str | None) -> None:
            if doi and doi not in seen and len(ordered) < max_papers:
                seen.add(doi)
                ordered.append((doi, role))

        seed_dois = []
        for s in seeds:
            doi = norm_doi(s.get("doi"))
            if not doi and s.get("title"):
                doi = self.resolve_title(s["title"], s.get("year"), s.get("journal"))
                if not doi:
                    log.warning("Başlık DOI'ye çözülemedi: %s", s["title"])
                    continue
            seed_dois.append(doi)
            add(doi, s.get("role", "seed"))

        frontier = list(seed_dois)
        for hop in range(1, hops + 1):
            nxt: list[str] = []
            for doi in frontier:
                refs, citing = self.neighbors(doi)
                cites_edges += [(doi, r) for r in refs]
                cites_edges += [(c, doi) for c in citing]
                if include_references:
                    for r in refs:
                        add(r, f"reference_hop{hop}")
                        nxt.append(r)
                if include_citations:
                    for c in citing:
                        add(c, f"citing_hop{hop}")
                        nxt.append(c)
            frontier = nxt

        for term in query_terms:
            try:
                for rec in self.epmc_search(term, page_size=max_query_results):
                    add(norm_doi(rec.get("doi")), f"query:{term}")
            except ConnectionError as exc:
                log.warning("Sorgu başarısız (%s): %s", term, exc)

        papers = []
        for doi, role in ordered:
            papers.append(self.paper_metadata(doi, seed_set=seed_set, role=role))
        kept = {p.doi for p in papers}
        cites_edges = [(a, b) for a, b in set(cites_edges) if a in kept and b in kept]
        return papers, cites_edges


def _dedupe(items: Iterable[str | None]) -> list[str]:
    out, seen = [], set()
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out
