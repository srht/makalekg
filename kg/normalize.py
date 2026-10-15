"""Normalizasyon: gen/protein → UniProt (+ MGI / HGNC xref), hastalık → MeSH, hücre tipi → Cell Ontology,
ilaç → ChEBI. Eşlenemeyen varlık unresolved=true ile tutulur; yeni düğüm açılmadan önce mevcut
düğümlere karşı bulanık eşleşme önerisi üretilir (otomatik birleştirme yapılmaz).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz import fuzz, process

from .config import DATA_DIR
from .http import Http
from .schema import Entity, ExtractedEntity, normalize_species

log = logging.getLogger(__name__)

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
HGNC = "https://rest.genenames.org/fetch/symbol"
MESH = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
OLS = "https://www.ebi.ac.uk/ols4/api/search"

TAXON = {"mouse": 10090, "human": 9606, "rat": 10116, "pig": 9823}
FUZZY_THRESHOLD = 90


class Normalizer:
    def __init__(self, http: Http | None = None, cache_path: Path | None = None, offline: bool = False):
        self.http = http or Http()
        self.offline = offline
        self.cache_path = cache_path or DATA_DIR / "cache" / "normalize.json"
        self.cache: dict[str, Any] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.known_names: dict[str, list[str]] = {}  # tip → mevcut düğüm adları (bulanık eşleşme için)

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")

    def register_known(self, entities: Iterable[Entity]) -> None:
        for e in entities:
            self.known_names.setdefault(e.type, [])
            label = e.symbol or e.name
            if label not in self.known_names[e.type]:
                self.known_names[e.type].append(label)

    def _cached(self, key: str, fn) -> Any:
        if key in self.cache:
            return self.cache[key]
        if self.offline:
            return None
        try:
            value = fn()
        except ConnectionError as exc:
            log.warning("Normalizasyon servisi erişilemedi (%s): %s", key, exc)
            return None
        self.cache[key] = value
        return value

    # ------------------------------------------------------------ servisler

    def _uniprot(self, symbol: str, species: str) -> dict[str, Any] | None:
        taxon = TAXON.get(species)
        if not taxon:
            return None

        def query(q: str) -> list[dict[str, Any]]:
            data = self.http.get(UNIPROT, params={
                "query": f"{q} AND organism_id:{taxon} AND reviewed:true",
                "fields": "accession,gene_primary,gene_synonym,xref_mgi,xref_hgnc,xref_rgd,protein_name",
                "format": "json", "size": 5})
            return (data or {}).get("results") or []

        def pick(results: list[dict[str, Any]], exact_only: bool) -> dict[str, Any] | None:
            for r in results:
                genes = r.get("genes") or []
                primary = ((genes[0].get("geneName") or {}).get("value")) if genes else None
                synonyms = [s.get("value") for g in genes for s in g.get("synonyms") or []]
                if primary and primary.lower() == symbol.lower():
                    return self._uniprot_row(r, primary, exact=True)
                if not exact_only and any(s and s.lower() == symbol.lower() for s in synonyms):
                    return self._uniprot_row(r, primary, exact=True)
            return None

        return self._cached(f"uniprot:{species}:{symbol.lower()}", lambda: (
            pick(query(f"gene_exact:{symbol}"), True) or pick(query(f"gene:{symbol}"), False)))

    @staticmethod
    def _uniprot_row(r: dict[str, Any], primary: str | None, exact: bool) -> dict[str, Any]:
        xrefs = {x.get("database"): x.get("id") for x in r.get("uniProtKBCrossReferences") or []}
        return {"accession": r.get("primaryAccession"), "symbol": primary, "exact": exact,
                "mgi": xrefs.get("MGI"), "hgnc": xrefs.get("HGNC"), "rgd": xrefs.get("RGD")}

    def hgnc(self, symbol: str) -> dict[str, Any] | None:
        def fn():
            data = self.http.get(f"{HGNC}/{symbol.upper()}", headers={"Accept": "application/json"})
            docs = ((data or {}).get("response") or {}).get("docs") or []
            if not docs:
                return None
            d = docs[0]
            return {"hgnc_id": d.get("hgnc_id"), "symbol": d.get("symbol"), "mgd_id": d.get("mgd_id") or [],
                    "uniprot": (d.get("uniprot_ids") or [None])[0]}
        return self._cached(f"hgnc:{symbol.upper()}", fn)

    def mesh(self, name: str) -> dict[str, Any] | None:
        def fn():
            for match in ("exact", "contains"):
                data = self.http.get(MESH, params={"label": name, "match": match, "limit": 3})
                if data:
                    top = data[0]
                    return {"id": "MESH:" + top["resource"].rsplit("/", 1)[-1], "label": top["label"],
                            "exact": match == "exact"}
            return None
        return self._cached(f"mesh:{name.lower()}", fn)

    def ols(self, name: str, ontology: str) -> dict[str, Any] | None:
        def fn():
            data = self.http.get(OLS, params={"q": name, "ontology": ontology, "exact": "true", "rows": 1})
            docs = ((data or {}).get("response") or {}).get("docs") or []
            if docs:
                return {"id": docs[0].get("obo_id"), "label": docs[0].get("label"), "exact": True}
            return None
        return self._cached(f"ols:{ontology}:{name.lower()}", fn)

    # ------------------------------------------------------------ ana giriş

    def normalize(self, e: ExtractedEntity, default_species: str | None = None) -> Entity:
        species = normalize_species(e.species) or (normalize_species(default_species) if e.type == "Gene" else None)
        name = e.name or e.symbol or e.measure or "unknown"
        ent = Entity(type=e.type, name=name, species=species if e.type == "Gene" else None, symbol=e.symbol,
                     intervention_type=e.intervention_type, dose=e.dose, schedule=e.schedule,
                     severity=e.severity, measure=e.measure, direction=e.direction)
        if e.type == "Outcome" and e.measure and not e.name:
            ent.name = e.measure

        if e.type == "Gene":
            self._gene(ent)
        elif e.type == "Disease":
            hit = self.mesh(name)
            if hit and hit.get("exact"):
                ent.xref, ent.name, ent.unresolved = hit["id"], hit["label"], False
            elif hit:
                ent.match_suggestion = f"{hit['label']} ({hit['id']})"
        elif e.type == "CellType":
            hit = self.ols(name, "cl")
            if hit and hit.get("id"):
                ent.xref, ent.name, ent.unresolved = hit["id"], hit["label"], False
        elif e.type == "Intervention" and (e.intervention_type or "").lower() == "drug":
            hit = self.ols(name, "chebi")
            if hit and hit.get("id"):
                ent.xref, ent.unresolved = hit["id"], False
        elif e.type in ("Outcome", "Model", "Intervention"):
            ent.unresolved = False  # serbest metin kimlik (SPEC: "serbest")

        if ent.unresolved and not ent.match_suggestion:
            ent.match_suggestion = self.fuzzy_suggestion(ent)
        return ent

    def _gene(self, ent: Entity) -> None:
        symbol = (ent.symbol or ent.name).strip()
        species = ent.species or "unknown"
        hit = self._uniprot(symbol, species) if species in TAXON else None
        if hit:
            ent.symbol = hit.get("symbol") or symbol
            ent.uniprot_id = hit.get("accession")
            if species == "mouse" and hit.get("mgi"):
                ent.xref = hit["mgi"]
            elif species == "human" and hit.get("hgnc"):
                ent.xref = hit["hgnc"]
            elif species == "rat" and hit.get("rgd"):
                ent.xref = "RGD:" + hit["rgd"].removeprefix("RGD:")
            else:
                ent.xref = f"UniProt:{hit['accession']}"
            ent.unresolved = False
        elif species == "human":
            h = self.hgnc(symbol)
            if h:
                ent.symbol, ent.xref, ent.uniprot_id, ent.unresolved = h["symbol"], h["hgnc_id"], h.get("uniprot"), False

    def fuzzy_suggestion(self, ent: Entity) -> str | None:
        candidates = self.known_names.get(ent.type) or []
        label = ent.symbol or ent.name
        if not candidates:
            return None
        best = process.extractOne(label, candidates, scorer=fuzz.WRatio)
        if best and best[1] >= FUZZY_THRESHOLD and best[0].lower() != label.lower():
            return best[0]
        return None

    # ------------------------------------------------------------ ortologlar

    def orthologs(self, genes: list[Entity]) -> list[tuple[str, str, str]]:
        """(fare_gen_key, insan_gen_key, yöntem). HGNC mgd_id varsa onu, yoksa sembol eşleşmesini kullanır."""
        mice = [g for g in genes if g.species == "mouse"]
        humans = [g for g in genes if g.species == "human"]
        pairs = []
        for h in humans:
            info = self.hgnc(h.symbol or h.name) if not self.offline or f"hgnc:{(h.symbol or h.name).upper()}" in self.cache else None
            mgd_ids = set((info or {}).get("mgd_id") or [])
            for m in mice:
                if m.xref and m.xref in mgd_ids:
                    pairs.append((m.key, h.key, "HGNC"))
                elif (m.symbol or m.name).upper() == (h.symbol or h.name).upper():
                    pairs.append((m.key, h.key, "symbol_match"))
        return pairs
