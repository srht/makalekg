"""OAI-PMH ile kurumsal arşivlerden (DSpace vb.) Dublin Core metadata çekme.

- Identify / ListSets / ListRecords (metadataPrefix=oai_dc), resumptionToken ile sayfalama.
- Her kayıt projeye "belge" olarak eklenir (başlık + yazarlar + konular + özet; istenirse tam metin PDF).
  Aynı OAI kimliği tekrar çekilirse belge güncellenir, çift kayıt oluşmaz.
- Metadata grafı: dc:creator / dc:contributor / dc:subject / dc:publisher alanlarından dil modeli olmadan
  doğrudan varlık ve ilişki üretilir (Dublin Core / FOAF / SKOS URI'leriyle). Bu katkılar "<belge>#meta"
  kaynağıyla tutulur; belge yeniden işlense de silinmez, belge silinince birlikte silinir.
- Tam metin: DSpace'in ORE (metadataPrefix=ore) kaydındaki PDF bağlantısı denenir.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx
from lxml import etree

from ..http import Http
from .documents import extract_text
from .ontology import EntityType, Ontology, RelationType
from .store import Project

log = logging.getLogger(__name__)

NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
      "dc": "http://purl.org/dc/elements/1.1/", "atom": "http://www.w3.org/2005/Atom",
      "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}
DC_FIELDS = ("title", "creator", "contributor", "subject", "description", "publisher", "date", "type",
             "identifier", "language", "rights", "source", "relation", "coverage", "format")
META_SUFFIX = "#meta"
MAX_PDF_BYTES = 60 * 1024 * 1024

# Metadata grafı için türler ve ilişkiler (projede aynı adlı tür varsa o kullanılır)
META_TYPES = {
    "publication": EntityType(name="Yayın", description="Makale, tez, kitap, rapor gibi yayın (arşiv kaydı)",
                              uri="http://purl.org/dc/terms/BibliographicResource"),
    "person": EntityType(name="Kişi", description="Gerçek bir kişi", uri="http://xmlns.com/foaf/0.1/Person"),
    "subject": EntityType(name="Konu", description="Anahtar kelime / konu başlığı",
                          uri="http://www.w3.org/2004/02/skos/core#Concept"),
    "org": EntityType(name="Kurum", description="Yayıncı veya kurum", uri="http://xmlns.com/foaf/0.1/Organization"),
}
META_RELATIONS = {
    "creator": ("YAZARI", "publication", "person", "http://purl.org/dc/elements/1.1/creator", "yayının yazarı"),
    "contributor": ("KATKIDA_BULUNAN", "publication", "person", "http://purl.org/dc/elements/1.1/contributor",
                    "danışman, editör vb. katkıda bulunan"),
    "subject": ("KONUSU", "publication", "subject", "http://purl.org/dc/elements/1.1/subject", "yayının konusu"),
    "publisher": ("YAYINCISI", "publication", "org", "http://purl.org/dc/elements/1.1/publisher", "yayıncı kurum"),
}


class OaiError(RuntimeError):
    pass


@dataclass
class Record:
    identifier: str
    datestamp: str
    deleted: bool = False
    dc: dict[str, list[str]] = field(default_factory=dict)
    sets: list[str] = field(default_factory=list)

    def first(self, key: str) -> str:
        return (self.dc.get(key) or [""])[0]

    @property
    def title(self) -> str:
        return self.first("title") or self.identifier

    @property
    def landing_url(self) -> str | None:
        for v in self.dc.get("identifier", []):
            if v.startswith("http"):
                return v
        return None


def check_base_url(base: str) -> str:
    base = base.strip()
    u = urlparse(base)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise OaiError("OAI adresi http:// veya https:// ile başlamalı (ör. https://acikerisim.ornek.edu.tr/oai/request)")
    return base


class OaiClient:
    def __init__(self, base_url: str, http: Http | None = None):
        self.base = check_base_url(base_url)
        self.http = http or Http(min_interval=0.5)

    def _get(self, params: dict[str, Any]) -> etree._Element:
        try:
            text = self.http.get(self.base, params=params, as_text=True, use_cache=False)
        except ConnectionError as exc:
            raise OaiError(f"arşive ulaşılamadı: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise OaiError(f"arşiv {exc.response.status_code} yanıtı verdi; adres doğru mu?") from exc
        if not text:
            raise OaiError("arşiv boş yanıt verdi; OAI adresi doğru mu? (DSpace'te genelde .../oai/request)")
        try:
            root = etree.fromstring(text.encode("utf-8") if isinstance(text, str) else text,
                                    parser=etree.XMLParser(recover=True, huge_tree=True))
        except etree.XMLSyntaxError as exc:
            raise OaiError("yanıt XML değil; bu adres bir OAI-PMH uç noktası olmayabilir") from exc
        if root is None or etree.QName(root).localname != "OAI-PMH":
            raise OaiError("yanıt OAI-PMH değil; adres doğru mu? (DSpace'te genelde .../oai/request)")
        err = root.find("oai:error", NS)
        if err is not None:
            code = err.get("code")
            if code == "noRecordsMatch":
                return root
            raise OaiError(f"arşiv hatası ({code}): {(err.text or '').strip()}")
        return root

    def identify(self) -> dict[str, str]:
        root = self._get({"verb": "Identify"})
        ident = root.find("oai:Identify", NS)
        if ident is None:
            raise OaiError("Identify yanıtı yok")
        get = lambda tag: (ident.findtext(f"oai:{tag}", default="", namespaces=NS) or "").strip()
        return {"name": get("repositoryName"), "version": get("protocolVersion"),
                "earliest": get("earliestDatestamp"), "granularity": get("granularity")}

    def list_sets(self, limit: int = 2000) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        params: dict[str, Any] = {"verb": "ListSets"}
        while len(out) < limit:
            root = self._get(params)
            for s in root.iterfind(".//oai:set", NS):
                out.append({"spec": s.findtext("oai:setSpec", default="", namespaces=NS),
                            "name": (s.findtext("oai:setName", default="", namespaces=NS) or "").strip()})
            token = root.findtext(".//oai:resumptionToken", namespaces=NS)
            if not token or not token.strip():
                break
            params = {"verb": "ListSets", "resumptionToken": token.strip()}
        return out[:limit]

    def list_records(self, set_spec: str = "", date_from: str = "", date_until: str = "",
                     limit: int = 50) -> Iterator[Record]:
        params: dict[str, Any] = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
        if set_spec:
            params["set"] = set_spec
        if date_from:
            params["from"] = date_from
        if date_until:
            params["until"] = date_until
        n = 0
        while True:
            root = self._get(params)
            for rec in root.iterfind(".//oai:record", NS):
                r = parse_record(rec)
                if r is None:
                    continue
                yield r
                n += 1
                if n >= limit:
                    return
            token = root.findtext(".//oai:resumptionToken", namespaces=NS)
            if not token or not token.strip():
                return
            log.info("OAI: %d kayıt alındı, sonraki sayfa isteniyor", n)
            params = {"verb": "ListRecords", "resumptionToken": token.strip()}

    def pdf_links(self, identifier: str) -> list[str]:
        """DSpace ORE kaydından PDF bitstream bağlantıları (desteklenmiyorsa boş liste)."""
        try:
            root = self._get({"verb": "GetRecord", "metadataPrefix": "ore", "identifier": identifier})
        except OaiError:
            return []
        links = []
        for link in root.iterfind(".//atom:link", NS):
            if link.get("rel", "").endswith("aggregates") and "pdf" in (link.get("type") or "").lower():
                links.append(link.get("href"))
        for desc in root.iterfind(".//rdf:Description", NS):  # bazı sürümler RDF ile verir
            about = desc.get(f"{{{NS['rdf']}}}about") or ""
            if about.lower().endswith(".pdf") or "/bitstream/" in about:
                links.append(about)
        return list(dict.fromkeys(l for l in links if l and l.startswith("http")))

    def download_pdf(self, url: str) -> bytes | None:
        try:
            with self.http.client.stream("GET", url, timeout=120) as r:
                if r.status_code != 200:
                    return None
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > MAX_PDF_BYTES:
                        return None
                data = bytes(buf)
                return data if data.startswith(b"%PDF") else None
        except httpx.HTTPError:
            return None


def parse_record(rec: etree._Element) -> Record | None:
    header = rec.find("oai:header", NS)
    if header is None:
        return None
    r = Record(identifier=(header.findtext("oai:identifier", default="", namespaces=NS) or "").strip(),
               datestamp=(header.findtext("oai:datestamp", default="", namespaces=NS) or "").strip(),
               deleted=header.get("status") == "deleted",
               sets=[s.text or "" for s in header.iterfind("oai:setSpec", NS)])
    dc = rec.find(".//oai_dc:dc", NS)
    if dc is not None:
        for f in DC_FIELDS:
            vals = [re.sub(r"\s+", " ", (e.text or "")).strip() for e in dc.iterfind(f"dc:{f}", NS)]
            vals = [v for v in vals if v]
            if vals:
                r.dc[f] = vals
    return r


# ---------------------------------------------------------------- projeye aktarma


def person_name(value: str) -> str:
    """DSpace 'Soyad, Ad' biçimini 'Ad Soyad' yapar; belgelerden çıkan adlarla birleşebilsin diye."""
    value = re.sub(r"\s*\(.*?\)\s*$", "", value).strip()  # "(Danışman)" gibi ekler
    if value.count(",") == 1:
        last, first = (p.strip() for p in value.split(","))
        if first and last and not any(ch.isdigit() for ch in first):
            return f"{first} {last}"
    return value


def split_subjects(values: list[str]) -> list[str]:
    out = []
    for v in values:
        for part in re.split(r"\s*[;|]\s*", v):
            part = part.strip(" .")
            if part and len(part) <= 150:
                out.append(part)
    return list(dict.fromkeys(out))


def record_text(r: Record) -> str:
    lines = [f"Başlık: {r.title}"]
    if r.dc.get("creator"):
        lines.append("Yazarlar: " + "; ".join(person_name(c) for c in r.dc["creator"]))
    if r.dc.get("contributor"):
        lines.append("Katkıda bulunanlar: " + "; ".join(person_name(c) for c in r.dc["contributor"]))
    for key, label in (("date", "Tarih"), ("type", "Tür"), ("publisher", "Yayıncı"), ("language", "Dil")):
        if r.dc.get(key):
            lines.append(f"{label}: " + "; ".join(r.dc[key]))
    if r.dc.get("subject"):
        lines.append("Konular: " + "; ".join(split_subjects(r.dc["subject"])))
    for d in r.dc.get("description", []):
        lines.append("")
        lines.append(d)
    return "\n".join(lines)


def ensure_metadata_ontology(project: Project) -> dict[str, str]:
    """Metadata grafı için gereken türleri/ilişkileri ontolojiye ekler (aynı adlı varsa onu kullanır)."""
    onto = project.ontology()
    names = {t.name.casefold(): t.name for t in onto.entity_types}
    ents = list(onto.entity_types)
    type_name = {}
    for key, t in META_TYPES.items():
        if t.name.casefold() in names:
            type_name[key] = names[t.name.casefold()]
        else:
            ents.append(t)
            type_name[key] = t.name
    rels = list(onto.relation_types)
    rel_names = {r.name for r in rels}
    for _, (name, src, tgt, uri, desc) in META_RELATIONS.items():
        if name not in rel_names:
            rels.append(RelationType(name=name, source=[type_name[src]], target=[type_name[tgt]], uri=uri,
                                     description=desc))
    project.save_ontology(Ontology(entity_types=ents, relation_types=rels, domain=onto.domain))
    return type_name


def metadata_graph(project: Project, doc_id: str, r: Record, type_name: dict[str, str]) -> int:
    pub = r.title[:300]
    entities = [{"type": type_name["publication"], "name": pub, "mention": pub,
                 "description": "; ".join(r.dc.get("type", [])[:1] + r.dc.get("date", [])[:1]) or None}]
    relations = []

    def add(kind: str, field_name: str, raw: str, name: str) -> None:
        rel, _, tgt, _, _ = META_RELATIONS[kind]
        entities.append({"type": type_name[tgt], "name": name, "mention": raw, "description": None})
        relations.append({"subject_key": Project.entity_key(type_name["publication"], pub), "predicate": rel,
                          "object_key": Project.entity_key(type_name[tgt], name),
                          "sentence": f"Metadata (dc:{field_name}): {raw}", "confidence": 1.0})

    for c in r.dc.get("creator", []):
        add("creator", "creator", c, person_name(c))
    for c in r.dc.get("contributor", []):
        add("contributor", "contributor", c, person_name(c))
    for s in split_subjects(r.dc.get("subject", [])):
        add("subject", "subject", s, s)
    for p in r.dc.get("publisher", []):
        add("publisher", "publisher", p, p)
    meta_id = doc_id + META_SUFFIX
    project.remove_contributions(meta_id)
    project.add_extraction(meta_id, entities, relations, [])
    return len(relations)


def harvest(project: Project, base_url: str, set_spec: str = "", date_from: str = "", date_until: str = "",
            limit: int = 50, build_metadata_graph: bool = True, fulltext: bool = False,
            client: OaiClient | None = None) -> dict[str, Any]:
    client = client or OaiClient(base_url)
    info = client.identify()
    log.info("Arşiv: %s (OAI-PMH %s)", info.get("name") or base_url, info.get("version"))
    type_name = ensure_metadata_ontology(project) if build_metadata_graph else {}
    existing = {(d.get("source") or {}).get("identifier"): d for d in project.documents() if d.get("source")}
    added, updated, deleted, n_rel, n_pdf = [], [], 0, 0, 0
    for r in client.list_records(set_spec, date_from, date_until, limit):
        old = existing.get(r.identifier)
        if r.deleted:
            if old:
                project.delete_document(old["id"])
                deleted += 1
            continue
        text, filename = record_text(r), None
        data = text.encode("utf-8")
        if fulltext:
            for url in client.pdf_links(r.identifier)[:2]:
                pdf = client.download_pdf(url)
                if pdf:
                    try:
                        full = extract_text("x.pdf", pdf)
                    except Exception:
                        continue
                    if len(full.strip()) >= 200:  # taranmış/boş PDF değilse
                        text = record_text(r) + "\n\n--- TAM METİN ---\n\n" + full
                        data, filename = pdf, "tam_metin.pdf"
                        n_pdf += 1
                        break
        source = {"type": "oai", "base": base_url, "identifier": r.identifier, "datestamp": r.datestamp,
                  "url": r.landing_url, "dc": r.dc, "sets": r.sets, "fulltext": bool(filename)}
        title = r.title[:150]
        if old:
            changed = old.get("source", {}).get("datestamp") != r.datestamp or (fulltext and not old["source"].get("fulltext"))
            if changed:
                project.replace_document_text(old["id"], text, source=source, filename=title)
                updated.append(old["id"])
            doc_id = old["id"]
        else:
            doc = project.add_document(f"{title}.txt" if not filename else f"{title}.pdf", data, text, source=source)
            doc_id = doc["id"]
            added.append(doc_id)
        if build_metadata_graph:
            n_rel += metadata_graph(project, doc_id, r, type_name)
    project.sync_neo4j()
    log.info("OAI: %d yeni, %d güncellenen, %d silinen kayıt; %d metadata ilişkisi; %d tam metin",
             len(added), len(updated), deleted, n_rel, n_pdf)
    return {"repository": info.get("name"), "added": added, "updated": updated, "deleted": deleted,
            "metadata_relations": n_rel, "fulltext": n_pdf}
