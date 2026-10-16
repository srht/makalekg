"""Hazır (standart) ontolojileri içe aktarma: OWL / RDFS dosyaları (Turtle, RDF/XML, JSON-LD, N-Triples).

Büyük ontolojilerden (ör. schema.org ~800 sınıf) her şeyi almak istemleri şişirir; bu yüzden içe aktarma
iki adımlıdır: dosyadaki sınıflar ve özellikler listelenir, kullanıcı istediklerini seçer.
Seçilen her tür/ilişki özgün URI'sini saklar; graf dışa aktarılırken bu eşleme korunur.
"""
from __future__ import annotations

from typing import Any

from .ontology import EntityType, Ontology, RelationType

FORMATS = {"ttl": "turtle", "owl": "xml", "rdf": "xml", "xml": "xml", "jsonld": "json-ld", "json": "json-ld",
           "nt": "nt", "n3": "n3"}


def _label(g, node, langs=("tr", "en")) -> str:
    from rdflib import RDFS, Literal
    from rdflib.namespace import SKOS

    labels = [o for o in g.objects(node, RDFS.label) if isinstance(o, Literal)]
    labels += [o for o in g.objects(node, SKOS.prefLabel) if isinstance(o, Literal)]
    for lang in langs:
        for l in labels:
            if l.language == lang:
                return str(l)
    for l in labels:
        if not l.language:
            return str(l)
    s = str(node)
    return s.rsplit("#", 1)[-1].rsplit("/", 1)[-1] if ("#" in s or "/" in s) else s


def _comment(g, node, langs=("tr", "en")) -> str:
    from rdflib import RDFS, Literal

    comments = [o for o in g.objects(node, RDFS.comment) if isinstance(o, Literal)]
    for lang in langs:
        for c in comments:
            if c.language == lang:
                return str(c)[:300]
    return str(comments[0])[:300] if comments else ""


def parse_ontology_file(filename: str, data: bytes) -> dict[str, Any]:
    """Dosyadaki sınıfları ve nesne özelliklerini listeler (seçim için)."""
    from rdflib import OWL, RDF, RDFS, Graph, URIRef

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    g = Graph()
    fmt = FORMATS.get(ext)
    try:
        g.parse(data=data, format=fmt) if fmt else g.parse(data=data)
    except Exception as exc:
        raise ValueError(f"ontoloji dosyası okunamadı ({ext or 'bilinmeyen biçim'}): {exc}") from exc

    classes = {c for c in g.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    classes |= {c for c in g.subjects(RDF.type, RDFS.Class) if isinstance(c, URIRef)}
    props = {p for p in g.subjects(RDF.type, OWL.ObjectProperty) if isinstance(p, URIRef)}
    props |= {p for p in g.subjects(RDF.type, RDF.Property) if isinstance(p, URIRef)}
    # schema.org gibi ontolojilerde özellik aralığı sınıf olanlar ilişki, literal olanlar nitelik sayılır
    literal_ranges = {"Text", "Number", "Integer", "Float", "Boolean", "Date", "DateTime", "Time", "URL", "string",
                      "integer", "decimal", "boolean", "date", "dateTime", "Literal"}
    SCHEMA_DOMAIN = URIRef("https://schema.org/domainIncludes")
    SCHEMA_RANGE = URIRef("https://schema.org/rangeIncludes")
    SCHEMA_DOMAIN_HTTP = URIRef("http://schema.org/domainIncludes")
    SCHEMA_RANGE_HTTP = URIRef("http://schema.org/rangeIncludes")

    cls_rows = {}
    for c in classes:
        cls_rows[str(c)] = {"uri": str(c), "name": _label(g, c), "description": _comment(g, c), "n_props": 0}
    prop_rows = []
    for p in props:
        domains = set(g.objects(p, RDFS.domain)) | set(g.objects(p, SCHEMA_DOMAIN)) | set(g.objects(p, SCHEMA_DOMAIN_HTTP))
        ranges = set(g.objects(p, RDFS.range)) | set(g.objects(p, SCHEMA_RANGE)) | set(g.objects(p, SCHEMA_RANGE_HTTP))
        range_names = {_label(g, r) for r in ranges}
        if ranges and range_names <= literal_ranges:
            continue  # nitelik (ör. ad, tarih), ilişki değil
        for d in domains:
            if str(d) in cls_rows:
                cls_rows[str(d)]["n_props"] += 1
        prop_rows.append({"uri": str(p), "name": _label(g, p), "description": _comment(g, p),
                          "source": sorted(str(d) for d in domains if str(d) in cls_rows),
                          "target": sorted(str(r) for r in ranges if str(r) in cls_rows)})
    classes_sorted = sorted(cls_rows.values(), key=lambda r: (-r["n_props"], r["name"].lower()))
    return {"classes": classes_sorted, "properties": sorted(prop_rows, key=lambda r: r["name"].lower()),
            "n_triples": len(g)}


def merge_selection(onto: Ontology, parsed: dict[str, Any], class_uris: list[str], prop_uris: list[str]) -> Ontology:
    """Seçilen sınıf ve özellikleri mevcut ontolojiye ekler (ad çakışırsa mevcut olan kalır)."""
    by_uri = {c["uri"]: c for c in parsed["classes"]}
    name_of = {c["uri"]: c["name"] for c in parsed["classes"]}
    existing = {t.name.lower() for t in onto.entity_types}
    ents = list(onto.entity_types)
    for u in class_uris:
        c = by_uri.get(u)
        if c and c["name"].lower() not in existing:
            ents.append(EntityType(name=c["name"], description=c["description"], uri=u))
            existing.add(c["name"].lower())
    selected_names = {name_of[u] for u in class_uris if u in name_of} | {t.name for t in onto.entity_types}
    rels = list(onto.relation_types)
    rel_names = {r.name for r in rels}
    for p in parsed["properties"]:
        if p["uri"] not in prop_uris:
            continue
        rt = RelationType(name=p["name"], description=p["description"], uri=p["uri"],
                          source=[name_of[s] for s in p["source"] if name_of.get(s) in selected_names],
                          target=[name_of[t] for t in p["target"] if name_of.get(t) in selected_names])
        if rt.name not in rel_names:
            rels.append(rt)
            rel_names.add(rt.name)
    return Ontology(entity_types=ents, relation_types=rels, domain=onto.domain).cleaned()
