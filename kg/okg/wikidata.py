"""Açık knowledge graph bağlantısı: Wikidata.

1. Eşleme (linking): her varlık adı Wikidata'da aranır; en iyi aday kaydedilir (QID, etiket, açıklama).
   Kullanıcı yanlış eşlemeyi kaldırabilir.
2. Zenginleştirme (enrichment): eşlenmiş varlık çiftleri arasında Wikidata'da doğrudan bir ifade varsa
   (ör. Türkiye —üyesi→ NATO), grafa "wikidata" kaynaklı ilişki olarak eklenir. Bu ilişkiler belgelerden
   çıkarılanlardan ayrı tutulur ve grafikte farklı görünür; belge kanıtı sayılmaz.
"""
from __future__ import annotations

import logging
from typing import Any

from ..http import Http
from .store import Project

log = logging.getLogger(__name__)

API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_SOURCE = "wikidata"


def search(http: Http, name: str, lang: str = "tr") -> list[dict[str, Any]]:
    data = http.get(API, params={"action": "wbsearchentities", "search": name[:200], "language": lang,
                                 "uselang": lang, "type": "item", "limit": 3, "format": "json"})
    hits = (data or {}).get("search") or []
    if not hits and lang != "en":
        data = http.get(API, params={"action": "wbsearchentities", "search": name[:200], "language": "en",
                                     "uselang": lang, "type": "item", "limit": 3, "format": "json"})
        hits = (data or {}).get("search") or []
    return [{"qid": h["id"], "label": h.get("label") or h["id"], "description": h.get("description") or "",
             "url": f"https://www.wikidata.org/wiki/{h['id']}"} for h in hits]


def link_entities(project: Project, http: Http | None = None, lang: str = "tr", relink: bool = False) -> dict[str, int]:
    http = http or Http(min_interval=0.2)
    g = project.graph()
    linked = skipped = failed = 0
    for key, e in g["entities"].items():
        if e.get("wikidata") and not relink:
            skipped += 1
            continue
        if e.get("wikidata_rejected"):
            skipped += 1
            continue
        try:
            hits = search(http, e["name"])
        except ConnectionError as exc:
            log.warning("Wikidata'ya ulaşılamadı: %s", exc)
            failed += 1
            continue
        if hits:
            e["wikidata"] = hits[0]
            e["wikidata_alternatives"] = hits[1:]
            linked += 1
    with project.lock:
        fresh = project.graph()
        for key, e in g["entities"].items():  # arada işlenen belgeleri ezmemek için yalnızca alanları aktar
            if key in fresh["entities"]:
                for f in ("wikidata", "wikidata_alternatives"):
                    if f in e:
                        fresh["entities"][key][f] = e[f]
        project._save_graph(fresh)
    log.info("Wikidata eşleme: %d yeni, %d atlandı, %d hata", linked, skipped, failed)
    return {"linked": linked, "skipped": skipped, "failed": failed}


def set_link(project: Project, entity_key: str, qid: str | None) -> None:
    """Kullanıcı düzeltmesi: eşlemeyi kaldır (qid=None) veya alternatiflerden birini seç."""
    with project.lock:
        g = project.graph()
        e = g["entities"].get(entity_key)
        if e is None:
            return
        if qid is None:
            e.pop("wikidata", None)
            e["wikidata_rejected"] = True
        else:
            alts = [e.get("wikidata")] + (e.get("wikidata_alternatives") or [])
            pick = next((a for a in alts if a and a["qid"] == qid), None)
            if pick:
                e["wikidata"] = pick
                e.pop("wikidata_rejected", None)
        project._save_graph(g)


def enrich(project: Project, http: Http | None = None, lang: str = "tr", batch: int = 150) -> dict[str, int]:
    """Eşlenmiş varlıklar arasındaki doğrudan Wikidata ifadelerini ilişki olarak ekler (yeniden çalıştırılabilir)."""
    http = http or Http(min_interval=1.0)
    g = project.graph()
    qid_to_key = {e["wikidata"]["qid"]: k for k, e in g["entities"].items() if e.get("wikidata")}
    qids = sorted(qid_to_key)
    rows: list[dict[str, Any]] = []
    for i in range(0, len(qids), batch):
        part = qids[i:i + batch]
        values = " ".join(f"wd:{q}" for q in part)
        all_values = " ".join(f"wd:{q}" for q in qids)
        query = f"""SELECT ?a ?b ?prop ?propLabel WHERE {{
  VALUES ?a {{ {values} }} VALUES ?b {{ {all_values} }}
  ?a ?p ?b . ?prop wikibase:directClaim ?p .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang},en". }}
}} LIMIT 5000"""
        data = http.get(SPARQL, params={"query": query, "format": "json"},
                        headers={"Accept": "application/sparql-results+json"})
        for b in ((data or {}).get("results") or {}).get("bindings") or []:
            rows.append({"a": b["a"]["value"].rsplit("/", 1)[-1], "b": b["b"]["value"].rsplit("/", 1)[-1],
                         "pid": b["prop"]["value"].rsplit("/", 1)[-1], "label": b["propLabel"]["value"]})
    rels = []
    for r in rows:
        if r["a"] == r["b"] or r["a"] not in qid_to_key or r["b"] not in qid_to_key:
            continue
        sk, ok = qid_to_key[r["a"]], qid_to_key[r["b"]]
        rels.append({"subject_key": sk, "predicate": f"WD:{r['label']}", "object_key": ok,
                     "sentence": f"Wikidata: {g['entities'][sk]['name']} — {r['label']} ({r['pid']}) → "
                                 f"{g['entities'][ok]['name']} · https://www.wikidata.org/wiki/{r['a']}",
                     "confidence": 1.0})
    project.remove_contributions(WIKIDATA_SOURCE)
    project.add_extraction(WIKIDATA_SOURCE, [], rels, [])
    log.info("Wikidata zenginleştirme: %d varlık arasında %d ilişki eklendi", len(qids), len(rels))
    return {"linked_entities": len(qids), "relations": len(rels)}
