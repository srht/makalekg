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

from ..http import Http, RateLimited
from .store import Project

log = logging.getLogger(__name__)

API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
WIKIDATA_SOURCE = "wikidata"
SAVE_EVERY = 10


class WikidataBusy(RuntimeError):
    """Wikidata istekleri sınırladı; o ana kadarki ilerleme kaydedildi."""


def search(http: Http, name: str, lang: str = "tr") -> list[dict[str, Any]]:
    # maxlag: sunucu yoğunsa Wikidata bekletir (Wikimedia'nın önerdiği nazik kullanım)
    base = {"action": "wbsearchentities", "search": name[:200], "uselang": lang, "type": "item", "limit": 3,
            "format": "json", "maxlag": 5}
    data = http.get(API, params={**base, "language": lang})
    hits = (data or {}).get("search") or []
    if not hits and lang != "en":
        data = http.get(API, params={**base, "language": "en"})
        hits = (data or {}).get("search") or []
    return [{"qid": h["id"], "label": h.get("label") or h["id"], "description": h.get("description") or "",
             "url": f"https://www.wikidata.org/wiki/{h['id']}"} for h in hits]


def link_entities(project: Project, http: Http | None = None, lang: str = "tr", relink: bool = False) -> dict[str, int]:
    """Varlıkları Wikidata ile eşler. İlerleme düzenli kaydedilir; Wikidata sınırlarsa durur ve
    tekrar çalıştırıldığında kaldığı yerden devam eder (aranmış adlar atlanır)."""
    http = http or Http()
    todo = [k for k, e in project.graph()["entities"].items()
            if relink or not (e.get("wikidata") or e.get("wikidata_rejected") or e.get("wikidata_searched"))]
    total_done = len(project.graph()["entities"]) - len(todo)
    results: dict[str, list[dict[str, Any]]] = {}
    linked = failed = 0

    def flush() -> None:
        with project.lock:
            fresh = project.graph()
            for key, hits in results.items():
                e = fresh["entities"].get(key)
                if e is None:  # arada silinmiş
                    continue
                e["wikidata_searched"] = True
                if hits:
                    e["wikidata"] = hits[0]
                    e["wikidata_alternatives"] = hits[1:]
            project._save_graph(fresh)
        results.clear()

    names = {k: e["name"] for k, e in project.graph()["entities"].items()}
    try:
        for i, key in enumerate(todo, 1):
            try:
                hits = search(http, names[key], lang)
            except RateLimited:
                raise
            except ConnectionError as exc:
                log.warning("Wikidata araması başarısız (%s): %s", names[key], exc)
                failed += 1
                continue
            results[key] = hits
            linked += bool(hits)
            if i % SAVE_EVERY == 0:
                flush()
                log.info("Wikidata eşleme: %d / %d", i, len(todo))
    except RateLimited as exc:
        flush()
        raise WikidataBusy(
            f"Wikidata çok fazla istek geldiğini söyledi ve geçici olarak sınırladı. Şu ana kadar {linked} varlık "
            f"eşlendi ve kaydedildi. Birkaç dakika sonra aynı düğmeye tekrar bas; kaldığı yerden devam eder. ({exc})")
    flush()
    log.info("Wikidata eşleme: %d yeni eşleme, %d önceden yapılmıştı, %d hata", linked, total_done, failed)
    return {"linked": linked, "skipped": total_done, "failed": failed}


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
            e["wikidata_searched"] = True
        else:
            alts = [e.get("wikidata")] + (e.get("wikidata_alternatives") or [])
            pick = next((a for a in alts if a and a["qid"] == qid), None)
            if pick:
                e["wikidata"] = pick
                e.pop("wikidata_rejected", None)
        project._save_graph(g)


def enrich(project: Project, http: Http | None = None, lang: str = "tr", batch: int = 50) -> dict[str, int]:
    """Eşlenmiş varlıklar arasındaki doğrudan Wikidata ifadelerini ilişki olarak ekler (yeniden çalıştırılabilir).
    Sorgular POST ile, küçük partiler halinde ve aralıklı gönderilir; tamamlanan partiler önbelleğe alınır."""
    http = http or Http()
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
        try:
            data = http.get(SPARQL, params={"query": query, "format": "json"}, method="POST",
                            headers={"Accept": "application/sparql-results+json"})
        except RateLimited as exc:
            raise WikidataBusy(
                "Wikidata sorgu servisi çok fazla istek geldiğini söyledi ve geçici olarak sınırladı. Birkaç dakika "
                f"sonra tekrar bas; tamamlanan kısımlar önbellekte, baştan sorulmaz. ({exc})")
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
