"""Graf üzerinde kelime araması: birkaç kelime yaz, eşleşen ilişkileri (ve varlıkları) sırala.

Eşleşme sözcük başı üzerinden yapılır; Türkçe ekler için "üniversite" araması "Üniversitesi"ni,
"ankara" araması "Ankara'da"yı bulur. Bir kelime; özne/nesne adında, ilişki adında, varlık türünde,
açıklamada veya kaynak cümlede geçebilir — ad eşleşmeleri daha yüksek puan alır.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

STOPWORDS = {
    "ve", "ile", "veya", "ya", "da", "de", "bir", "bu", "şu", "o", "için", "gibi", "mi", "mı", "mu", "mü", "ne",
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for", "with", "is", "are", "by",
}
WORD = re.compile(r"\w+", re.UNICODE)

# alan → ağırlık
W_NAME, W_PRED, W_TYPE, W_SENT, W_DESC = 3.0, 2.0, 1.0, 1.0, 0.5


def fold(text: str) -> str:
    """Küçük harf + aksansız: "ÜNİVERSİTESİ", "universitesi"; "IŞIK", "isik" aynı bulunur."""
    t = unicodedata.normalize("NFKD", (text or "").casefold().replace("ı", "i"))
    return "".join(c for c in t if not unicodedata.combining(c))


STOPWORDS = {fold(w) for w in STOPWORDS}


def words(text: str) -> list[str]:
    return WORD.findall(fold(text).replace("_", " "))


def tokenize(query: str) -> list[str]:
    toks, seen = [], set()
    for w in words(query):
        if w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        toks.append(w)
    return toks


def _word_match(tok: str, w: str) -> bool:
    if w.startswith(tok):
        return True
    # "üniversitesi" araması "üniversite" kelimesini de bulsun (ek farkı)
    n = max(4, len(tok) - 3)
    return len(w) >= n and len(tok) >= n and w[:n] == tok[:n]


def _match(tok: str, ws: list[str]) -> bool:
    return any(_word_match(tok, w) for w in ws)


def spans(text: str, toks: list[str]) -> list[tuple[int, int]]:
    """Metinde vurgulanacak kelime aralıkları."""
    out = []
    if not toks:
        return out
    for m in WORD.finditer(text or ""):
        w = fold(m.group())
        if any(_word_match(t, w) for t in toks):
            out.append((m.start(), m.end()))
    return out


def _origin(sources: list[dict[str, Any]], wikidata_source: str, meta_suffix: str) -> str:
    if sources and all(s["doc_id"] == wikidata_source for s in sources):
        return "wikidata"
    if sources and all(s["doc_id"].endswith(meta_suffix) for s in sources):
        return "metadata"
    return "document"


def search(graph: dict[str, Any], query: str = "", *, entity: str | None = None, origin: str = "",
           wikidata_source: str = "wikidata", meta_suffix: str = "#meta", limit: int = 100) -> dict[str, Any]:
    """İlişkileri ve varlıkları puanlayarak döndürür.

    entity verilirse o varlığın tüm ilişkileri listelenir (sorgu kelimeleri yine süzgeç olarak uygulanır).
    Tüm kelimeleri içeren ilişki yoksa en az birini içerenler gösterilir (partial=True).
    """
    toks = tokenize(query)
    ents = graph.get("entities", {})
    ent_words = {k: (words(e["name"]), words(e.get("type", "")), words(e.get("description") or ""))
                 for k, e in ents.items()}

    scored: list[tuple[int, float, int, dict]] = []
    for r in graph.get("relations", {}).values():
        if entity and entity not in (r["subject"], r["object"]):
            continue
        org = _origin(r["sources"], wikidata_source, meta_suffix)
        if origin and org != origin:
            continue
        s_name, s_type, s_desc = ent_words.get(r["subject"], ([], [], []))
        o_name, o_type, o_desc = ent_words.get(r["object"], ([], [], []))
        fields = [
            (W_NAME, s_name + o_name), (W_PRED, words(r["predicate"])), (W_TYPE, s_type + o_type),
            (W_SENT, [w for s in r["sources"] for w in words(s.get("sentence", ""))]), (W_DESC, s_desc + o_desc),
        ]
        hits, score = 0, 0.0
        for t in toks:
            best = next((wt for wt, ws in fields if _match(t, ws)), 0.0)
            if best:
                hits += 1
                score += best
        if toks and not hits:
            continue
        scored.append((hits, score, len(r["sources"]), {**r, "origin": org}))

    partial = False
    if toks:
        full = [x for x in scored if x[0] == len(toks)]
        if full or len(toks) == 1:
            scored = full
        else:
            partial = True
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], ents.get(x[3]["subject"], {}).get("name", "")))
    relations = [x[3] for x in scored]

    entities = []
    if toks and not entity:
        for k, (nw, tw, dw) in ent_words.items():
            hits = sum(1 for t in toks if _match(t, nw) or _match(t, tw))
            if hits:
                name_hits = sum(1 for t in toks if _match(t, nw))
                entities.append((hits, name_hits, k))
        entities.sort(key=lambda x: (-x[0], -x[1], ents[x[2]]["name"]))
        if any(h == len(toks) for h, _, _ in entities):
            entities = [x for x in entities if x[0] == len(toks)]

    degree: dict[str, int] = {}
    for r in graph.get("relations", {}).values():
        degree[r["subject"]] = degree.get(r["subject"], 0) + 1
        degree[r["object"]] = degree.get(r["object"], 0) + 1

    return {
        "tokens": toks,
        "partial": partial,
        "total": len(relations),
        "relations": relations[:limit],
        "entities": [{**ents[k], "key": k, "degree": degree.get(k, 0)} for _, _, k in entities[:12]],
    }
