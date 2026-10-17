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


def _phrase(toks: list[str], ws: list[str]) -> bool:
    """Kelimeler bu sırayla ve yan yana geçiyor mu? ("yapay zeka" → "Yapay Zekâ'nın", "yapay zekası")"""
    n = len(toks)
    return any(all(_word_match(t, w) for t, w in zip(toks, ws[i:i + n])) for i in range(len(ws) - n + 1))


def _name_level(toks: list[str], ws: list[str]) -> int:
    """Bir ad için eşleşme düzeyi: 3 ad = sorgu, 2 ad sorguyu öbek olarak içerir, 1 tüm kelimeler adda, 0 değil."""
    if not toks or not ws:
        return 0
    if _phrase(toks, ws):
        return 3 if len(ws) == len(toks) else 2
    return 1 if all(_match(t, ws) for t in toks) else 0


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
    Her ilişkiye "match" eklenir: phrase (kelimeler yan yana), all (hepsi var, ayrı yerlerde), partial.
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
        # Birden çok kelimede öbek önceliği: "yapay zeka" araması önce adı/cümlesi "yapay zeka" olanları,
        # sonra kelimelerin aynı ad/cümlede geçtiklerini, en son kelimeleri farklı yerlerde içerenleri getirir.
        level = 0
        if len(toks) > 1 and hits == len(toks):
            sents = [words(x.get("sentence", "")) for x in r["sources"]]
            name_lv = max(_name_level(toks, s_name), _name_level(toks, o_name))
            if name_lv >= 2:
                level = 3 + name_lv          # 6: ad tam olarak sorgu, 5: ad öbeği içeriyor
            elif any(_phrase(toks, ws) for ws in [words(r["predicate"])] + sents):
                level = 4                    # öbek ilişki adında / kaynak cümlede
            elif name_lv == 1 or any(all(_match(t, ws) for t in toks) for ws in sents):
                level = 2                    # kelimeler aynı ad ya da cümlede, ama yan yana değil
            else:
                level = 1                    # kelimeler ilişkinin farklı yerlerinde
        match = "partial" if hits < len(toks) else ("phrase" if level >= 4 else "all")
        scored.append((hits, level, score, len(r["sources"]), {**r, "origin": org, "match": match}))

    partial = False
    if toks:
        full = [x for x in scored if x[0] == len(toks)]
        if full or len(toks) == 1:
            scored = full
        else:
            partial = True
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3], ents.get(x[4]["subject"], {}).get("name", "")))
    relations = [x[4] for x in scored]

    entities = []
    if toks and not entity:
        for k, (nw, tw, dw) in ent_words.items():
            hits = sum(1 for t in toks if _match(t, nw) or _match(t, tw))
            if hits:
                entities.append((hits, _name_level(toks, nw), sum(1 for t in toks if _match(t, nw)), k))
        entities.sort(key=lambda x: (-x[0], -x[1], -x[2], ents[x[3]]["name"]))
        if any(h == len(toks) for h, *_ in entities):
            entities = [x for x in entities if x[0] == len(toks)]

    degree: dict[str, int] = {}
    for r in graph.get("relations", {}).values():
        degree[r["subject"]] = degree.get(r["subject"], 0) + 1
        degree[r["object"]] = degree.get(r["object"], 0) + 1

    return {
        "tokens": toks,
        "partial": partial,
        "total": len(relations),
        "n_phrase": sum(1 for r in relations if r["match"] == "phrase"),
        "relations": relations[:limit],
        "entities": [{**ents[k], "key": k, "degree": degree.get(k, 0)} for *_, k in entities[:12]],
    }
