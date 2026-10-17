"""Hızlı varlık modeli (NER) boru hattı.

1. Varlık türleri: alana göre hazır öneriler (ör. elektrik) veya dil modelinden öneri.
2. Eğitim verisi: seçilen N belge (ör. 200 arşiv kaydı) dil modeline gruplar hâlinde gönderilir; model
   varlıkları metinde geçtiği biçimiyle döndürür, sistem tüm geçişlerini karakter aralığıyla işaretler.
   Anthropic API'de Message Batches (yarı fiyat, asenkron) kullanılır; diğer modlarda bir istemde birden çok belge.
3. Eğitim: veri seti GLiNER biçimine çevrilip GLiNER ince ayarlanır (gliner + torch kuruluysa). Kurulu değilse
   bağımlılıksız "sözlük modeli" eğitilir (görülen ifadeleri tanır; yeni adları genelleyemez).
   Ayrılan test belgelerinde dil modeli etiketlerine göre kesinlik/duyarlılık/F1 ölçülür.
4. Uygulama: model kalan belgelerden varlık çıkarır; varlıklar grafa "<belge>#ner" katkısı olarak yazılır ve
   arşiv kaydının Yayın düğümüne BAHSEDER ilişkisiyle bağlanır.

Dosyalar: data/projects/<proje>/ner/{config.json, dataset.jsonl, state.json, metrics.json, gazetteer.json, model/}
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel, Field, create_model

from ..config import env
from ..llm import LLM, is_manual
from .ontology import EntityType, Ontology, RelationType
from .store import Project, norm_name

log = logging.getLogger("kg.ner")

NER_SUFFIX = "#ner"
MENTIONS_PREDICATE = "BAHSEDER"
PUBLICATION_TYPE = "Yayın"
DEFAULT_BASE_MODEL = "urchade/gliner_multi-v2.1"  # çok dilli (Türkçe dahil), Apache-2.0
API_GROUP_CHARS = 14000
MANUAL_GROUP_CHARS = 60000
WINDOW_WORDS = 200  # GLiNER en fazla ~384 alt-kelime görür; Türkçe ekler için pay bırakılır
TOKEN_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")  # GLiNER'in WhitespaceTokenSplitter deseni
SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")

# ---------------------------------------------------------------- alan önerileri

L = tuple[str, str]
DOMAIN_PRESETS: dict[str, dict[str, Any]] = {
    "elektrik": {"title": "Elektrik / elektrik-elektronik mühendisliği", "keys": ["elektrik", "elektronik", "enerji", "güç sistem"],
                 "labels": [
        ("Ekipman", "Elektrik cihazı veya donanım: transformatör, jeneratör, kesici, motor, inverter, röle"),
        ("Bileşen", "Devre elemanı / parça: kondansatör, direnç, diyot, transistör, IGBT, bobin"),
        ("Sistem", "Elektrik sistemi veya şebeke: dağıtım şebekesi, akıllı şebeke, mikro şebeke, SCADA"),
        ("Elektriksel Büyüklük", "Ölçülen/hesaplanan büyüklük: gerilim, akım, güç faktörü, harmonik, frekans, empedans"),
        ("Birim / Değer", "Sayısal değer ve birimi: 154 kV, 50 Hz, 2 MW, 10 kA"),
        ("Malzeme", "Malzeme: bakır, alüminyum, silisyum, SF6, XLPE"),
        ("Enerji Kaynağı", "Üretim kaynağı: güneş, rüzgâr, hidroelektrik, nükleer, doğal gaz"),
        ("Yöntem / Teknik", "Analiz, kontrol veya tasarım yöntemi: PWM, MPPT, PID, yük akışı analizi, sonlu elemanlar"),
        ("Arıza / Olay", "Arıza veya olay türü: kısa devre, aşırı gerilim, kesinti, ark"),
        ("Standart", "Standart veya yönetmelik: IEC 61850, IEEE 1547, TS EN 50160"),
        ("Yazılım / Araç", "Yazılım veya simülasyon aracı: MATLAB/Simulink, PSCAD, DIgSILENT, ETAP"),
        ("Kurum", "Kurum, şirket, üniversite: TEİAŞ, EPDK, TEDAŞ"),
    ]},
    "bilgisayar": {"title": "Bilgisayar bilimi / yazılım / yapay zekâ", "keys": ["bilgisayar", "yazılım", "yapay zeka", "yapay zekâ", "bilişim", "veri"],
                   "labels": [
        ("Yöntem / Algoritma", "Algoritma veya yöntem: derin öğrenme, CNN, transformer, k-means, rastgele orman"),
        ("Model", "Adı geçen belirli model: BERT, GPT-4, ResNet-50, YOLOv8"),
        ("Veri Seti", "Veri seti: ImageNet, MNIST, COCO"),
        ("Görev", "Problem/görev: görüntü sınıflandırma, nesne tespiti, duygu analizi"),
        ("Metrik", "Başarım ölçütü: doğruluk, F1, RMSE, AUC"),
        ("Yazılım / Araç", "Yazılım, kütüphane, dil, platform: Python, TensorFlow, PyTorch, Hadoop"),
        ("Donanım", "Donanım: GPU, FPGA, Raspberry Pi"),
        ("Uygulama Alanı", "Uygulandığı alan: sağlık, tarım, siber güvenlik"),
        ("Kurum", "Kurum, şirket, üniversite"),
    ]},
    "tip": {"title": "Tıp / sağlık", "keys": ["tıp", "sağlık", "hastalık", "klinik", "biyomedikal"],
            "labels": [
        ("Hastalık", "Hastalık, bozukluk, sendrom"), ("Belirti", "Belirti veya bulgu"),
        ("İlaç", "İlaç veya etken madde"), ("Tedavi / İşlem", "Tedavi, ameliyat, girişim"),
        ("Test / Biyobelirteç", "Tanı testi, laboratuvar değeri, biyobelirteç"), ("Anatomi", "Organ, doku, vücut bölgesi"),
        ("Gen / Protein", "Gen, protein, molekül"), ("Popülasyon", "Hasta grubu veya popülasyon"), ("Kurum", "Hastane, kurum"),
    ]},
    "hukuk": {"title": "Hukuk", "keys": ["hukuk", "kanun", "yargı", "mevzuat"],
              "labels": [
        ("Mevzuat", "Kanun, yönetmelik, madde: 6698 sayılı KVKK, TCK m. 141"), ("Mahkeme / Kurum", "Mahkeme, kurul, kurum"),
        ("Karar", "Yargı kararı: Yargıtay 4. HD, E. 2020/123"), ("Hukuki Kavram", "Hukuki kavram: zamanaşımı, tazminat"),
        ("Taraf", "Davacı, davalı, kişi veya kurum"), ("Suç / İhlal", "Suç veya ihlal türü"), ("Tarih", "Tarih"),
    ]},
    "uluslararasi": {"title": "Uluslararası ilişkiler / siyaset", "keys": ["uluslararası", "siyaset", "dış politika", "diplomasi"],
                     "labels": [
        ("Ülke", "Devlet veya ülke"), ("Uluslararası Örgüt", "NATO, BM, AB gibi örgütler"), ("Kişi", "Siyasetçi, diplomat, lider"),
        ("Anlaşma", "Antlaşma, sözleşme, protokol"), ("Olay", "Savaş, kriz, zirve, seçim"), ("Politika / Doktrin", "Politika, doktrin, strateji"),
        ("Bölge", "Coğrafi bölge"), ("Kurum", "Bakanlık, parti, düşünce kuruluşu"),
    ]},
    "egitim": {"title": "Eğitim bilimleri", "keys": ["eğitim", "öğretim", "pedagoji", "öğrenci"],
               "labels": [
        ("Öğrenci / Katılımcı Grubu", "Öğrenci, öğretmen veya katılımcı grubu"), ("Ders / Alan", "Ders veya disiplin"),
        ("Yöntem / Yaklaşım", "Öğretim yöntemi, yaklaşım, model"), ("Ölçme Aracı", "Ölçek, test, anket"),
        ("Kavram / Beceri", "Öğrenme kavramı veya beceri"), ("Kurum", "Okul, üniversite, bakanlık"), ("Program", "Öğretim programı, proje"),
    ]},
    "ekonomi": {"title": "Ekonomi / işletme", "keys": ["ekonomi", "işletme", "finans", "iktisat"],
                "labels": [
        ("Ekonomik Gösterge", "Enflasyon, GSYH, faiz, işsizlik"), ("Kurum / Şirket", "Kurum, banka, şirket"), ("Ülke / Bölge", "Ülke veya bölge"),
        ("Politika", "Para/maliye politikası, düzenleme"), ("Sektör", "Sektör veya endüstri"), ("Finansal Araç", "Hisse, tahvil, kripto para"),
        ("Yöntem / Model", "Ekonometrik yöntem veya model"), ("Dönem / Olay", "Kriz, dönem, olay"),
    ]},
    "insaat": {"title": "İnşaat / yapı", "keys": ["inşaat", "yapı", "deprem", "beton"],
               "labels": [
        ("Yapı Elemanı", "Kolon, kiriş, döşeme, perde"), ("Malzeme", "Beton, çelik, donatı, ahşap"), ("Yapı Türü", "Köprü, bina, tünel, baraj"),
        ("Yöntem / Analiz", "Analiz veya tasarım yöntemi"), ("Büyüklük / Değer", "Dayanım, yük, değer ve birimi"), ("Standart / Yönetmelik", "TBDY 2018, Eurocode"),
        ("Afet / Olay", "Deprem, sel, göçük"), ("Yazılım / Araç", "SAP2000, ETABS"), ("Yer", "Konum"),
    ]},
    "genel": {"title": "Genel (her alan)", "keys": [],
              "labels": [
        ("Kişi", "Gerçek kişi"), ("Kurum", "Kurum, şirket, üniversite"), ("Yer", "Ülke, şehir, bölge"), ("Kavram", "Önemli alan kavramı"),
        ("Yöntem", "Yöntem, teknik, yaklaşım"), ("Eser / Ürün", "Yayın, ürün, yazılım"), ("Olay", "Olay"), ("Tarih", "Tarih"),
    ]},
}


def match_preset(domain: str) -> str | None:
    d = norm_name(domain)
    if d in DOMAIN_PRESETS:
        return d
    for key, p in DOMAIN_PRESETS.items():
        if any(k in d for k in p["keys"]):
            return key
    return None


class SuggestedLabels(BaseModel):
    labels: list[EntityType]


SUGGEST_LABELS_SYSTEM = """Bir alan için adlandırılmış varlık tanıma (NER) etiket kümesi öner.
- 6–14 etiket; her biri o alandaki metinlerde (makale, tez özetleri) sık geçen, METİNDE ADIYLA GEÇEN somut varlık türü olsun.
- Etiket adı kısa ve Türkçe olsun (ör. "Ekipman", "Malzeme"); açıklama tek cümle ve 3–5 örnek içersin.
- Kişi, kurum gibi genel türleri yalnızca alanda gerçekten önemliyse ekle. Soyut "konu/kavram" türlerini en aza indir."""


def suggest_labels(llm: LLM, domain: str, sample: str = "") -> list[EntityType]:
    user = f"Alan: {domain}\n" + (f"\nÖrnek metinler:\n{sample[:8000]}" if sample else "")
    res = llm.complete_json(SUGGEST_LABELS_SYSTEM, user, SuggestedLabels)
    return [EntityType(name=l.name.strip(), description=l.description) for l in res.labels if l.name.strip()]


# ---------------------------------------------------------------- yapılandırma ve veri seti


def ner_dir(project: Project) -> Path:
    d = project.dir / "ner"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_config(project: Project) -> dict[str, Any]:
    cfg = _read(ner_dir(project) / "config.json", {})
    cfg.setdefault("labels", [])
    cfg.setdefault("domain", "")
    cfg.setdefault("base_model", env("GLINER_BASE_MODEL", DEFAULT_BASE_MODEL))
    cfg.setdefault("threshold", 0.5)
    return cfg


DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
    "annotate": {"n": 200, "count_mode": "add", "selection": "random", "source": "all", "keyword": "", "min_chars": 40,
                 "per_request": 10, "max_chars": 0, "batch_api": True, "add_to_graph": True, "model": "", "seed": 7},
    "train": {"kind": "auto", "epochs": 5, "batch_size": 8, "lr": 5e-6, "others_lr": 1e-5, "test_ratio": 0.2},
    "apply": {"scope": "unlabeled", "limit": 0, "selection": "first", "source": "all", "keyword": "", "min_chars": 0,
              "skip_done": False, "batch_docs": 16},
}
_LIMITS = {"n": (1, 20000), "min_chars": (0, 100000), "per_request": (1, 100), "max_chars": (0, 400000), "seed": (0, 10**9),
           "epochs": (1, 100), "batch_size": (1, 128), "lr": (1e-8, 1e-2), "others_lr": (1e-8, 1e-2),
           "test_ratio": (0.0, 0.5), "limit": (0, 10**7), "batch_docs": (1, 512)}
_CHOICES = {"count_mode": ("add", "total"), "selection": ("random", "first", "last", "longest", "shortest"),
            "source": ("all", "archive", "upload"), "kind": ("auto", "gliner", "sozluk"), "scope": ("unlabeled", "all", "new")}


def get_settings(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    saved = cfg.get("settings") or {}
    return {step: {**defaults, **{k: v for k, v in (saved.get(step) or {}).items() if k in defaults}}
            for step, defaults in DEFAULT_SETTINGS.items()}


def clean_settings(step: str, values: dict[str, Any]) -> dict[str, Any]:
    """Formdan gelen değerleri türüne çevirip sınırlar içine alır; bilinmeyen alanları atar."""
    out = {}
    for k, default in DEFAULT_SETTINGS[step].items():
        if k not in values:
            continue
        v = values[k]
        try:
            if isinstance(default, bool):
                v = v in (True, 1, "1", "on", "true", "evet")
            elif isinstance(default, int):
                v = int(float(v))
            elif isinstance(default, float):
                v = float(v)
            else:
                v = str(v).strip()[:300]
        except (TypeError, ValueError):
            continue
        if k in _LIMITS and not isinstance(v, bool):
            lo, hi = _LIMITS[k]
            v = type(default)(min(hi, max(lo, v)))
        if k in _CHOICES and v not in _CHOICES[k]:
            continue
        out[k] = v
    return out


def update_settings(project: Project, step: str, values: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config(project)
    st = cfg.setdefault("settings", {})
    st[step] = {**(st.get(step) or {}), **clean_settings(step, values)}
    _write(ner_dir(project) / "config.json", cfg)
    return get_settings(cfg)[step]


def select_docs(project: Project, *, exclude: set[str] = frozenset(), n: int = 0, selection: str = "first",
                source: str = "all", keyword: str = "", min_chars: int = 0, seed: int = 7,
                only: Callable[[dict[str, Any]], bool] | None = None) -> list[str]:
    """Belge seçimi: kaynak (arşiv / yüklenen), anahtar kelime (başlıkta veya metinde), en az uzunluk, sıralama, adet."""
    kws = [k for k in norm_name(keyword).split() if k]
    out = []
    for d in project.documents():
        if d["id"] in exclude or d.get("chars", 0) < min_chars:
            continue
        if source == "archive" and not d.get("source"):
            continue
        if source == "upload" and d.get("source"):
            continue
        if only and not only(d):
            continue
        if kws:
            hay = norm_name(d.get("filename", "") + " " + project.text(d["id"])[:20000])
            if not all(k in hay for k in kws):
                continue
        out.append(d)
    if selection == "random":
        random.Random(seed).shuffle(out)
    elif selection == "last":
        out.reverse()
    elif selection in ("longest", "shortest"):
        out.sort(key=lambda d: d.get("chars", 0), reverse=selection == "longest")
    ids = [d["id"] for d in out]
    return ids[:n] if n else ids


def save_config(project: Project, cfg: dict[str, Any]) -> None:
    seen, labels = set(), []
    for l in cfg.get("labels", []):
        name = (l.get("name") or "").strip()
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            item = {"name": name, "description": (l.get("description") or "").strip()}
            if (l.get("prompt") or "").strip():
                item["prompt"] = l["prompt"].strip()
            labels.append(item)
    cfg["labels"] = labels
    _write(ner_dir(project) / "config.json", cfg)
    _sync_ontology(project, labels)


def _sync_ontology(project: Project, labels: list[dict[str, str]]) -> None:
    """Etiketleri ontolojiye varlık türü olarak, BAHSEDER'i ilişki olarak ekler (grafikte renk ve dışa aktarım için)."""
    onto = project.ontology()
    names = {t.name.casefold() for t in onto.entity_types}
    ents = list(onto.entity_types) + [EntityType(name=l["name"], description=l["description"] or None)
                                      for l in labels if l["name"].casefold() not in names]
    rels = list(onto.relation_types)
    targets = [l["name"] for l in labels]
    rel = next((r for r in rels if r.name == MENTIONS_PREDICATE), None)
    if targets:
        if rel is None:
            rels.append(RelationType(name=MENTIONS_PREDICATE, source=[PUBLICATION_TYPE], target=targets,
                                     description="Yayın/belge bu varlıktan bahseder (varlık modeli)"))
        else:
            rel.target = sorted(set(rel.target) | set(targets))
    if ents != list(onto.entity_types) or rels != list(onto.relation_types):
        project.save_ontology(Ontology(entity_types=ents, relation_types=rels, domain=onto.domain))


def humanize(name: str) -> str:
    """"AnalizYöntemi" / "Analiz_Yöntemi" → "analiz yöntemi" (GLiNER doğal dil etiketleriyle daha iyi çalışır)."""
    t = re.sub(r"(?<=[a-zçğıöşü])(?=[A-ZÇĞİÖŞÜ])", " ", name).replace("_", " ").replace("/", " / ")
    t = t.replace("İ", "i").replace("I", "ı").lower() if re.search("[çğışöüÇĞİŞÖÜ]", t) else t.lower()
    return re.sub(r"\s+", " ", t).strip()


def gliner_label_map(cfg: dict[str, Any]) -> dict[str, str]:
    """tür adı → GLiNER'in gördüğü etiket metni.

    Öncelik: türün "prompt" alanı; "gliner_use_description" açıksa açıklama; yoksa adın doğal yazılışı.
    Metinler benzersiz olmalı (çakışırsa adın doğal yazılışına düşülür)."""
    out: dict[str, str] = {}
    used: set[str] = set()
    for l in cfg["labels"]:
        cands = [l.get("prompt") or ""]
        if cfg.get("gliner_use_description"):
            cands.append(re.sub(r"\s+", " ", l.get("description") or "")[:200])
        cands += [humanize(l["name"]), l["name"]]
        text = next(c.strip() for c in cands if c.strip() and c.strip().casefold() not in used)
        used.add(text.casefold())
        out[l["name"]] = text
    return out


def parse_labels_json(raw: str | bytes) -> dict[str, Any]:
    """Tür listesi JSON'u: {"domain", "entity_types": [{"name","description","uri"?,"prompt"?,"examples"?}],
    "relation_types": [...]} (ontoloji biçimi). Ayrıca {"labels": [...]}, düz liste veya {"Ad": "açıklama"} kabul edilir."""
    data = json.loads(raw)
    domain, rels = "", []
    if isinstance(data, dict):
        domain = str(data.get("domain") or "")
        rels = data.get("relation_types") or []
        items = data.get("entity_types") or data.get("labels") or data.get("types")
        if items is None:  # {"Ad": "açıklama"}
            items = [{"name": k, "description": v} for k, v in data.items() if isinstance(v, str) and k != "domain"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON bir nesne veya liste olmalı")
    labels = []
    for it in items:
        if isinstance(it, str):
            it = {"name": it}
        if not isinstance(it, dict) or not str(it.get("name") or it.get("label") or "").strip():
            continue
        desc = str(it.get("description") or "").strip()
        ex = it.get("examples") or it.get("örnekler") or []
        if isinstance(ex, list) and ex:
            desc = (desc + " " if desc else "") + "(örn. " + ", ".join(map(str, ex[:12])) + ")"
        label = {"name": str(it.get("name") or it.get("label")).strip(), "description": desc}
        prompt = it.get("prompt") or it.get("gliner") or it.get("gliner_label")
        if prompt:
            label["prompt"] = str(prompt).strip()
        if it.get("uri"):
            label["uri"] = str(it["uri"])
        labels.append(label)
    if not labels:
        raise ValueError("JSON'da varlık türü bulunamadı (entity_types / labels listesi)")
    return {"domain": domain, "labels": labels, "relation_types": rels if isinstance(rels, list) else []}


def merge_relation_types(project: Project, rels: list[dict[str, Any]], labels: list[dict[str, Any]]) -> int:
    """JSON'daki ilişki türlerini (ve türlerin URI'lerini) proje ontolojisine ekler/günceller."""
    onto = project.ontology()
    ents = list(onto.entity_types)
    by_name = {t.name.casefold(): t for t in ents}
    for l in labels:
        t = by_name.get(l["name"].casefold())
        if t is not None:
            if l.get("description"):
                t.description = l["description"]
            if l.get("uri"):
                t.uri = l["uri"]
    current = {r.name: r for r in onto.relation_types}
    n = 0
    for r in rels:
        try:
            rt = RelationType.model_validate(r)
        except Exception:
            continue
        current[rt.name] = rt
        n += 1
    project.save_ontology(Ontology(entity_types=ents, relation_types=list(current.values()), domain=onto.domain).cleaned())
    return n


def labels_json(project: Project, cfg: dict[str, Any]) -> dict[str, Any]:
    names = {l["name"] for l in cfg["labels"]}
    onto = project.ontology()
    uris = {t.name: t.uri for t in onto.entity_types}
    rels = [r.model_dump() for r in onto.relation_types
            if r.name != MENTIONS_PREDICATE and set(r.source) <= names and set(r.target) <= names]
    return {"domain": cfg.get("domain", ""),
            "entity_types": [{"name": l["name"], "description": l["description"], "uri": uris.get(l["name"]),
                              **({"prompt": l["prompt"]} if l.get("prompt") else {})} for l in cfg["labels"]],
            "relation_types": rels}


def merge_labels(existing: list[dict[str, Any]], new: list[dict[str, Any]], mode: str = "merge") -> list[dict[str, Any]]:
    """mode="replace": yeni liste; "merge": var olanlar korunur, aynı adlıların boş olmayan açıklaması/etiketi
    güncellenir, yeniler sona eklenir."""
    if mode == "replace":
        return [dict(l) for l in new]
    out = [dict(l) for l in existing]
    idx = {l["name"].casefold(): l for l in out}
    for l in new:
        cur = idx.get(l["name"].strip().casefold())
        if cur is None:
            cur = dict(l)
            out.append(cur)
            idx[l["name"].strip().casefold()] = cur
            continue
        for k in ("description", "prompt", "uri"):
            if (l.get(k) or "").strip():
                cur[k] = l[k]
    return out


def label_names(cfg: dict[str, Any]) -> list[str]:
    return [l["name"] for l in cfg["labels"]]


def load_dataset(project: Project) -> list[dict[str, Any]]:
    path = ner_dir(project) / "dataset.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def save_dataset(project: Project, rows: list[dict[str, Any]]) -> None:
    path = ner_dir(project) / "dataset.jsonl"
    tmp = path.with_name(".dataset.jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, int] = {}
    for r in rows:
        for e in r["entities"]:
            by_label[e["label"]] = by_label.get(e["label"], 0) + 1
    return {"n_docs": len(rows), "n_entities": sum(by_label.values()),
            "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1]))}


# ---------------------------------------------------------------- aralık hizalama ve GLiNER biçimi


def find_spans(text: str, mention: str) -> list[tuple[int, int]]:
    """İfadenin metindeki tüm (kelime sınırına oturan) geçişleri; önce birebir, yoksa büyük/küçük harf duyarsız."""
    mention = mention.strip()
    if len(mention) < 2:
        return []
    pat = r"(?<!\w)" + re.escape(mention) + r"(?!\w)"
    spans = [m.span() for m in re.finditer(pat, text)]
    if not spans:
        spans = [m.span() for m in re.finditer(pat, text, re.IGNORECASE)]
    return spans


def resolve_overlaps(ents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Çakışan aralıklarda uzun olan kalır (düz NER)."""
    out: list[dict[str, Any]] = []
    for e in sorted(ents, key=lambda e: (-(e["end"] - e["start"]), e["start"])):
        if all(e["end"] <= o["start"] or e["start"] >= o["end"] for o in out):
            out.append(e)
    return sorted(out, key=lambda e: e["start"])


def windows(text: str, max_words: int = WINDOW_WORDS) -> list[tuple[int, str]]:
    """Metni cümle sınırlarından, her biri en fazla max_words kelime olan pencerelere böler: [(başlangıç, metin)]."""
    pieces, pos = [], 0
    for m in SENT_RE.finditer(text):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(text)))
    out: list[tuple[int, str]] = []
    cur_s = cur_e = None
    cur_words = 0
    for s, e in pieces:
        if s >= e:
            continue
        n = len(text[s:e].split())
        if n > max_words:  # çok uzun cümle: kelime sayısına göre kes
            if cur_s is not None:
                out.append((cur_s, text[cur_s:cur_e]))
                cur_s, cur_words = None, 0
            toks = list(re.finditer(r"\S+", text[s:e]))
            for i in range(0, len(toks), max_words):
                a, b = s + toks[i].start(), s + toks[min(i + max_words, len(toks)) - 1].end()
                out.append((a, text[a:b]))
            continue
        if cur_s is not None and cur_words + n > max_words:
            out.append((cur_s, text[cur_s:cur_e]))
            cur_s, cur_words = None, 0
        if cur_s is None:
            cur_s = s
        cur_e, cur_words = e, cur_words + n
    if cur_s is not None:
        out.append((cur_s, text[cur_s:cur_e]))
    return out


def to_gliner(text: str, entities: list[dict[str, Any]], max_words: int = WINDOW_WORDS) -> list[dict[str, Any]]:
    """{"tokenized_text": [...], "ner": [[ilk_token, son_token, etiket], ...]} örnekleri (GLiNER eğitim biçimi)."""
    out = []
    for off, w in windows(text, max_words):
        toks = [(m.start() + off, m.end() + off, m.group()) for m in TOKEN_RE.finditer(w)]
        if not toks:
            continue
        ner = []
        for e in entities:
            if e["start"] < off or e["end"] > off + len(w):
                continue
            idx = [i for i, (s, t, _) in enumerate(toks) if s >= e["start"] and t <= e["end"]]
            if idx and toks[idx[0]][0] == e["start"] and toks[idx[-1]][1] == e["end"]:
                ner.append([idx[0], idx[-1], e["label"]])
        out.append({"tokenized_text": [t for _, _, t in toks], "ner": ner})
    return out


def split_dataset(rows: list[dict[str, Any]], test_ratio: float = 0.2, seed: int = 13) -> tuple[list, list]:
    if len(rows) < 5:
        return rows, []
    rs = sorted(rows, key=lambda r: r["doc_id"])
    random.Random(seed).shuffle(rs)
    n_test = max(1, int(round(len(rs) * test_ratio)))
    return rs[n_test:], rs[:n_test]


# ---------------------------------------------------------------- dil modeliyle etiketleme

ANNOTATE_SYSTEM = """Sen bir adlandırılmış varlık tanıma (NER) etiketleyicisisin. Birden çok metin verilecek; her metinde,
verilen türlerden birine giren varlıkları bul.

Kurallar:
- "text" alanına varlığı METİNDE GEÇTİĞİ BİÇİMİYLE harfi harfine yaz; kelimeyi ortadan bölme (ekleriyle birlikte
  kelimenin tamamı), kesme işaretinden sonraki eki yazma ("Ankara'da" → "Ankara").
- Aynı yazılışı bir kez yazman yeter; sistem metindeki tüm geçişlerini kendisi bulur.
- Yalnızca tür tanımına uyan, adıyla geçen belirli varlıkları al; genel/soyut kelimeleri ve emin olmadıklarını alma.
- Her metnin sonucunu "index" alanında o metnin numarasıyla ver. Varlık yoksa boş liste ver."""


def _annotation_schema(labels: list[str]) -> type[BaseModel]:
    Lit = Literal[tuple(labels)]  # type: ignore[valid-type]
    Ent = create_model("NerEntity", text=(str, Field(description="Varlık, metinde geçtiği biçimiyle")),
                       label=(Lit, Field(description="Varlık türü")))
    Doc = create_model("NerDoc", index=(int, Field(description="Metin numarası")), entities=(list[Ent], ...))
    return create_model("NerAnnotation", documents=(list[Doc], ...))


def annotation_prompt(cfg: dict[str, Any], texts: list[str]) -> str:
    types = "\n".join(f"- {l['name']}: {l['description']}" if l["description"] else f"- {l['name']}" for l in cfg["labels"])
    body = "\n\n".join(f"=== METİN {i + 1} ===\n{t}" for i, t in enumerate(texts))
    return f"Varlık türleri:\n{types}\n\n{body}"


def plan_groups(texts: dict[str, str], max_chars: int, per_request: int) -> list[list[str]]:
    groups, cur, size = [], [], 0
    for d, t in texts.items():
        if cur and (size + len(t) > max_chars or len(cur) >= per_request):
            groups.append(cur)
            cur, size = [], 0
        cur.append(d)
        size += len(t)
    if cur:
        groups.append(cur)
    return groups


def spans_from_answer(text: str, items: Iterable[dict[str, Any]], labels: set[str]) -> tuple[list[dict], list[dict]]:
    ents, rejected = [], []
    for it in items:
        label, mention = it.get("label"), (it.get("text") or "").strip()
        if label not in labels:
            continue
        spans = find_spans(text, mention)
        if not spans:
            rejected.append({"text": mention, "label": label, "reason": "metinde bulunamadı"})
            continue
        ents += [{"start": s, "end": e, "label": label, "text": text[s:e]} for s, e in spans]
    return resolve_overlaps(ents), rejected


def choose_docs(project: Project, n: int, exclude: set[str], seed: int = 7, min_chars: int = 40) -> list[str]:
    return select_docs(project, exclude=exclude, n=n, selection="random", seed=seed, min_chars=min_chars)


def annotate(project: Project, llm: LLM, n: int | None = None, per_request: int | None = None,
             use_batch_api: bool | None = None, doc_ids: list[str] | None = None, add_to_graph: bool | None = None,
             poll_seconds: float = 30.0, **overrides: Any) -> dict[str, Any]:
    """Belgeleri dil modeliyle etiketleyip veri setine ekler. Var olan kayıtlar atlanır (kaldığı yerden devam).
    Verilmeyen parametreler projenin kayıtlı ayarlarından (settings.annotate) gelir."""
    cfg = load_config(project)
    labels = label_names(cfg)
    if not labels:
        raise ValueError("Önce varlık türlerini (etiketleri) belirle")
    s = get_settings(cfg)["annotate"]
    s.update(clean_settings("annotate", {k: v for k, v in
                                         {"n": n, "per_request": per_request, "add_to_graph": add_to_graph, **overrides}.items()
                                         if v is not None}))
    rows = load_dataset(project)
    done = {r["doc_id"] for r in rows}
    if doc_ids is None:
        want = max(0, s["n"] - len(done)) if s["count_mode"] == "total" else s["n"]
        doc_ids = select_docs(project, exclude=done, n=want, selection=s["selection"], source=s["source"],
                              keyword=s["keyword"], min_chars=s["min_chars"], seed=s["seed"]) if want else []
    ids = [d for d in doc_ids if d not in done]
    texts = {d: project.text(d) for d in ids}
    texts = {d: t for d, t in texts.items() if t.strip()}
    if not texts:
        log.info("Etiketlenecek yeni belge yok (veri setinde %d belge)", len(done))
        return {"added": 0, "skipped": len(done), "total_docs": len(done)}
    if s["model"] and not is_manual(llm) and hasattr(llm, "model_name"):
        llm.model_name = s["model"]  # etiketleme için farklı (ör. daha ucuz) model
    add_to_graph = s["add_to_graph"]
    per_request = s["per_request"]
    max_chars = s["max_chars"] or (MANUAL_GROUP_CHARS if is_manual(llm) else API_GROUP_CHARS)
    groups = plan_groups(texts, max_chars, max(1, per_request))
    schema = _annotation_schema(labels)
    if use_batch_api is None:
        use_batch_api = s["batch_api"] and hasattr(llm, "client") and hasattr(getattr(llm, "client"), "messages")
    log.info("%d belge, %d istekte etiketlenecek (%s)", len(texts), len(groups),
             "Anthropic Message Batches" if use_batch_api else "istek başına birden çok belge")

    answers: dict[str, list[dict[str, Any]]] = {}
    if use_batch_api:
        answers = _anthropic_batch(project, llm, cfg, groups, texts, schema, poll_seconds)
    else:
        for i, g in enumerate(groups, 1):
            log.info("İstek %d/%d (%d belge)", i, len(groups), len(g))
            try:
                res = llm.complete_json(ANNOTATE_SYSTEM, annotation_prompt(cfg, [texts[d] for d in g]), schema)
            except Exception as exc:  # bir grubun hatası tüm işi düşürmesin
                log.warning("İstek %d başarısız: %s", i, exc)
                continue
            answers.update(_answers_by_doc(g, res.model_dump()))

    added, n_ent, n_rej = 0, 0, 0
    model_name = getattr(llm, "model_name", "llm")
    by_doc_rows = {r["doc_id"]: r for r in rows}
    for d, items in answers.items():
        ents, rej = spans_from_answer(texts[d], items, set(labels))
        by_doc_rows[d] = {"doc_id": d, "text": texts[d], "entities": ents, "rejected": rej, "source": model_name}
        added += 1
        n_ent += len(ents)
        n_rej += len(rej)
        if add_to_graph:
            write_entities(project, d, texts[d], ents, origin="llm")
    save_dataset(project, list(by_doc_rows.values()))
    log.info("Veri setine %d belge eklendi: %d varlık geçişi, %d doğrulanamayan öneri", added, n_ent, n_rej)
    return {"added": added, "entities": n_ent, "rejected": n_rej, "total_docs": len(by_doc_rows)}


def _answers_by_doc(group: list[str], data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {d: [] for d in group}
    for doc in data.get("documents", []):
        i = doc.get("index", 0) - 1
        if 0 <= i < len(group):
            out[group[i]] += doc.get("entities", [])
    return out


def _anthropic_batch(project: Project, llm: Any, cfg: dict[str, Any], groups: list[list[str]], texts: dict[str, str],
                     schema: type[BaseModel], poll_seconds: float) -> dict[str, list[dict[str, Any]]]:
    """Anthropic Message Batches API: tüm istekler tek seferde gönderilir (standart fiyatın yarısı), sonuçlar beklenir.
    Gönderilen paketin kimliği state.json'a yazılır; iş yarıda kalırsa yeniden çalıştırınca aynı paket beklenir."""
    from ..schema import strict_json_schema

    state_path = ner_dir(project) / "state.json"
    state = _read(state_path, {})
    client = llm.client
    if state.get("batch_id"):  # yarıda kalan paket: yeniden göndermeden sonucunu bekle
        batch_id, gmap = state["batch_id"], state["groups"]
        for g in gmap.values():
            for d in g:
                if d not in texts and project.document(d):
                    texts[d] = project.text(d)
        log.info("Önceki paket bekleniyor: %s", batch_id)
    else:
        gmap = {f"g{i}": g for i, g in enumerate(groups)}
        fmt = {"type": "json_schema", "schema": strict_json_schema(schema)}
        requests = [{"custom_id": cid, "params": {
            "model": llm.model_name, "max_tokens": 16000, "system": ANNOTATE_SYSTEM,
            "messages": [{"role": "user", "content": annotation_prompt(cfg, [texts[d] for d in g])}],
            "output_config": {"effort": getattr(llm, "effort", "high"), "format": fmt},
        }} for cid, g in gmap.items()]
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        _write(state_path, {"batch_id": batch_id, "groups": gmap, "created": time.time()})
        log.info("Paket gönderildi: %s (%d istek). Sonuçlar genelde birkaç dakikada, en geç 24 saatte gelir.", batch_id, len(requests))
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        if b.processing_status == "ended":
            break
        log.info("Paket işleniyor: %d bitti, %d sürüyor", c.succeeded + c.errored + c.canceled + c.expired, c.processing)
        time.sleep(poll_seconds)
    answers: dict[str, list[dict[str, Any]]] = {}
    for entry in client.messages.batches.results(batch_id):
        g = [d for d in gmap.get(entry.custom_id, []) if d in texts]
        if len(g) != len(gmap.get(entry.custom_id, [])) or not g:
            continue
        if entry.result.type != "succeeded":
            log.warning("İstek %s: %s", entry.custom_id, entry.result.type)
            continue
        text = next((blk.text for blk in entry.result.message.content if blk.type == "text"), "")
        try:
            answers.update(_answers_by_doc(g, schema.model_validate_json(text).model_dump()))
        except Exception as exc:
            log.warning("İstek %s yanıtı okunamadı: %s", entry.custom_id, exc)
    state_path.unlink(missing_ok=True)
    return answers


# ---------------------------------------------------------------- modeller


def gliner_available() -> bool:
    try:
        import gliner  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class GazetteerModel:
    """Bağımlılıksız yedek model: veri setinde görülen ifadeleri (etiketi ve güvenilirliğiyle) tanır."""
    kind = "sozluk"

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None):
        self.entries = entries or {}  # norm ifade -> {label, score, surface}
        self._re: re.Pattern | None = None

    @classmethod
    def train(cls, rows: list[dict[str, Any]], min_score: float = 0.34) -> "GazetteerModel":
        labeled: dict[str, dict[str, int]] = {}
        surface: dict[str, str] = {}
        for r in rows:
            for e in r["entities"]:
                k = norm_name(e["text"])
                labeled.setdefault(k, {})
                labeled[k][e["label"]] = labeled[k].get(e["label"], 0) + 1
                surface.setdefault(k, e["text"])
        entries = {}
        for k, counts in labeled.items():
            if len(k) < 2:
                continue
            total = sum(len(find_spans(r["text"], surface[k])) for r in rows) or 1
            label, n = max(counts.items(), key=lambda x: x[1])
            score = min(1.0, n / total)
            if score >= min_score:
                entries[k] = {"label": label, "score": round(score, 3), "surface": surface[k]}
        return cls(entries)

    def _pattern(self) -> re.Pattern | None:
        if self._re is None and self.entries:
            alts = sorted({re.escape(e["surface"]) for e in self.entries.values()}, key=len, reverse=True)
            self._re = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE)
        return self._re

    def predict(self, texts: list[str], labels: list[str], threshold: float = 0.5) -> list[list[dict[str, Any]]]:
        pat, allowed = self._pattern(), set(labels)
        out = []
        for t in texts:
            ents = []
            if pat:
                for m in pat.finditer(t):
                    e = self.entries.get(norm_name(m.group()))
                    if e and e["label"] in allowed and e["score"] >= threshold:
                        ents.append({"start": m.start(), "end": m.end(), "label": e["label"], "text": m.group(), "score": e["score"]})
            out.append(resolve_overlaps(ents))
        return out

    def save(self, path: Path) -> None:
        _write(path, {"kind": self.kind, "entries": self.entries})

    @classmethod
    def load(cls, path: Path) -> "GazetteerModel":
        return cls(_read(path, {}).get("entries", {}))


class GlinerModel:
    kind = "gliner"

    def __init__(self, model: Any, name: str, label_map: dict[str, str] | None = None):
        self.model, self.name = model, name
        self.label_map = label_map or {}  # tür adı → GLiNER etiket metni

    @classmethod
    def load(cls, name_or_path: str | Path) -> "GlinerModel":
        from gliner import GLiNER
        p = Path(name_or_path)
        local = p.is_dir()
        log.info("GLiNER modeli yükleniyor: %s%s", name_or_path, "" if local else " (ilk seferde indirilir)")
        m = GLiNER.from_pretrained(str(p) if local else str(name_or_path), local_files_only=local)
        return cls(m, str(name_or_path))

    def predict(self, texts: list[str], labels: list[str], threshold: float = 0.5, batch_size: int = 8) -> list[list[dict[str, Any]]]:
        wins: list[tuple[int, int, str]] = []  # (metin no, ofset, pencere)
        for i, t in enumerate(texts):
            wins += [(i, off, w) for off, w in windows(t)]
        out: list[list[dict[str, Any]]] = [[] for _ in texts]
        infer = getattr(self.model, "inference", None) or self.model.batch_predict_entities
        prompts = [self.label_map.get(n, n) for n in labels]
        back = dict(zip(prompts, labels))
        for a in range(0, len(wins), 64):
            chunk = wins[a:a + 64]
            preds = infer([w for _, _, w in chunk], prompts, threshold=threshold, batch_size=batch_size)
            for (i, off, _), ps in zip(chunk, preds):
                for p in ps:
                    out[i].append({"start": p["start"] + off, "end": p["end"] + off, "label": back.get(p["label"], p["label"]),
                                   "text": p["text"], "score": round(float(p.get("score", 0)), 3)})
        return [resolve_overlaps(o) for o in out]

    def train(self, train_rows: list[dict[str, Any]], out_dir: Path, epochs: int = 5, batch_size: int = 8,
              lr: float = 5e-6, others_lr: float = 1e-5, progress: Callable[[str], None] | None = None) -> None:
        import torch

        examples = [ex for r in train_rows for ex in to_gliner(r["text"], r["entities"])]
        examples = [{**ex, "ner": [[a, b, self.label_map.get(lab, lab)] for a, b, lab in ex["ner"]]}
                    for ex in examples if ex["tokenized_text"]]
        if not examples:
            raise ValueError("Eğitim örneği yok")
        steps = max(1, epochs * math.ceil(len(examples) / batch_size))
        cpu = not torch.cuda.is_available()
        log.info("GLiNER eğitimi: %d örnek, %d adım, %s", len(examples), steps, "CPU" if cpu else "GPU")
        work = out_dir.parent / "_train_tmp"
        args = self.model.create_training_args(
            output_dir=str(work), learning_rate=lr, weight_decay=0.01, others_lr=others_lr, others_weight_decay=0.01,
            focal_loss_alpha=0.75, focal_loss_gamma=2, lr_scheduler_type="linear", warmup_ratio=0.1,
            per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, max_steps=steps,
            save_strategy="no", eval_strategy="no", logging_steps=max(1, steps // 20), use_cpu=cpu,
            dataloader_num_workers=0, report_to="none", disable_tqdm=True)
        try:
            from transformers import TrainerCallback

            class _Log(TrainerCallback):
                def on_log(self, _args, state, _control, logs=None, **_kw):
                    if logs and "loss" in logs:
                        msg = f"adım {state.global_step}/{steps} · kayıp {logs['loss']:.3f}"
                        log.info(msg)
                        if progress:
                            progress(msg)

            from gliner.training import Trainer
            from packaging import version
            import transformers
            kw: dict[str, Any] = {"model": self.model, "args": args, "train_dataset": examples, "eval_dataset": None,
                                  "data_collator": self.model._create_data_collator(), "callbacks": [_Log()]}
            if version.parse(transformers.__version__) < version.parse("5.0.0"):
                kw["tokenizer"] = self.model.data_processor.transformer_tokenizer
            else:
                kw["processing_class"] = self.model.data_processor.transformer_tokenizer
            Trainer(**kw).train()
        except AttributeError:  # farklı gliner sürümü: hazır eğitim yardımcısı
            self.model.train_model(train_dataset=examples, eval_dataset=None, training_args=args)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        self.model.save_pretrained(str(out_dir))
        shutil.rmtree(work, ignore_errors=True)
        self.name = str(out_dir)


_model_cache: dict[str, tuple[float, Any]] = {}


def trained_model(project: Project) -> Any | None:
    """Projenin eğitilmiş modeli (GLiNER klasörü veya sözlük dosyası), yoksa None. Bellekte önbelleklenir."""
    d = ner_dir(project)
    meta = _read(d / "metrics.json", {})
    kind = meta.get("model_kind")
    path = d / "model" if kind == "gliner" else d / "gazetteer.json"
    if not kind or not path.exists():
        return None
    mtime = path.stat().st_mtime
    hit = _model_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    m = GlinerModel.load(path) if kind == "gliner" else GazetteerModel.load(path)
    if kind == "gliner":  # eğitimde kullanılan etiket metinleri; yüklenen modelde güncel ayar
        m.label_map = meta.get("gliner_labels") or gliner_label_map(load_config(project))
    _model_cache[str(path)] = (mtime, m)
    return m


# ---------------------------------------------------------------- değerlendirme


def evaluate(model: Any, rows: list[dict[str, Any]], labels: list[str], threshold: float = 0.5) -> dict[str, Any]:
    """Dil modeli etiketlerine göre tam eşleşme (aralık + tür) kesinlik / duyarlılık / F1."""
    if not rows:
        return {}
    preds = model.predict([r["text"] for r in rows], labels, threshold)
    tp = fp = fn = 0
    per: dict[str, list[int]] = {l: [0, 0, 0] for l in labels}
    for r, ps in zip(rows, preds):
        gold = {(e["start"], e["end"], e["label"]) for e in r["entities"] if e["label"] in labels}
        pred = {(e["start"], e["end"], e["label"]) for e in ps}
        for x in pred & gold:
            per[x[2]][0] += 1
        for x in pred - gold:
            per.setdefault(x[2], [0, 0, 0])[1] += 1
        for x in gold - pred:
            per[x[2]][2] += 1
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)

    def prf(a: int, b: int, c: int) -> dict[str, float]:
        p = a / (a + b) if a + b else 0.0
        r = a / (a + c) if a + c else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                "support": a + c}

    return {"micro": prf(tp, fp, fn), "per_label": {l: prf(*v) for l, v in per.items() if sum(v)}, "n_docs": len(rows)}


# ---------------------------------------------------------------- eğitim


def train(project: Project, kind: str | None = None, epochs: int | None = None, test_ratio: float | None = None,
          base_model: str | None = None, evaluate_base: bool = True, **overrides: Any) -> dict[str, Any]:
    """Verilmeyen parametreler projenin kayıtlı ayarlarından (settings.train) gelir."""
    cfg = load_config(project)
    labels = label_names(cfg)
    s = get_settings(cfg)["train"]
    s.update(clean_settings("train", {k: v for k, v in {"kind": kind, "epochs": epochs, "test_ratio": test_ratio,
                                                        **overrides}.items() if v is not None}))
    kind, epochs, test_ratio = s["kind"], s["epochs"], s["test_ratio"]
    rows = [r for r in load_dataset(project) if r["entities"] or r["text"]]
    if len(rows) < 2:
        raise ValueError("Eğitim için en az birkaç etiketli belge gerekir (önce 2. adımı çalıştır)")
    if kind == "auto":
        kind = "gliner" if gliner_available() else "sozluk"
    if kind == "gliner" and not gliner_available():
        raise ValueError("GLiNER kurulu değil (gliner + torch). Docker imajını WITH_GLINER=1 ile kur veya sözlük modelini seç.")
    train_rows, test_rows = split_dataset(rows, test_ratio)
    d = ner_dir(project)
    th = float(cfg.get("threshold", 0.5))
    metrics: dict[str, Any] = {"model_kind": kind, "n_train": len(train_rows), "n_test": len(test_rows), "test_ratio": test_ratio,
                               "labels": labels, "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    if kind == "gliner":
        base = base_model or cfg["base_model"]
        m = GlinerModel.load(base)
        m.label_map = gliner_label_map(cfg)
        metrics["gliner_labels"] = m.label_map
        if evaluate_base and test_rows:
            log.info("Eğitimden önce temel model (sıfır örnekle) test ediliyor")
            metrics["base"] = evaluate(m, test_rows, labels, th)
        m.train(train_rows, d / "model", epochs=epochs, batch_size=s["batch_size"], lr=s["lr"], others_lr=s["others_lr"])
        metrics["base_model"] = base
        metrics.update(epochs=epochs, batch_size=s["batch_size"], lr=s["lr"], others_lr=s["others_lr"])
    else:
        m = GazetteerModel.train(train_rows)
        m.save(d / "gazetteer.json")
        metrics["n_entries"] = len(m.entries)
    if test_rows:
        log.info("Test belgelerinde değerlendiriliyor (%d belge)", len(test_rows))
        metrics["test"] = evaluate(m, test_rows, labels, th)
    if kind == "sozluk" and test_rows:  # son model tüm veriyle (test dahil) yeniden kurulur
        GazetteerModel.train(rows).save(d / "gazetteer.json")
    _write(d / "metrics.json", metrics)
    _model_cache.clear()
    return metrics


# ---------------------------------------------------------------- uygulama ve graf


def _sentence_at(text: str, start: int, end: int, limit: int = 300) -> str:
    a = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    b_dot, b_nl = text.find(".", end), text.find("\n", end)
    cands = [x for x in (b_dot, b_nl) if x != -1]
    b = (min(cands) + 1) if cands else len(text)
    s = re.sub(r"\s+", " ", text[a:b]).strip()
    if len(s) > limit:
        mid = (start - a)
        s = s[max(0, mid - limit // 2): max(0, mid - limit // 2) + limit].strip()
    return s


def publication_keys(project: Project) -> dict[str, str]:
    """belge kimliği → arşiv metadata grafındaki Yayın düğümü."""
    out = {}
    for key, e in project.graph()["entities"].items():
        if e["type"] != PUBLICATION_TYPE:
            continue
        for m in e["mentions"]:
            if m["doc_id"].endswith("#meta"):
                out[m["doc_id"][:-5]] = key
    return out


def write_entities(project: Project, doc_id: str, text: str, ents: list[dict[str, Any]], origin: str,
                   pub_keys: dict[str, str] | None = None) -> int:
    """Bir belgenin NER sonucunu grafa yazar (önceki NER katkısının yerine)."""
    contrib = doc_id + NER_SUFFIX
    project.remove_contributions(contrib)
    first: dict[tuple[str, str], dict[str, Any]] = {}
    for e in ents:
        k = (e["label"], norm_name(e["text"]))
        if k[1] and k not in first:
            first[k] = e
    entities = [{"type": e["label"], "name": e["text"], "mention": e["text"], "description": None} for e in first.values()]
    pub = (pub_keys if pub_keys is not None else publication_keys(project)).get(doc_id)
    relations = []
    if pub:
        for e in first.values():
            relations.append({"subject_key": pub, "predicate": MENTIONS_PREDICATE,
                              "object_key": Project.entity_key(e["label"], e["text"]),
                              "sentence": _sentence_at(text, e["start"], e["end"]),
                              "confidence": float(e.get("score", 1.0))})
    if entities:
        project.add_extraction(contrib, entities, relations, [])
    project.update_document(doc_id, ner=origin, n_ner=len(entities))
    return len(entities)


def apply(project: Project, doc_ids: list[str] | None = None, threshold: float | None = None,
          batch_docs: int | None = None, **overrides: Any) -> dict[str, Any]:
    """Eğitilmiş modelle belgelerden varlık çıkarır. doc_ids verilmezse kayıtlı ayarlara (settings.apply) göre seçilir:
    scope = unlabeled (veri setinde olmayanlar) | new (henüz hiç NER uygulanmamışlar) | all; ayrıca süzgeç ve adet sınırı."""
    cfg = load_config(project)
    labels = label_names(cfg)
    model = trained_model(project)
    if model is None:
        raise ValueError("Önce modeli eğit (3. adım)")
    s = get_settings(cfg)["apply"]
    s.update(clean_settings("apply", {k: v for k, v in {"batch_docs": batch_docs, **overrides}.items() if v is not None}))
    batch_docs = s["batch_docs"]
    th = float(cfg.get("threshold", 0.5) if threshold is None else threshold)
    labeled = {r["doc_id"] for r in load_dataset(project)}
    if doc_ids is None:
        exclude = set() if s["scope"] == "all" else labeled
        only = (lambda d: not d.get("ner")) if s["scope"] == "new" or s["skip_done"] else None
        doc_ids = select_docs(project, exclude=exclude, n=s["limit"], selection=s["selection"], source=s["source"],
                              keyword=s["keyword"], min_chars=s["min_chars"], only=only)
    log.info("%d belgeye uygulanacak (eşik %.2f)", len(doc_ids), th)
    pubs = publication_keys(project)
    total, n_docs = 0, 0
    t0 = time.time()
    for a in range(0, len(doc_ids), batch_docs):
        chunk = doc_ids[a:a + batch_docs]
        texts = [project.text(d) for d in chunk]
        preds = model.predict(texts, labels, th)
        for d, t, ps in zip(chunk, texts, preds):
            total += write_entities(project, d, t, ps, origin="model", pub_keys=pubs)
            n_docs += 1
        log.info("%d/%d belge işlendi (%d varlık, %.1f sn)", n_docs, len(doc_ids), total, time.time() - t0)
    project.sync_neo4j()
    return {"docs": n_docs, "entities": total, "seconds": round(time.time() - t0, 1)}


def remove_from_dataset(project: Project, doc_ids: Iterable[str] | None = None, keep: int | None = None,
                        seed: int = 7) -> int:
    """Veri setinden belge çıkarır: belirli belgeler veya rastgele seçilip `keep` belgeye indirerek.
    Çıkarılan belgelerin dil modeli etiketleri graftan da kaldırılır."""
    rows = load_dataset(project)
    if keep is not None:
        order = list(rows)
        random.Random(seed).shuffle(order)
        drop = {r["doc_id"] for r in order[max(0, keep):]}
    else:
        drop = set(doc_ids or [])
    kept = [r for r in rows if r["doc_id"] not in drop]
    removed = [r["doc_id"] for r in rows if r["doc_id"] in drop]
    save_dataset(project, kept)
    for d in removed:
        if (project.document(d) or {}).get("ner") == "llm":
            project.remove_contributions(d + NER_SUFFIX)
            project.update_document(d, ner=None, n_ner=0)
    return len(removed)


def clear_ner(project: Project) -> int:
    n = 0
    for d in project.documents():
        if d.get("ner"):
            project.remove_contributions(d["id"] + NER_SUFFIX)
            project.update_document(d["id"], ner=None, n_ner=0)
            n += 1
    return n


def install_model_zip(project: Project, data: bytes) -> None:
    """Başka makinede (ör. GPU'da) eğitilmiş GLiNER klasörünü .zip olarak yükler."""
    import io
    import zipfile

    d = ner_dir(project)
    tmp = d / "_upload"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            target = (tmp / info.filename).resolve()
            if not str(target).startswith(str(tmp.resolve())):
                raise ValueError("Geçersiz zip (klasör dışına yol)")
        z.extractall(tmp)
    cfgs = list(tmp.rglob("gliner_config.json"))
    if not cfgs:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError("Zip içinde gliner_config.json yok (model.save_pretrained çıktısı olmalı)")
    shutil.rmtree(d / "model", ignore_errors=True)
    shutil.move(str(cfgs[0].parent), str(d / "model"))
    shutil.rmtree(tmp, ignore_errors=True)
    meta = _read(d / "metrics.json", {})
    meta.update({"model_kind": "gliner", "uploaded_at": time.strftime("%Y-%m-%d %H:%M"), "labels": label_names(load_config(project))})
    for k in ("test", "base", "gliner_labels"):
        meta.pop(k, None)
    _write(d / "metrics.json", meta)
    _model_cache.clear()


@dataclass
class Status:
    labels: list[dict[str, str]]
    dataset: dict[str, Any]
    metrics: dict[str, Any]
    n_docs: int
    n_unlabeled: int
    n_applied: int
    gliner: bool
    pending_batch: str | None


def status(project: Project) -> Status:
    cfg = load_config(project)
    rows = load_dataset(project)
    labeled = {r["doc_id"] for r in rows}
    docs = project.documents()
    return Status(labels=cfg["labels"], dataset=dataset_stats(rows), metrics=_read(ner_dir(project) / "metrics.json", {}),
                  n_docs=len(docs), n_unlabeled=sum(1 for d in docs if d["id"] not in labeled),
                  n_applied=sum(1 for d in docs if d.get("ner") == "model"), gliner=gliner_available(),
                  pending_batch=_read(ner_dir(project) / "state.json", {}).get("batch_id"))
