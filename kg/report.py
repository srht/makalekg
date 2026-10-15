"""Hipotez kartları (düz Türkçe, markdown): iddia, zincir (makale + şekil), en zayıf halka,
güven skoru, uyarılar ve "bunu test etmek için gereken deney" önerisi."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import REPORTS_DIR
from .graph import GraphSnapshot
from .schema import Hypothesis

PRED_TR = {"INCREASES": "artırır", "DECREASES": "azaltır", "REQUIRES": "gerektirir", "MEDIATES": "aracılık eder",
           "PROTECTS": "korur / iyileştirir", "ASSOCIATED": "ilişkilidir"}
STATUS_TR = {"untested": "test edilmemiş", "tested_supported": "sonradan test edildi — destekleniyor",
             "tested_refuted": "sonradan test edildi — çürütüldü"}
PENALTY_TR = {"species_mismatch": "halkalar farklı türlerden (×0.6)",
              "tissue_mismatch": "halkalar farklı doku/hücre tiplerinden (×0.7)",
              "dose_unclear": "doz veya rejim belirsiz (×0.8)",
              "direction_conflict": "ilişki yönü çelişkili (×0.3)"}


def confidence_words(score: float) -> str:
    if score >= 0.4:
        return "görece güçlü"
    if score >= 0.2:
        return "orta"
    if score >= 0.08:
        return "zayıf"
    return "çok zayıf"


def suggested_experiment(snap: GraphSnapshot, h: Hypothesis, penalties: dict[str, float]) -> str:
    s, o = snap.name(h.subject_key), snap.name(h.object_key)
    weak = min(h.path, key=lambda l: l.get("E", 1.0)) if h.path else None
    lines = []
    if weak:
        lines.append(f"Önce en zayıf halka (*{weak['from_name']} → {weak['to_name']}*, {weak.get('evidence')}) "
                     "bağımsız bir laboratuvarda, körlemeli ve randomize, grup başına n ≥ 20 ile tekrarlanmalı.")
    if "species_mismatch" in penalties:
        lines.append("Zincirin halkaları farklı türlerden geliyor; tüm halkalar aynı türde gösterilmeli.")
    if "dose_unclear" in penalties:
        lines.append("Müdahalenin dozu/rejimi net tanımlanmalı ve doz–yanıt ilişkisi ölçülmeli.")
    lines.append(f"Doğrudan test: {s} manipüle edilerek (ör. koşullu gen silme, antagonist veya nötralize edici antikor) "
                 f"{o} ölçülmeli; {s} olmadan etkinin kaybolup kaybolmadığı ayrı bir grupla kontrol edilmeli.")
    return "\n".join(f"- {l}" for l in lines)


def hypothesis_card(snap: GraphSnapshot, h: Hypothesis, penalties: dict[str, float],
                    warnings: list[str]) -> str:
    s, o = snap.name(h.subject_key), snap.name(h.object_key)
    pred = PRED_TR.get(h.predicate, h.predicate)
    lines = [
        f"# Hipotez: {s} → {o} ({h.predicate})",
        "",
        f"> **Bu bir HİPOTEZDİR, kanıt değildir.** Sistem tarafından grafiğe bakarak üretildi; "
        f"durum: **{STATUS_TR.get(h.status.value, h.status.value)}**.",
        "",
        f"**İddia (düz Türkçe):** {s}, {o} üzerinde etkilidir — {s} {o} için *{pred}* olabilir.",
        "",
        f"**Güven skoru:** {h.score:.3f} ({confidence_words(h.score)}) — zincir güveni C = {h.chain_confidence:.3f}"
        + (f", embedding P = {h.embed_score:.2f}" if h.embed_score is not None else ", embedding yok") + ".",
        "",
        f"**En zayıf halka:** {h.weakest_link}",
        "",
        "## Zincir (kanıt halkaları)",
        "",
    ]
    for i, l in enumerate(h.path, 1):
        lines.append(f"{i}. **{l['from_name']} -{l['predicate']}{' (ters yönde okunur)' if l.get('inverse') else ''}"
                     f"-> {l['to_name']}** — E = {l['E']:.3f}")
        lines.append(f"   - Makale: doi:{l.get('source_doi')} · {l.get('figure') or 'şekil belirtilmemiş'}")
        lines.append(f"   - Kanıt düzeyi: {l.get('evidence')}")
        if l.get("source_sentence"):
            lines.append(f"   - Kaynak cümle: \"{l['source_sentence']}\"")
    lines += ["", "## Gerekçe", "", h.rationale, ""]
    if penalties:
        lines += ["## Uyumsuzluk cezaları", ""] + [f"- {PENALTY_TR.get(k, k)}" for k in penalties] + [""]
    if warnings:
        lines += ["## Uyarılar", ""] + [f"- {w}" for w in warnings] + [""]
    lines += ["## Bunu test etmek için gereken deney", "", suggested_experiment(snap, h, penalties), ""]
    if h.literature_dois:
        lines += ["## Literatür kontrolü", "",
                  "Aşağıdaki makaleler bu ilişkiyi test etmiş olabilir; kanıt grafına ancak boru hattından "
                  "geçtikten sonra girerler:", ""] + [f"- doi:{d}" for d in h.literature_dois] + [""]
    lines += [f"Üreticiler: {', '.join(h.generators)} · oluşturma: {h.created_at}", "",
              "_Bu kart bir araştırma hipotezidir; insan için klinik veya tedavi önerisi değildir._", ""]
    return "\n".join(lines)


def write_card(snap: GraphSnapshot, h: Hypothesis, penalties: dict[str, float], warnings: list[str],
               out_dir: Path | None = None) -> Path:
    out_dir = out_dir or REPORTS_DIR / "hypotheses"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", f"{snap.name(h.subject_key)}_{h.predicate}_{snap.name(h.object_key)}".lower()).strip("_")
    path = out_dir / f"{slug[:120]}.md"
    path.write_text(hypothesis_card(snap, h, penalties, warnings), encoding="utf-8")
    return path


def write_index(rows: list[dict[str, Any]], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or REPORTS_DIR / "hypotheses"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Hipotezler (skora göre)", "", "| # | Hipotez | Skor | Durum | En zayıf halka | Kart |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['title']} | {r['score']:.3f} | {STATUS_TR.get(r['status'], r['status'])} | "
                     f"{r['weakest']} | [{Path(r['path']).name}]({Path(r['path']).name}) |")
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
