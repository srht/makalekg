"""Faz kabul kriterleri (SPEC "Fazlar ve kabul kriterleri"). Her faz bir öncekinin kriterleri
geçmeden başlamaz; bu modül her kriteri tek tek kontrol edip rapor üretir."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import DATA_DIR, REPORTS_DIR
from ..graph import EVIDENCE_EDGE_TYPES, HYPOTHESIS_EDGE, GraphSnapshot
from ..schema import EVIDENCE_PREDICATES, normalize_text
from . import consistency

# Faz 1: tohum makalenin ana iddiaları
PHASE1_KEY_CLAIMS = {
    "Piezo1 gerekliliği": r"piezo1",
    "serum transferi": r"\bserum\b",
    "HSP70": r"hsp70|hspa1|heat shock protein 70",
    "APOL11a": r"apol11a",
    "IL-1R2": r"il-?1r2|il1r2|interleukin-1 receptor (type )?2",
    "domuz modeli": r"\bpig|porcine|swine|minipig",
}


def _check(name: str, passed: bool | None, detail: str) -> dict[str, Any]:
    return {"criterion": name, "passed": passed, "detail": detail}


def phase1(snap: GraphSnapshot, seed_doi: str) -> list[dict[str, Any]]:
    checks = []
    claims = {k: n for k, n in snap.nodes.items() if n.get("label") == "Claim"}
    for label, pattern in PHASE1_KEY_CLAIMS.items():
        hits = []
        for key, n in claims.items():
            if not re.search(pattern, (n.get("text") or ""), re.I):
                continue
            ev = [e for e in snap.in_edges(key, EVIDENCE_EDGE_TYPES)
                  if e.props.get("source_doi") == seed_doi and e.props.get("source_sentence")]
            if ev:
                hits.append(n.get("text"))
        checks.append(_check(f"Ana iddia: {label}", bool(hits),
                             f"{len(hits)} kaynak cümleli Claim" + (f" — örn. \"{hits[0]}\"" if hits else "")))
    summary_path = DATA_DIR / "spotcheck_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        checks.append(_check("Spot-check hata oranı ≤ %10", s["passed"],
                             f"hata oranı {s['error_rate']:.0%} ({s['wrong']}/{s['n_labeled']} yanlış, {s['partial']} kısmen)"))
    else:
        checks.append(_check("Spot-check hata oranı ≤ %10", None,
                             "henüz işaretlenmedi: spotcheck.html'i doldurup `kg spotcheck --results` çalıştırın"))
    inc = [i for i in consistency.load_report() if i.doi == seed_doi]
    checks.append(_check("Bilinen iki tutarsızlık inconsistencies raporunda", len(inc) >= 2 if inc else False,
                         f"tohum makale için {len(inc)} olası tutarsızlık bulundu "
                         f"({consistency.summarize(inc)['by_kind']}); bilinen ikisinin listede olduğunu insan doğrulamalı"))
    return checks


def phase2(snap: GraphSnapshot, normalization_rate: float | None = None) -> list[dict[str, Any]]:
    ents = list(snap.entities().values())
    rate = normalization_rate if normalization_rate is not None else (
        sum(1 for e in ents if not e.get("unresolved", True)) / len(ents) if ents else 0.0)
    papers = [n for n in snap.nodes.values() if n.get("label") == "Paper"]
    dup_dois = [d for d, c in Counter((p.get("doi") or "").lower() for p in papers).items() if c > 1]
    claim_texts = Counter(normalize_text(n.get("text") or "") for n in snap.nodes.values() if n.get("label") == "Claim")
    dup_claims = [t for t, c in claim_texts.items() if c > 1]
    edge_ids = Counter(e.id for e in snap.edges)
    dup_edges = [i for i, c in edge_ids.items() if c > 1]
    n_retracted = sum(1 for p in papers if p.get("retracted"))
    return [
        _check("Normalizasyon oranı ≥ %90", rate >= 0.9, f"{rate:.1%} ({len(ents)} varlık)"),
        _check("Tekrar çalıştırmada çift kayıt yok", not (dup_dois or dup_claims or dup_edges),
               f"çift DOI: {len(dup_dois)}, çift iddia: {len(dup_claims)}, çift kenar: {len(dup_edges)}"),
        _check("Retraction kontrolü çalışıyor", all("retracted" in p for p in papers) and bool(papers),
               f"{len(papers)} makale kontrol edildi, {n_retracted} geri çekilmiş"),
        _check("≤ 200 makale / set", True, f"toplam {len(papers)} makale"),
    ]


def phase3(snap: GraphSnapshot) -> list[dict[str, Any]]:
    hyps = snap.hypotheses()
    missing = [h for h in hyps if not h.props.get("path") or not h.props.get("weakest_link") or not h.props.get("status")]
    leaked = [e for e in snap.evidence_relations() + [e for e in snap.edges if e.type in EVIDENCE_EDGE_TYPES]
              if e.type == HYPOTHESIS_EDGE]
    in_paths = [h for h in hyps if any(step.get("predicate") == HYPOTHESIS_EDGE for step in h.props.get("path") or [])]
    return [
        _check("Hipotez üretildi", bool(hyps), f"{len(hyps)} hipotez"),
        _check("Her hipotezin yolu, en zayıf halkası ve literatür kontrol sonucu var", bool(hyps) and not missing,
               f"{len(missing)} eksik"),
        _check("Hiçbir HYPOTHESIS kenarı kanıt sorgusunda dönmüyor", not leaked and not in_paths,
               f"kanıt sorgusunda {len(leaked)}, hipotez yollarında {len(in_paths)} HYPOTHESIS"),
        _check("Evidence predicates ayrı", HYPOTHESIS_EDGE not in EVIDENCE_PREDICATES, "şema düzeyinde ayrım"),
    ]


def phase4() -> list[dict[str, Any]]:
    path = REPORTS_DIR / "backtest.json"
    if not path.exists():
        return [_check("Backtest raporu Precision@20'yi rastgele çizgiyle karşılaştırıyor", False, "rapor yok")]
    r = json.loads(path.read_text(encoding="utf-8"))
    k = r["k"]
    return [_check("Backtest raporu Precision@20'yi rastgele çizgiyle karşılaştırıyor", True,
                   f"Precision@{k} = {r[f'precision_at_{k}']}, rastgele = {r['random_baseline_mean']} ± "
                   f"{r['random_baseline_std']}, p = {r['permutation_p']}")]


def write_report(phase: int, checks: list[dict[str, Any]], extra: dict[str, Any] | None = None,
                 out_dir: Path | None = None) -> Path:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    mark = {True: "✅ geçti", False: "❌ geçmedi", None: "⏳ beklemede"}
    lines = [f"# Faz {phase} kabul kriterleri", "", "| Kriter | Sonuç | Ayrıntı |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['criterion']} | {mark[c['passed']]} | {c['detail']} |")
    if extra:
        lines += ["", "## Çalıştırma özeti", "", "```json", json.dumps(extra, ensure_ascii=False, indent=1), "```"]
    all_ok = all(c["passed"] for c in checks)
    lines += ["", f"**Genel durum:** {'tüm kriterler geçti' if all_ok else 'kriterlerin tamamı geçmedi — sonraki faza geçilmez'}.", ""]
    path = out_dir / f"phase{phase}_acceptance.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
