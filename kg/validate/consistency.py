"""Sayısal tutarlılık: otomatik kontroller.

1. n_mismatch_extraction — çıkarılan n değerleri o şeklin açıklamasında (legend) geçiyor mu?
2. n_mismatch_text_legend — gövde metninde bir şekle atıfla verilen n, o şeklin legend'indeki n ile uyuşuyor mu?
3. exclusion_conflict — "hiçbir hayvan dışlanmadı" ile dışlama bildirimi aynı makalede çelişiyor mu?
4. protocol_parameter_conflict — aynı protokol parametresi (frekans, kuvvet, döngü, süre, doz...) Methods ile
   legend/sonuç metninde farklı değerlerle mi veriliyor?

Çıktı: reports/inconsistencies.md (+ .json). Bunlar "olası" tutarsızlıklardır; insan kontrol eder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..config import REPORTS_DIR
from ..extract import PaperExtraction
from ..fetch import Document, figure_key

N_RE = re.compile(r"\bn\s*=\s*(\d+(?:\s*(?:[–\-]|to)\s*\d+)?(?:\s*(?:,|and|/)\s*\d+(?:\s*[–\-]\s*\d+)?)*)", re.I)
FIG_REF_RE = re.compile(r"(?:extended\s+data\s+|supplementary\s+)?fig(?:ure)?s?\.?\s*s?\d+[a-z]?", re.I)
SENT_SPLIT = re.compile(r"(?<!\bFig\.)(?<!\bal\.)(?<!\bvs\.)(?<=[.!?])\s+(?=[A-Z(])")

NO_EXCLUSION_RE = re.compile(r"\b(no (?:animals?|mice|rats|samples?|data(?: points)?|subjects?) (?:were|was) excluded|"
                             r"none (?:of the animals )?(?:were|was) excluded|without exclusion)\b", re.I)
EXCLUSION_RE = re.compile(r"\b(?:were|was) (?:excluded|removed from (?:the )?analys[ie]s)|\bexclusion criteri", re.I)

PARAM_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>Hz|N|cycles?|min(?:utes)?|s(?:econds)?|h(?:ours)?|days?|weeks?|mg/kg|µg|μg|ng|"
    r"times(?: per| a)? (?:day|week)|%)\b", re.I)
PARAM_KEYWORDS = {
    "frequency": r"\bfrequenc|\bHz\b",
    "force/load": r"\b(force|load|loading|compression|peak)\b",
    "cycles": r"\bcycles?\b",
    "duration": r"\b(duration|for \d+|lasted|session)\b",
    "dose": r"\b(dose|dosage|mg/kg|injected|administered)\b",
    "schedule": r"\b(per week|per day|daily|times)\b",
}
UNIT_NORM = {"cycle": "cycles", "minutes": "min", "minute": "min", "seconds": "s", "second": "s", "hours": "h",
             "hour": "h", "day": "days", "week": "weeks", "μg": "µg"}


class Inconsistency(BaseModel):
    doi: str
    kind: str
    figure: str | None = None
    detail: str
    evidence_a: str
    evidence_b: str | None = None


def parse_n_values(text: str) -> set[int]:
    values: set[int] = set()
    for m in N_RE.finditer(text):
        for part in re.split(r"\s*(?:,|and|/)\s*", m.group(1)):
            rng = re.split(r"\s*(?:[–\-]|to)\s*", part.strip())
            nums = [int(x) for x in rng if x.isdigit()]
            if len(nums) == 2 and nums[0] <= nums[1] <= nums[0] + 50:
                values.update(range(nums[0], nums[1] + 1))
            else:
                values.update(nums)
    return values


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.replace("\n", " ")) if s.strip()]


def _short(s: str, n: int = 300) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def check_extraction_n(doc: Document, extraction: PaperExtraction | None) -> list[Inconsistency]:
    out = []
    if extraction is None:
        return out
    for rec in extraction.accepted:
        x = rec.claim.experiment
        if not x.n_per_group or not x.figure:
            continue
        legend = doc.legend_for(x.figure)
        if legend is None:
            continue
        legend_n = parse_n_values(legend.text)
        missing = [n for n in x.n_per_group if n not in legend_n]
        if legend_n and missing:
            out.append(Inconsistency(doi=doc.doi, kind="n_mismatch_extraction", figure=x.figure,
                                     detail=f"Çıkarılan n={x.n_per_group} değerlerinden {missing} legend'de yok "
                                            f"(legend n: {sorted(legend_n)}).",
                                     evidence_a=_short(rec.claim.source_sentence), evidence_b=_short(legend.text)))
    return out


def check_text_vs_legend(doc: Document) -> list[Inconsistency]:
    out = []
    for sec in doc.sections:
        if sec.kind == "legend":
            continue
        for sent in sentences(sec.text):
            n_vals = parse_n_values(sent)
            if not n_vals:
                continue
            for ref in {figure_key(m.group(0)) for m in FIG_REF_RE.finditer(sent)} - {None}:
                legend = next((l for l in doc.legends() if figure_key(l.figure) == ref), None)
                if legend is None:
                    continue
                legend_n = parse_n_values(legend.text)
                diff = sorted(n_vals - legend_n)
                if legend_n and diff:
                    out.append(Inconsistency(doi=doc.doi, kind="n_mismatch_text_legend", figure=legend.figure,
                                             detail=f"{sec.name} bölümünde n={sorted(n_vals)}, legend'de n={sorted(legend_n)}; "
                                                    f"legend'de olmayan: {diff}.",
                                             evidence_a=_short(sent), evidence_b=_short(legend.text)))
    return out


def check_exclusions(doc: Document) -> list[Inconsistency]:
    no_excl, excl = [], []
    for sec in doc.sections:
        for sent in sentences(sec.text):
            if NO_EXCLUSION_RE.search(sent):
                no_excl.append((sec.name, sent))
            elif EXCLUSION_RE.search(sent) and not re.search(r"\bexclusion criteria (?:were|was) (?:not|none)", sent, re.I):
                excl.append((sec.name, sent))
    out = []
    for (s1, a) in no_excl:
        for (s2, b) in excl:
            out.append(Inconsistency(doi=doc.doi, kind="exclusion_conflict",
                                     detail=f"'{s1}' dışlama olmadığını söylüyor, '{s2}' dışlama bildiriyor.",
                                     evidence_a=_short(a), evidence_b=_short(b)))
    return out


def _params(text: str) -> list[tuple[str, str, float, str]]:
    """(parametre adı, birim, değer, cümle)"""
    out = []
    for sent in sentences(text):
        for m in PARAM_RE.finditer(sent):
            unit = m.group("unit").lower()
            unit = UNIT_NORM.get(unit, unit)
            window = sent[max(0, m.start() - 80): m.end() + 40]
            for name, pat in PARAM_KEYWORDS.items():
                if re.search(pat, window, re.I):
                    out.append((name, unit, float(m.group("val")), sent))
                    break
    return out


def check_protocol(doc: Document, topic_pattern: str = r"\b(load|loading|compress|tibia|tibial|DCTAL)\w*") -> list[Inconsistency]:
    """Methods'taki ve diğer bölümlerdeki aynı (parametre, birim) değerlerini karşılaştırır;
    yalnızca konu kalıbına (varsayılan: tibial yükleme protokolü) uyan cümlelerde."""
    methods = [p for s in doc.sections if s.kind == "methods" for p in _params(s.text) if re.search(topic_pattern, p[3], re.I)]
    others = [p for s in doc.sections if s.kind in ("legend", "body", "abstract") for p in _params(s.text)
              if re.search(topic_pattern, p[3], re.I)]
    out = []
    seen = set()
    for name, unit, val, sent in others:
        m_vals = {(v, s) for n, u, v, s in methods if n == name and u == unit}
        if m_vals and val not in {v for v, _ in m_vals}:
            ref_val, ref_sent = sorted(m_vals)[0]
            key = (name, unit, val, ref_val)
            if key in seen:
                continue
            seen.add(key)
            out.append(Inconsistency(doi=doc.doi, kind="protocol_parameter_conflict",
                                     detail=f"{name}: Methods'ta {sorted({v for v, _ in m_vals})} {unit}, başka bölümde {val} {unit}.",
                                     evidence_a=_short(ref_sent), evidence_b=_short(sent)))
    return out


def check_document(doc: Document, extraction: PaperExtraction | None = None) -> list[Inconsistency]:
    if not doc.fulltext:
        return []
    return (check_extraction_n(doc, extraction) + check_text_vs_legend(doc) + check_exclusions(doc) +
            check_protocol(doc))


def write_report(items: list[Inconsistency], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inconsistencies.json").write_text(json.dumps([i.model_dump() for i in items], ensure_ascii=False,
                                                             indent=1), encoding="utf-8")
    kinds = {"n_mismatch_extraction": "Çıkarılan n legend ile uyuşmuyor",
             "n_mismatch_text_legend": "Metin ve legend'deki n farklı",
             "exclusion_conflict": "Dışlama bildirimi çelişkisi",
             "protocol_parameter_conflict": "Protokol parametresi çelişkisi"}
    lines = ["# Tutarsızlık raporu", "",
             f"Toplam {len(items)} olası tutarsızlık. Bunlar otomatik kontrollerdir; her biri insan tarafından "
             "kaynak metinle doğrulanmalıdır.", ""]
    for i, it in enumerate(items, 1):
        lines += [f"## {i}. {kinds.get(it.kind, it.kind)} — doi:{it.doi}" + (f" · {it.figure}" if it.figure else ""),
                  "", it.detail, "", f"> {it.evidence_a}", ""]
        if it.evidence_b:
            lines += [f"> {it.evidence_b}", ""]
    path = out_dir / "inconsistencies.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_report(out_dir: Path | None = None) -> list[Inconsistency]:
    path = (out_dir or REPORTS_DIR) / "inconsistencies.json"
    if not path.exists():
        return []
    return [Inconsistency.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def merge_reports(existing: list[Inconsistency], new: list[Inconsistency], doi: str) -> list[Inconsistency]:
    """Aynı DOI'nin eski kayıtlarını yenileriyle değiştirir (idempotent)."""
    return [i for i in existing if i.doi != doi] + new


__all__ = ["check_document", "write_report", "load_report", "merge_reports", "Inconsistency", "parse_n_values"]


def summarize(items: list[Inconsistency]) -> dict[str, Any]:
    out: dict[str, int] = {}
    for i in items:
        out[i.kind] = out.get(i.kind, 0) + 1
    return {"total": len(items), "by_kind": out}
