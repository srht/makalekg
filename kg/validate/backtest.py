"""Backtesting: graf kesim tarihinden önceki makalelerle kurulur, hipotezler üretilir; sonra kesim
sonrası makalelerde bu hipotezlerin test edilip edilmediğine bakılır.

Metrikler: Precision@20, Refuted@20, rastgele çizgi (aynı tip düğüm çiftlerinden rastgele 20 hipotez),
kalibrasyon (skor dilimleri ile gerçekleşme oranı; monoton artan olmalı).
Sayısal hedef eşikleri ilk backtest sonucuna göre belirlenecek (SPEC: açık soru).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..config import REPORTS_DIR, scoring_config
from ..graph import GraphSnapshot
from ..infer.rules import has_direct_evidence, rule_candidates
from ..infer.run import merge_queue, score_candidate
from ..schema import EVIDENCE_PREDICATES

OPPOSITE = {"INCREASES": "DECREASES", "DECREASES": "INCREASES"}


def split_papers(snap: GraphSnapshot, cutoff_year: int) -> tuple[set[str], set[str]]:
    pre, post = set(), set()
    for k, n in snap.nodes.items():
        if n.get("label") != "Paper" or n.get("year") is None:
            continue
        (pre if int(n["year"]) < cutoff_year else post).add(k)
    return pre, post


def outcome(snap: GraphSnapshot, s: str, p: str, o: str, post_dois: set[str]) -> str:
    """Kesim sonrası makalelerden gelen doğrudan kanıta göre: supported | refuted | untested."""
    sup = ref = False
    for e in snap.out_edges(s, EVIDENCE_PREDICATES):
        if e.dst != o or e.props.get("source_doi") not in post_dois:
            continue
        stance = e.props.get("stance", "affirmed")
        if stance == "negated" or e.type == OPPOSITE.get(p):
            ref = True
        elif e.type == p or p in ("MEDIATES", "ASSOCIATED"):
            sup = True
    return "supported" if sup else ("refuted" if ref else "untested")


def random_baseline(snap_pre: GraphSnapshot, snap_full: GraphSnapshot, top: list[tuple[str, str, str]],
                    post: set[str], k: int = 20, trials: int = 200, seed: int = 7) -> dict[str, Any]:
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = {}
    for key, n in snap_pre.entities().items():
        by_type.setdefault(n["label"], []).append(key)
    rates = []
    for _ in range(trials):
        hits = 0
        n = 0
        for (s, p, o) in top[:k]:
            ts, to = snap_pre.label(s), snap_pre.label(o)
            pool_s, pool_o = by_type.get(ts, []), by_type.get(to, [])
            for _attempt in range(20):
                a, b = rng.choice(pool_s), rng.choice(pool_o)
                if a != b and not has_direct_evidence(snap_pre, a, b):
                    break
            else:
                continue
            n += 1
            hits += outcome(snap_full, a, p, b, post) == "supported"
        if n:
            rates.append(hits / n)
    return {"mean": mean(rates) if rates else 0.0, "std": pstdev(rates) if len(rates) > 1 else 0.0,
            "trials": len(rates), "rates": rates}


def calibration(scored: list[tuple[float, str]], bins: int = 5) -> list[dict[str, Any]]:
    if not scored:
        return []
    scored = sorted(scored)
    size = max(1, len(scored) // bins)
    out = []
    for i in range(0, len(scored), size):
        chunk = scored[i:i + size]
        out.append({"score_min": chunk[0][0], "score_max": chunk[-1][0], "n": len(chunk),
                    "supported_rate": sum(1 for _, o in chunk if o == "supported") / len(chunk)})
    return out


def run_backtest(snap: GraphSnapshot, cutoff_year: int, k: int = 20, cfg: dict[str, Any] | None = None,
                 out_dir: Path | None = None) -> dict[str, Any]:
    cfg = cfg or scoring_config()
    pre, post = split_papers(snap, cutoff_year)
    snap_pre = snap.filtered(pre)
    queue = merge_queue(rule_candidates(snap_pre, scoring=cfg))
    scored = sorted((score_candidate(snap_pre, c, cfg) for c in queue), key=lambda s: s.hypothesis.score, reverse=True)
    triples = [(s.hypothesis.subject_key, s.hypothesis.predicate, s.hypothesis.object_key) for s in scored]
    outcomes = [outcome(snap, *t, post) for t in triples]
    top = outcomes[:k]
    n_top = len(top)
    baseline = random_baseline(snap_pre, snap, triples, post, k)
    precision = sum(o == "supported" for o in top) / n_top if n_top else 0.0
    refuted = sum(o == "refuted" for o in top) / n_top if n_top else 0.0
    p_perm = (sum(1 for r in baseline["rates"] if r >= precision) + 1) / (len(baseline["rates"]) + 1) if baseline["rates"] else None
    calib = calibration([(s.hypothesis.score, o) for s, o in zip(scored, outcomes)])
    rates = [c["supported_rate"] for c in calib]
    result = {
        "cutoff_year": cutoff_year, "n_pre_papers": len(pre), "n_post_papers": len(post),
        "n_hypotheses": len(scored), "k": k, "n_top": n_top,
        f"precision_at_{k}": round(precision, 3), f"refuted_at_{k}": round(refuted, 3),
        "random_baseline_mean": round(baseline["mean"], 3), "random_baseline_std": round(baseline["std"], 3),
        "permutation_p": None if p_perm is None else round(p_perm, 4),
        "calibration": calib, "calibration_monotonic": all(a <= b for a, b in zip(rates, rates[1:])),
        "top": [{"subject": snap.name(t[0]), "predicate": t[1], "object": snap.name(t[2]),
                 "score": s.hypothesis.score, "outcome": o} for t, s, o in zip(triples[:k], scored[:k], top)],
    }
    write_report(result, out_dir)
    return result


def write_report(r: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    k = r["k"]
    lines = [f"# Backtest raporu (kesim: {r['cutoff_year']})", "",
             f"Kesim öncesi {r['n_pre_papers']} makale ile graf kuruldu, {r['n_hypotheses']} hipotez üretildi; "
             f"kesim sonrası {r['n_post_papers']} makaleye bakıldı.", "",
             "| Metrik | Değer |", "|---|---|",
             f"| Precision@{k} (ilk {r['n_top']} hipotez) | {r[f'precision_at_{k}']:.3f} |",
             f"| Refuted@{k} | {r[f'refuted_at_{k}']:.3f} |",
             f"| Rastgele çizgi (ortalama ± sd) | {r['random_baseline_mean']:.3f} ± {r['random_baseline_std']:.3f} |",
             f"| Permütasyon p (rastgele ≥ precision) | {r['permutation_p']} |",
             f"| Kalibrasyon monoton artan mı | {'evet' if r['calibration_monotonic'] else 'hayır'} |", "",
             "## Kalibrasyon", "", "| Skor aralığı | n | Gerçekleşme oranı |", "|---|---|---|"]
    for c in r["calibration"]:
        lines.append(f"| {c['score_min']:.3f}–{c['score_max']:.3f} | {c['n']} | {c['supported_rate']:.2f} |")
    lines += ["", f"## İlk {k} hipotez", "", "| Hipotez | Skor | Sonuç |", "|---|---|---|"]
    for t in r["top"]:
        lines.append(f"| {t['subject']} -{t['predicate']}-> {t['object']} | {t['score']:.3f} | {t['outcome']} |")
    lines += ["", "_Sayısal hedef eşikleri bu ilk sonuca göre belirlenecek (açık soru)._", ""]
    path = out_dir / "backtest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
