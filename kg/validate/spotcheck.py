"""Çıkarım doğruluğu (spot-check): her ingest sonrası rastgele 10 kenar, kaynak cümlesiyle yan yana
spotcheck.html sayfasında gösterilir; kullanıcı "doğru / kısmen / yanlış" işaretleyip sonucu JSON
olarak indirir. `kg spotcheck --results <json>` hata oranını hesaplar; %10'u aşarsa o partinin
kenarları needs_review olur, "doğru" işaretlenenler verified_by_human olur.

Hata oranı = yanlış / işaretlenen (kısmen ayrıca raporlanır).
"""
from __future__ import annotations

import html
import json
import random
from pathlib import Path
from typing import Any

from ..config import DATA_DIR, ROOT
from ..graph import EVIDENCE_EDGE_TYPES, GraphSnapshot, GraphStore
from ..schema import EVIDENCE_PREDICATES

ERROR_THRESHOLD = 0.10


def latest_run_id(snapshot: GraphSnapshot) -> str | None:
    runs = sorted({e.props.get("ingest_run_id") for e in snapshot.edges if e.props.get("ingest_run_id")})
    return runs[-1] if runs else None


def sample_edges(snapshot: GraphSnapshot, n: int = 10, run_id: str | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    run_id = run_id or latest_run_id(snapshot)
    pool = [e for e in snapshot.edges if e.type in EVIDENCE_EDGE_TYPES
            and (run_id is None or e.props.get("ingest_run_id") == run_id)]
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n, len(pool)))
    items = []
    for e in picked:
        claim = snapshot.nodes.get(e.dst, {})
        exp = snapshot.nodes.get(e.src, {})
        subj = next((snapshot.name(r.dst) for r in snapshot.out_edges(e.dst, ["SUBJECT"])), "?")
        obj = next((snapshot.name(r.dst) for r in snapshot.out_edges(e.dst, ["OBJECT"])), "?")
        rels = [r for r in snapshot.edges if r.type in EVIDENCE_PREDICATES and r.props.get("claim_id") == e.dst
                and r.props.get("source_doi") == e.props.get("source_doi")]
        items.append({
            "edge_id": e.id, "related_edge_ids": [r.id for r in rels], "relation": e.type,
            "claim": claim.get("text"), "polarity": claim.get("polarity"), "predicate": claim.get("predicate"),
            "subject": subj, "object": obj, "doi": e.props.get("source_doi"), "figure": e.props.get("figure"),
            "source_sentence": e.props.get("source_sentence"), "p_value": e.props.get("p_value"),
            "experiment": {k: exp.get(k) for k in ("design", "species", "sex", "n_per_group", "blinded", "randomized",
                                                   "model_name")},
            "extraction_confidence": e.props.get("extraction_confidence"),
            "verification": e.props.get("verification"),
        })
    return items


def render_html(items: list[dict[str, Any]], run_id: str | None) -> str:
    cards = []
    for i, it in enumerate(items):
        exp = ", ".join(f"{k}={v}" for k, v in it["experiment"].items() if v is not None) or "—"
        cards.append(f"""
<article class="card" data-edge="{html.escape(it['edge_id'])}">
  <div class="meta">#{i + 1} · doi:{html.escape(str(it['doi']))} · {html.escape(str(it['figure'] or 'şekil yok'))} · {it['relation']}</div>
  <div class="cols">
    <div><h3>Çıkarılan iddia</h3>
      <p class="claim">{html.escape(str(it['claim']))}</p>
      <p><b>{html.escape(it['subject'])}</b> —{html.escape(str(it['predicate']))} ({html.escape(str(it['polarity']))})→ <b>{html.escape(it['object'])}</b></p>
      <p class="small">Deney: {html.escape(exp)}<br>p: {html.escape(str(it['p_value']))} · güven: {it['extraction_confidence']} · öz-doğrulama: {it['verification']}</p>
    </div>
    <div><h3>Kaynak cümle</h3><blockquote>{html.escape(str(it['source_sentence']))}</blockquote></div>
  </div>
  <fieldset>
    <label><input type="radio" name="e{i}" value="correct"> doğru</label>
    <label><input type="radio" name="e{i}" value="partial"> kısmen</label>
    <label><input type="radio" name="e{i}" value="wrong"> yanlış</label>
    <input type="text" name="n{i}" placeholder="not (isteğe bağlı)">
  </fieldset>
</article>""")
    data = json.dumps({"run_id": run_id, "items": [{"edge_id": it["edge_id"], "related_edge_ids": it["related_edge_ids"]}
                                                   for it in items]})
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spot-check</title>
<style>
:root {{ --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#e3e2dc; --accent:#2f5d50; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#ecebe6; --muted:#a09f99; --card:#1f1f1d; --line:#33332f; --accent:#7fb8a6; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:14px 0; }}
.meta,.small {{ color:var(--muted); font-size:13px; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width:700px) {{ .cols {{ grid-template-columns:1fr; }} }}
h3 {{ font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:6px 0; }}
blockquote {{ margin:0; padding:8px 12px; border-left:3px solid var(--accent); }}
fieldset {{ border:0; padding:8px 0 0; display:flex; gap:16px; flex-wrap:wrap; align-items:center; }}
input[type=text] {{ flex:1; min-width:160px; padding:4px 8px; background:transparent; color:var(--fg); border:1px solid var(--line); border-radius:6px; }}
button {{ background:var(--accent); color:var(--bg); border:0; border-radius:8px; padding:10px 16px; font-size:15px; cursor:pointer; }}
</style></head><body>
<h1>Spot-check — ingest {html.escape(str(run_id))}</h1>
<p>Her kenarı kaynak cümlesiyle karşılaştırın. <b>doğru</b>: cümle iddiayı ve alanları tam söylüyor ·
<b>kısmen</b>: ana fikir doğru ama bir alan abartılı/eksik · <b>yanlış</b>: cümle bunu söylemiyor.
Bitince "Sonuçları indir" düğmesine basın ve dosyayı <code>kg spotcheck --results dosya.json</code> ile verin.</p>
{''.join(cards)}
<p><button id="dl">Sonuçları indir</button> <span id="status" class="small"></span></p>
<script>
const DATA = {data};
document.getElementById('dl').addEventListener('click', () => {{
  const out = {{run_id: DATA.run_id, items: DATA.items.map((it, i) => {{
    const r = document.querySelector('input[name=e' + i + ']:checked');
    return {{...it, label: r ? r.value : null, note: document.querySelector('input[name=n' + i + ']').value}};
  }})}};
  const missing = out.items.filter(x => !x.label).length;
  document.getElementById('status').textContent = missing ? missing + ' kenar işaretlenmedi' : 'tamam';
  const blob = new Blob([JSON.stringify(out, null, 1)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'spotcheck_results_' + (DATA.run_id || 'run') + '.json'; a.click();
}});
</script></body></html>"""


def build_spotcheck(store: GraphStore, n: int = 10, run_id: str | None = None, seed: int | None = None,
                    out: Path | None = None) -> tuple[Path, list[dict[str, Any]]]:
    snap = store.snapshot()
    run_id = run_id or latest_run_id(snap)
    items = sample_edges(snap, n, run_id, seed)
    out = out or ROOT / "spotcheck.html"
    out.write_text(render_html(items, run_id), encoding="utf-8")
    (DATA_DIR / "spotcheck_sample.json").write_text(json.dumps({"run_id": run_id, "items": items}, ensure_ascii=False,
                                                               indent=1), encoding="utf-8")
    return out, items


def apply_results(store: GraphStore, results: dict[str, Any]) -> dict[str, Any]:
    items = [it for it in results.get("items", []) if it.get("label")]
    n = len(items)
    wrong = sum(1 for it in items if it["label"] == "wrong")
    partial = sum(1 for it in items if it["label"] == "partial")
    rate = wrong / n if n else 0.0
    verified = [eid for it in items if it["label"] == "correct" for eid in [it["edge_id"], *it.get("related_edge_ids", [])]]
    store.mark_verified(verified)
    flagged = 0
    if rate > ERROR_THRESHOLD:
        run_id = results.get("run_id")
        snap = store.snapshot()
        batch = [e.id for e in snap.edges if e.props.get("ingest_run_id") == run_id]
        flagged = store.mark_needs_review(batch)
    summary = {"run_id": results.get("run_id"), "n_labeled": n, "wrong": wrong, "partial": partial,
               "error_rate": round(rate, 3), "threshold": ERROR_THRESHOLD, "passed": rate <= ERROR_THRESHOLD,
               "flagged_needs_review": flagged, "verified_by_human": len(verified)}
    (DATA_DIR / "spotcheck_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if hasattr(store, "flush"):
        store.flush()
    return summary
