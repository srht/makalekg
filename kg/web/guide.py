"""Başlangıç rehberi: sistemin durumuna bakıp kullanıcıya sıradaki adımı söyler.

Adımlar SPEC'teki fazları izler. Her adımın durumu grafın ve raporların gerçek durumundan hesaplanır;
tek elle tutulan bilgi, kullanıcının Faz 1 sonuçlarını onaylayıp onaylamadığıdır (SPEC: "Faz 2'ye benim
onayım olmadan geçme").
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import DATA_DIR, REPORTS_DIR


def _state_path():
    return DATA_DIR / "guide_state.json"


def load_state() -> dict[str, Any]:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(**updates: Any) -> None:
    state = {**load_state(), **updates}
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


@dataclass
class Step:
    key: str
    title: str
    why: str
    status: str = "todo"  # done | current | todo | running | waiting | attention
    detail: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)
    job: Any = None


def _active_job(jobs, kinds: tuple[str, ...]):
    for j in jobs.list():
        if j.kind in kinds and j.status in ("queued", "running", "waiting"):
            return j
    return None


def build_guide(stats: dict[str, Any] | None, cfg: dict[str, Any], spot: dict[str, Any] | None,
                latest_run: str | None, phase1_checks: list[dict[str, Any]] | None, jobs) -> dict[str, Any]:
    s = stats or {}
    state = load_state()
    steps = [
        Step("setup", "Ayarları kontrol et",
             "Sistem makalelerden bilgi çıkarmak için bir dil modeline ihtiyaç duyar: ya API anahtarı ya da "
             "kopyala–yapıştır modu."),
        Step("phase1", "Tohum makaleyi işle",
             "Başlangıç makalesindeki (Cai ve ark., 2026) deneyler ve iddialar çıkarılıp grafa yazılır. "
             "Her iddia makaledeki kaynak cümlesiyle birlikte saklanır."),
        Step("spotcheck", "Çıkarımları kontrol et",
             "Sistem rastgele 10 iddia seçer ve kaynak cümlesiyle yan yana gösterir. Sen \"doğru / kısmen / yanlış\" "
             "dersin; hata oranı %10'u aşarsa o çıkarımlar kullanılmaz."),
        Step("approve1", "Faz 1 sonuçlarını onayla",
             "Ana iddiaların çıkarılıp çıkarılmadığına ve tutarsızlık raporuna bakılır. Genişlemeye (çok sayıda "
             "makale) ancak sen onaylayınca geçilir."),
        Step("phase2", "Literatürü genişlet",
             "Tohum makalenin kaynakçası, ona atıf yapan çalışmalar ve anahtar kavramlarla bulunan makaleler "
             "aynı şekilde işlenir."),
        Step("infer", "Hipotez üret",
             "Sistem grafta zincirler arar (ör. yükleme → Hsp70 → nöron koruması) ve henüz doğrudan test "
             "edilmemiş ilişkileri hipotez olarak önerir. Hipotezler kanıttan her zaman ayrı tutulur."),
        Step("explore", "Sonuçları incele, soru sor",
             "İddiaları kanıt düzeyleriyle, hipotezleri en zayıf halkalarıyla incele; merak ettiğini düz Türkçe sor."),
    ]
    by = {st.key: st for st in steps}

    # 1. ayarlar
    if cfg.get("llm_ready"):
        by["setup"].status = "done"
        by["setup"].detail = ("Kopyala–yapıştır modu açık." if cfg.get("manual")
                              else f"Model: {cfg.get('llm_model')}.")
    else:
        by["setup"].detail = "Dil modeli ayarlı değil."

    # 2. Faz 1
    seed_done = s.get("claims", 0) > 0
    job = _active_job(jobs, ("ingest",))
    if seed_done:
        by["phase1"].status = "done"
        by["phase1"].detail = f"{s.get('claims', 0)} iddia, {s.get('experiments', 0)} deney çıkarıldı."
    if job is not None:
        by["phase1" if not seed_done else "phase2"].job = job

    # 3. spot-check
    if spot and latest_run and spot.get("run_id") == latest_run:
        if not spot.get("passed") and state.get("spotcheck_accepted_run") == latest_run:
            by["spotcheck"].status = "done"
            by["spotcheck"].detail = f"Hata oranı {spot['error_rate']:.0%} — bu haliyle devam etmeyi seçtin."
        elif spot.get("passed"):
            by["spotcheck"].status = "done"
            by["spotcheck"].detail = f"Hata oranı {spot['error_rate']:.0%} — eşik geçildi."
        else:
            by["spotcheck"].status = "attention"
            by["spotcheck"].detail = (f"Hata oranı {spot['error_rate']:.0%} (eşik %10). İşaretlediğin hatalı çıkarımlar "
                                      "kullanılmayacak. İnceleme kuyruğuna ve iddialara bakıp Faz 1'i "
                                      "\"yeniden çıkar\" ile tekrar çalıştırabilir ya da sonuçları bu haliyle onaylayabilirsin.")
    elif seed_done:
        by["spotcheck"].detail = "Bu çalıştırma için henüz işaretleme yapılmadı."

    # 4. Faz 1 onayı
    if phase1_checks is not None:
        by["approve1"].checks = phase1_checks
    if state.get("phase1_approved"):
        by["approve1"].status = "done"
        by["approve1"].detail = "Onaylandı."

    # 5. Faz 2
    if s.get("papers", 0) > 1:
        by["phase2"].status = "done"
        by["phase2"].detail = f"Grafta {s.get('papers')} makale var."

    # 6. hipotez
    infer_job = _active_job(jobs, ("infer",))
    if infer_job is not None:
        by["infer"].job = infer_job
    if s.get("hypotheses", 0) > 0:
        by["infer"].status = "done"
        by["infer"].detail = f"{s.get('hypotheses')} hipotez üretildi."

    # sıradaki adım: tamamlanmamış ilk adım ("attention" da tamamlanmamış sayılır)
    current = None
    for st in steps:
        if st.status in ("done",):
            continue
        current = st
        break
    if current is None:
        current = by["explore"]
    if current.status == "todo":
        current.status = "current"
    if current.job is not None:
        current.status = "waiting" if current.job.status == "waiting" else "running"
    # explore her zaman açık; hipotez varsa "hazır"
    if by["infer"].status == "done" and current is not by["explore"]:
        by["explore"].detail = "Hazır: hipotezler ve iddialar incelenebilir."

    done = sum(1 for st in steps[:-1] if st.status == "done")
    return {"steps": steps, "current": current, "done": done, "total": len(steps) - 1,
            "has_backtest": (REPORTS_DIR / "backtest.json").exists()}
