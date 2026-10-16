"""Başlangıç rehberi: durumdan sıradaki adımın doğru hesaplanması."""
from types import SimpleNamespace

from kg.web import guide


class Jobs:
    def __init__(self, items=()):
        self.items = list(items)

    def list(self):
        return self.items


def _g(stats=None, cfg=None, spot=None, run="r1", checks=None, jobs=()):
    return guide.build_guide(stats, cfg or {"llm_ready": True, "manual": True}, spot, run, checks, Jobs(jobs))


def test_progression(tmp_data, monkeypatch):
    monkeypatch.setattr(guide, "DATA_DIR", tmp_data / "data")
    monkeypatch.setattr(guide, "REPORTS_DIR", tmp_data / "reports")
    assert _g(cfg={"llm_ready": False}).get("current").key == "setup"
    assert _g().get("current").key == "phase1"
    seeded = {"claims": 3, "experiments": 3, "papers": 1, "hypotheses": 0}
    assert _g(seeded).get("current").key == "spotcheck"
    # başka bir çalıştırmanın spot-check'i sayılmaz
    assert _g(seeded, spot={"run_id": "eski", "passed": True, "error_rate": 0}).get("current").key == "spotcheck"
    bad = {"run_id": "r1", "passed": False, "error_rate": 0.3}
    g = _g(seeded, spot=bad)
    assert g["current"].key == "spotcheck" and g["current"].status == "attention"
    guide.save_state(spotcheck_accepted_run="r1")
    g = _g(seeded, spot=bad, checks=[{"criterion": "x", "passed": False, "detail": ""}])
    assert g["current"].key == "approve1" and g["current"].checks
    guide.save_state(phase1_approved=True)
    assert _g(seeded, spot=bad).get("current").key == "phase2"
    expanded = {**seeded, "papers": 6}
    assert _g(expanded, spot=bad).get("current").key == "infer"
    g = _g({**expanded, "hypotheses": 4}, spot=bad)
    assert g["current"].key == "explore" and g["done"] == g["total"]


def test_running_job_shown_on_current_step(tmp_data, monkeypatch):
    monkeypatch.setattr(guide, "DATA_DIR", tmp_data / "data")
    job = SimpleNamespace(kind="ingest", status="waiting", id="j1", title="Faz 1")
    g = _g(jobs=[job])
    assert g["current"].key == "phase1" and g["current"].status == "waiting" and g["current"].job is job
