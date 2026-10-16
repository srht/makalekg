import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kg.fetch import segment_plain_text  # noqa: E402
from kg.llm import FakeLLM  # noqa: E402
from kg.schema import ExtractionResult, VerificationResult  # noqa: E402

SEED_DOI = "10.1038/s41593-026-02422-w"


def _claim(text, pred, subj, obj, fig, panel, sentence, n=None, conf=0.9, rel="SUPPORTS", polarity="+"):
    return {"claim": text, "polarity": polarity, "predicate": pred, "evidence_relation": rel,
            "subject": subj, "object": obj,
            "experiment": {"figure": fig, "panel": panel, "design": "in vivo", "species": "mouse", "sex": None,
                           "n_per_group": n, "blinded": None, "randomized": None, "tracer_type": None,
                           "model_name": "CCI-TBI"},
            "source_sentence": sentence, "p_value": None, "effect_size": None, "extraction_confidence": conf}


LOADING = {"type": "Intervention", "name": "tibial loading (DCTAL)", "symbol": None, "species": None, "measure": None,
           "direction": None, "intervention_type": "mechanical", "dose": None, "schedule": None, "severity": None}
HSP70 = {"type": "Gene", "name": "Hsp70", "symbol": "Hsp70", "species": "mouse", "measure": None, "direction": None,
         "intervention_type": None, "dose": None, "schedule": None, "severity": None}
PIEZO1 = {**HSP70, "name": "Piezo1", "symbol": "Piezo1"}
NEURON = {"type": "Outcome", "name": "neuronal survival", "symbol": None, "species": None, "measure": "NeuN+ cell count",
          "direction": "improved", "intervention_type": None, "dose": None, "schedule": None, "severity": None}
MOTOR = {**NEURON, "name": "motor function", "measure": "pole test"}

FAKE_CLAIMS = [
    _claim("Tibial loading increases serum Hsp70 after TBI in mice", "INCREASES", LOADING, HSP70, "Fig. 1", "c",
           "DCTAL increased serum Hsp70 levels in TBI mice (Fig. 1c).", n=[8]),
    _claim("Recombinant Hsp70 protects against neuronal loss after TBI in mice", "PROTECTS", HSP70, NEURON, "Fig. 3", "a",
           "Recombinant Hsp70 reduced neuronal loss in TBI mice (Fig. 3a).", n=[6]),
    _claim("The motor benefit of tibial loading after TBI requires osteocyte Piezo1", "REQUIRES", MOTOR, PIEZO1,
           "Fig. 2", "d-h",
           "DCTAL ameliorated motor and cognitive deficits in TBI-Piezo1fl/fl mice, but not in TBI-Piezo1Sost-Cre/ERT2 mice (Fig. 2d-h).",
           n=[14, 14, 14, 17]),
    # Kaynak cümlesi metinde olmayan (uydurma) bir çıkarım — reddedilmeli
    _claim("Tibial loading cures Alzheimer's disease", "PROTECTS", LOADING, MOTOR, "Fig. 9", None,
           "Loading cured Alzheimer's disease in all patients.", conf=0.4),
]


def responder(system, user, schema):
    if schema is ExtractionResult:
        if "Section: Results" in user:
            return {"claims": FAKE_CLAIMS}
        return {"claims": []}
    if schema is VerificationResult:
        import json
        items = json.loads(user.split("=== ITEMS ===\n", 1)[1])
        return {"items": [{"index": it["index"], "label": "supported", "reason": "ok"} for it in items]}
    raise AssertionError(schema)


@pytest.fixture
def fake_llm():
    return FakeLLM(responder)


@pytest.fixture
def seed_doc():
    text = (ROOT / "tests" / "fixtures" / "synthetic_paper.txt").read_text()
    return segment_plain_text(text, SEED_DOI)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    for mod in ("kg.config", "kg.extract", "kg.pipeline", "kg.validate.spotcheck", "kg.validate.consistency",
                "kg.validate.acceptance", "kg.validate.backtest", "kg.report", "kg.service", "kg.llm", "kg.fetch", "kg.okg.store", "kg.graph", "kg.normalize", "kg.http"):
        m = __import__(mod, fromlist=["x"])
        if hasattr(m, "DATA_DIR"):
            monkeypatch.setattr(m, "DATA_DIR", tmp_path / "data")
        if hasattr(m, "REPORTS_DIR"):
            monkeypatch.setattr(m, "REPORTS_DIR", tmp_path / "reports")
        if hasattr(m, "ROOT") and mod != "kg.config":
            monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "reports").mkdir()
    return tmp_path
