import pytest

from kg.graph import Edge, GraphSnapshot
from kg.score import Link, chain_confidence, describe_experiment, evidence_strength, final_score, w_n

CFG = None


def _snap():
    nodes = {
        "p1": {"label": "Paper", "doi": "p1", "lab": "A", "year": 2020},
        "x1": {"label": "Experiment", "design": "in vivo", "species": "mouse", "n_per_group": [9, 10], "blinded": False},
        "c1": {"label": "Claim", "text": "A increases B"},
        "A": {"label": "Intervention", "name": "A"}, "B": {"label": "Gene", "name": "B", "species": "mouse"},
        "C": {"label": "Outcome", "name": "C"},
    }
    edges = [Edge("REPORTS", "p1", "x1"), Edge("SUPPORTS", "x1", "c1", {"extraction_confidence": 0.9}),
             Edge("INCREASES", "A", "B", {"claim_id": "c1", "experiment_id": "x1", "species": "mouse",
                                          "extraction_confidence": 0.9, "stance": "affirmed"}),
             Edge("PROTECTS", "B", "C", {"claim_id": "c1", "experiment_id": "x1", "species": "human",
                                         "extraction_confidence": 0.9, "stance": "affirmed"})]
    return GraphSnapshot(nodes, edges)


def test_evidence_strength_formula():
    snap = _snap()
    s = evidence_strength(snap, "x1", "c1", {"extraction_confidence": 0.9})
    # 0.5 (in vivo) · 0.6 (kemirgen) · 0.6 (n=9) · 0.7 (körlemesiz) · 0.6 (replikasyon yok) · 0.9
    assert s.E == pytest.approx(0.5 * 0.6 * 0.6 * 0.7 * 0.6 * 0.9, abs=1e-4)
    assert "fare" in s.explanation and "n = 9–10" in s.explanation and "körlemesiz" in s.explanation
    human_verified = evidence_strength(snap, "x1", "c1", {"extraction_confidence": 0.1, "verified_by_human": True})
    assert human_verified.factors["c_extract"] == 1.0


def test_w_n_tiers():
    assert w_n([25]) == 1.0 and w_n([12, 30]) == 0.8 and w_n([5]) == 0.6 and w_n([4]) == 0.4 and w_n(None) == 0.4


def test_chain_confidence_penalties():
    snap = _snap()
    from kg.score import relation_score
    links = [Link(e.src, e.type, e.dst, e, relation_score(snap, e)) for e in snap.evidence_relations()]
    ch = chain_confidence(snap, links)
    E = links[0].score.E
    assert ch.penalties == {"species_mismatch": 0.6}
    assert ch.C == pytest.approx(E * 0.6 * 0.8, abs=1e-4)
    assert final_score(ch.C, None) == ch.C
    assert final_score(0.5, 1.0) == pytest.approx(0.6 * 0.5 + 0.4)


def test_describe_experiment_plain_turkish():
    assert describe_experiment({"figure": "Fig. 5", "panel": "h", "species": "mouse", "n_per_group": [9, 10]}) == \
        "Fig. 5 h, fare, n = 9–10, körlemesiz veya belirtilmemiş, tek laboratuvar"
