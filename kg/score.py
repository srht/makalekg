"""Güven puanlama (SPEC "Güven puanlama"); katsayılar config/scoring.yaml'dan.

E = w_design · w_species · w_n · w_blind · w_repl · c_extract          (kanıt gücü, 0–1)
C = min_i(E_i) · Π_j p_j · 0.8^(L-1)                                   (zincir güveni)
S = 0.6 · C + 0.4 · P   (embedding tahmini P varsa), yoksa S = C       (nihai skor)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import scoring_config
from .graph import Edge, GraphSnapshot

SPECIES_TR = {"mouse": "fare", "rat": "sıçan", "human": "insan", "pig": "domuz", "primate": "primat",
              "cell_line": "hücre hattı"}
DESIGN_KEY = {"in vivo": "in_vivo", "in vitro": "in_vitro", "ex vivo": "ex_vivo",
              "rct human": "rct_human", "observational human": "observational_human"}


def _cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    return cfg or scoring_config()


def w_design(design: str | None, cfg: dict[str, Any] | None = None) -> float:
    c = _cfg(cfg)["w_design"]
    return c.get(DESIGN_KEY.get((design or "").lower(), "unknown"), c["unknown"])


def w_species(species: str | None, design: str | None = None, cfg: dict[str, Any] | None = None) -> float:
    c = _cfg(cfg)["w_species"]
    if (design or "").lower() == "in vitro" and species != "human":
        return c["cell_line"]
    if species in ("mouse", "rat"):
        return c["rodent"]
    return c.get(species or "unknown", c["unknown"])


def w_n(n_per_group: list[int] | None, cfg: dict[str, Any] | None = None) -> float:
    c = _cfg(cfg)
    if not n_per_group:
        return c["w_n_unknown"]
    n = min(n_per_group)
    for tier in c["w_n"]:
        if n >= tier["min"]:
            return tier["w"]
    return c["w_n"][-1]["w"]


def w_blind(blinded: bool | None, cfg: dict[str, Any] | None = None) -> float:
    c = _cfg(cfg)["w_blind"]
    return c["blinded"] if blinded else c["not_blinded"]


def w_repl(n_independent_labs: int, contradicted: bool, cfg: dict[str, Any] | None = None) -> float:
    c = _cfg(cfg)["w_repl"]
    w = c["none"] if n_independent_labs == 0 else (c["one"] if n_independent_labs == 1 else c["two_plus"])
    return w * (c["contradicted_factor"] if contradicted else 1.0)


@dataclass
class EvidenceScore:
    E: float
    factors: dict[str, float]
    experiment: dict[str, Any]
    paper: dict[str, Any]
    explanation: str
    funding_warning: bool = False
    tracer_warning: bool = False


def independent_replications(snapshot: GraphSnapshot, exp_key: str) -> int:
    """Bu deneyi REPLICATES ile bağlanan farklı laboratuvar sayısı."""
    labs = set()
    for e in snapshot.out_edges(exp_key, ["REPLICATES"]) + snapshot.in_edges(exp_key, ["REPLICATES"]):
        other = e.dst if e.src == exp_key else e.src
        for r in snapshot.in_edges(other, ["REPORTS"]):
            labs.add((snapshot.nodes.get(r.src, {}).get("lab") or r.src).lower())
    return len(labs)


def claim_contradicted(snapshot: GraphSnapshot, claim_key: str) -> bool:
    return any(e.type == "CONTRADICTS" for e in snapshot.in_edges(claim_key))


def evidence_strength(snapshot: GraphSnapshot, exp_key: str, claim_key: str, edge_props: dict[str, Any],
                      cfg: dict[str, Any] | None = None) -> EvidenceScore:
    exp = snapshot.nodes.get(exp_key, {})
    paper = next((snapshot.nodes.get(r.src, {}) for r in snapshot.in_edges(exp_key, ["REPORTS"])), {})
    labs = independent_replications(snapshot, exp_key)
    contradicted = claim_contradicted(snapshot, claim_key)
    c_extract = 1.0 if edge_props.get("verified_by_human") else float(edge_props.get("extraction_confidence") or 0.0)
    factors = {
        "w_design": w_design(exp.get("design"), cfg),
        "w_species": w_species(exp.get("species"), exp.get("design"), cfg),
        "w_n": w_n(exp.get("n_per_group"), cfg),
        "w_blind": w_blind(exp.get("blinded"), cfg),
        "w_repl": w_repl(labs, contradicted, cfg),
        "c_extract": c_extract,
    }
    E = 1.0
    for v in factors.values():
        E *= v
    return EvidenceScore(E=round(E, 4), factors=factors, experiment=exp, paper=paper,
                         explanation=describe_experiment(exp, paper, labs),
                         funding_warning=paper.get("funding_source") == "industry",
                         tracer_warning=exp.get("tracer_type") in ("NBD", "BODIPY"))


def describe_experiment(exp: dict[str, Any], paper: dict[str, Any] | None = None, n_labs: int = 0) -> str:
    """Kanıt düzeyini düz Türkçe söyler: 'Fig. 5h, fare, n = 9–10, körlemesiz, tek laboratuvar'."""
    parts = []
    fig = " ".join(p for p in (exp.get("figure"), exp.get("panel")) if p)
    if fig:
        parts.append(fig)
    species = exp.get("species")
    design = (exp.get("design") or "").lower()
    if design == "in vitro":
        parts.append("hücre kültürü" + (f" ({SPECIES_TR.get(species, species)})" if species else ""))
    elif design == "ex vivo":
        parts.append("ex vivo doku" + (f" ({SPECIES_TR.get(species, species)})" if species else ""))
    elif species:
        parts.append(SPECIES_TR.get(species, species))
    else:
        parts.append("tür belirtilmemiş")
    n = exp.get("n_per_group")
    if n:
        lo, hi = min(n), max(n)
        parts.append(f"n = {lo}" if lo == hi else f"n = {lo}–{hi}")
    else:
        parts.append("n belirtilmemiş")
    parts.append("körlemeli" if exp.get("blinded") else "körlemesiz veya belirtilmemiş")
    parts.append("tek laboratuvar" if n_labs == 0 else f"{n_labs + 1} bağımsız laboratuvar")
    if paper and paper.get("retracted"):
        parts.append("GERİ ÇEKİLMİŞ MAKALE")
    return ", ".join(parts)


# ---------------------------------------------------------------- zincirler


@dataclass
class Link:
    src: str
    predicate: str
    dst: str
    edge: Edge
    score: EvidenceScore
    inverse: bool = False

    def as_dict(self, snapshot: GraphSnapshot) -> dict[str, Any]:
        return {"from": self.src, "from_name": snapshot.name(self.src), "predicate": self.predicate,
                "to": self.dst, "to_name": snapshot.name(self.dst), "claim_id": self.edge.props.get("claim_id"),
                "claim": snapshot.nodes.get(self.edge.props.get("claim_id", ""), {}).get("text"),
                "source_doi": self.edge.props.get("source_doi"), "figure": self.edge.props.get("figure"),
                "source_sentence": self.edge.props.get("source_sentence"), "E": self.score.E,
                "evidence": self.score.explanation, "inverse": self.inverse}


def relation_score(snapshot: GraphSnapshot, edge: Edge, cfg: dict[str, Any] | None = None) -> EvidenceScore:
    claim_key = edge.props.get("claim_id", "")
    exp_key = edge.props.get("experiment_id")
    ev_props = dict(edge.props)
    if not exp_key:  # eski kayıtlar: claim'i destekleyen ilk deneyi kullan
        ev = snapshot.evidence_for_claim(claim_key)
        exp_key = ev[0][0].src if ev else ""
    return evidence_strength(snapshot, exp_key, claim_key, ev_props, cfg)


@dataclass
class ChainScore:
    C: float
    min_E: float
    penalties: dict[str, float] = field(default_factory=dict)
    weakest: Link | None = None


def direction_conflict(snapshot: GraphSnapshot, src: str, predicate: str, dst: str) -> bool:
    stances = {e.props.get("stance", "affirmed") for e in snapshot.out_edges(src, [predicate]) if e.dst == dst}
    opposite = {"INCREASES": "DECREASES", "DECREASES": "INCREASES"}.get(predicate)
    if opposite and any(e.dst == dst and e.props.get("stance", "affirmed") == "affirmed"
                        for e in snapshot.out_edges(src, [opposite])):
        return True
    return len(stances) > 1


def chain_confidence(snapshot: GraphSnapshot, links: list[Link], cfg: dict[str, Any] | None = None) -> ChainScore:
    c = _cfg(cfg)["chain"]
    pen = c["penalties"]
    penalties: dict[str, float] = {}
    species = [l.edge.props.get("species") for l in links]
    known = [s for s in species if s]
    if len(set(known)) > 1:
        penalties["species_mismatch"] = pen["species_mismatch"]
    cells = [l.edge.props.get("cell_type") for l in links if l.edge.props.get("cell_type")]
    if len(set(cells)) > 1:
        penalties["tissue_mismatch"] = pen["tissue_mismatch"]
    if any(l.edge.props.get("dose_known") is False for l in links):
        penalties["dose_unclear"] = pen["dose_unclear"]
    if any(direction_conflict(snapshot, l.edge.src, l.edge.type, l.edge.dst) for l in links):
        penalties["direction_conflict"] = pen["direction_conflict"]
    weakest = min(links, key=lambda l: l.score.E)
    C = weakest.score.E
    for p in penalties.values():
        C *= p
    C *= c["length_decay"] ** (len(links) - 1)
    return ChainScore(C=round(C, 4), min_E=weakest.score.E, penalties=penalties, weakest=weakest)


def final_score(C: float, P: float | None, cfg: dict[str, Any] | None = None) -> float:
    f = _cfg(cfg)["final"]
    return round(f["w_chain"] * C + f["w_embed"] * P, 4) if P is not None else round(C, 4)
