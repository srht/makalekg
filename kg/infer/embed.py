"""Bağlantı tahmini: PyKEEN (RotatE veya ComplEx) ile kanıt kenarları üzerinde embedding.

Model ASLA HYPOTHESIS kenarlarıyla eğitilmez: eğitim üçlüleri yalnızca
GraphSnapshot.evidence_relations() içindeki stance=affirmed kenarlardan gelir.
PyKEEN isteğe bağlıdır: `pip install -e .[embed]`.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from ..graph import GraphSnapshot
from .rules import Candidate, best_links, has_direct_evidence, shortest_evidence_path

log = logging.getLogger(__name__)


def training_triples(snapshot: GraphSnapshot) -> list[tuple[str, str, str]]:
    triples = {(e.src, e.type, e.dst) for e in snapshot.evidence_relations()
               if e.props.get("stance", "affirmed") == "affirmed"}
    return sorted(triples)


class EmbeddingPredictor:
    def __init__(self, model: str = "RotatE", epochs: int = 200, dim: int = 64, seed: int = 42):
        self.model_name, self.epochs, self.dim, self.seed = model, epochs, dim, seed
        self.model = None
        self.factory = None

    def fit(self, snapshot: GraphSnapshot) -> "EmbeddingPredictor":
        try:
            import numpy as np
            from pykeen.pipeline import pipeline
            from pykeen.triples import TriplesFactory
        except ImportError as exc:
            raise RuntimeError("PyKEEN kurulu değil: pip install -e '.[embed]'") from exc
        triples = training_triples(snapshot)
        if len(triples) < 10:
            raise RuntimeError(f"embedding için çok az kanıt üçlüsü var ({len(triples)})")
        self.factory = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))
        result = pipeline(training=self.factory, testing=self.factory, model=self.model_name,
                          model_kwargs={"embedding_dim": self.dim}, training_kwargs={"num_epochs": self.epochs},
                          random_seed=self.seed, evaluator_kwargs={"filtered": True})
        self.model = result.model
        return self

    def score(self, h: str, r: str, t: str) -> float | None:
        """0–1 aralığına sıkıştırılmış skor (sigmoid). Bilinmeyen varlık/ilişki → None."""
        if self.model is None or self.factory is None:
            return None
        e2id, r2id = self.factory.entity_to_id, self.factory.relation_to_id
        if h not in e2id or t not in e2id or r not in r2id:
            return None
        import torch

        batch = torch.tensor([[e2id[h], r2id[r], e2id[t]]])
        with torch.no_grad():
            raw = float(self.model.score_hrt(batch).item())
        return 1 / (1 + math.exp(-raw))

    def candidates(self, snapshot: GraphSnapshot, threshold: float = 0.8, top_k: int = 50) -> list[Candidate]:
        if self.model is None or self.factory is None:
            return []
        from pykeen.predict import predict_all

        pack = predict_all(model=self.model, k=top_k * 5)
        df = pack.process(factory=self.factory).df
        links = best_links(snapshot)
        out = []
        for row in df.itertuples():
            h, r, t = row.head_label, row.relation_label, row.tail_label
            P = 1 / (1 + math.exp(-float(row.score)))
            if P < threshold or h == t or has_direct_evidence(snapshot, h, t):
                continue
            path = shortest_evidence_path(snapshot, h, t, links=links)
            if not path:
                continue  # yolu olmayan aday raporlanamaz (en zayıf halka gösterilemez)
            out.append(Candidate(h, r, t, path, ["embed:" + self.model_name],
                                 f"{self.model_name} embedding skoru {P:.2f}; en kısa kanıt yolu üzerinden gösterildi.",
                                 embed_score=P))
            if len(out) >= top_k:
                break
        return out


def attach_embedding_scores(predictor: EmbeddingPredictor | None, candidates: list[Candidate]) -> None:
    if predictor is None:
        return
    for c in candidates:
        if c.embed_score is None:
            c.embed_score = predictor.score(c.subject, c.predicate, c.object)


def evidence_only(snapshot: GraphSnapshot) -> dict[str, Any]:
    """Teşhis: eğitim kümesinde HYPOTHESIS olmadığını gösterir."""
    triples = training_triples(snapshot)
    return {"n_triples": len(triples), "contains_hypothesis": any(r == "HYPOTHESIS" for _, r, _ in triples)}
